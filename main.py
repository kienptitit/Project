import pickle

import torch.optim
import os
from utils.utils import *
from utils.evaluation import *
from utils.model_utils import *
import numpy as np
from config import CFG
from torch.utils.data import DataLoader
from models.modules import PositionalEncoding1D
from Loss.losses import *
from dataset import MyDataset
import warnings
from utils.evaluation import evaluate_result
import random

warnings.filterwarnings("ignore")

torch.manual_seed(22)
torch.cuda.manual_seed(22)
np.random.seed(22)
random.seed(22)

log_theta = torch.nn.LogSigmoid()


def get_dataloader(args: CFG):
    data_train = torch.from_numpy(np.load(args.train_path)).reshape(-1, args.snippets, 1024)

    label_train = torch.load(args.label_train_path)
    dataset_train = MyDataset(data_train, label_train)
    train_loader = DataLoader(dataset_train, batch_size=CFG.Batch_size, shuffle=True)

    data_test = torch.from_numpy(np.load(args.test_path)).reshape(-1, args.snippets, 1024)

    label_test = None
    dataset_test = MyDataset(data_test, label_test, mode='test')
    test_loader = DataLoader(dataset_test, batch_size=CFG.Batch_size, shuffle=False)

    return train_loader, test_loader


def train_meta_epoch(args: CFG, epoch, trainloader, normalizing_flow, optimizer, POS_EMB: PositionalEncoding1D,
                     metric_recoder: MetricRecoder):
    """
    :param args:
    :param epoch:
    :param trainloader: [Batch_size , 32,1024]
    :param normalizing_flow:
    :param optimizer:
    :return:
    """
    normalizing_flow.to(args.device)
    normalizing_flow.train()
    adjust_learning_rate(args, optimizer, epoch)
    I = len(trainloader)
    for sub_epoch in range(args.sub_epochs):
        total_loss, loss_count = 0.0, 0
        loss_anomaly_boundary, loss_nomral_boundary = 0.0, 0.0
        logps_list = []
        boundaries_list = []
        for (i, loader) in enumerate(trainloader):
            # [Batch_size,64,1024]
            # lr = warmup_learning_rate(args, epoch, i + sub_epoch * I, I * args.sub_epochs, optimizer)

            m_b = torch.hstack([torch.zeros(loader.shape[1] // 2), torch.ones(loader.shape[1] // 2)]).unsqueeze(
                0).repeat(loader.shape[0], 1)  # [B,Snippet * 2]

            if epoch == 0 or sub_epoch < args.normal_sub_epoch:  # Normal only
                loader = loader[:, :args.snippets, :]
                m_b = m_b[:, :args.snippets]
            b, n, c = loader.shape
            loader = loader.reshape(-1, c)  # [B * Snippet , 1024]
            m_b = m_b.reshape(-1)  # [B * Snippet]
            pos_embed = POS_EMB(torch.rand(b, 32, c)).reshape(-1, args.pos_embed_dim)  # [B * Snippet , 128]

            e_b = loader.clone()
            m_b = m_b
            p_b = pos_embed

            if n == args.snippets:
                e_b = e_b.to(args.device)
                p_b = p_b.to(args.device)
                if args.flow_arch == 'flow_model':
                    z, log_jac_det = normalizing_flow(e_b)  # [4*16,1024] , [4*16]
                else:
                    z, log_jac_det = normalizing_flow(e_b, [p_b, ])
            else:
                if args.flow_arch == 'flow_model':
                    z, log_jac_det = normalizing_flow(e_b)
                else:

                    e_b_normal = e_b[m_b == 0]
                    e_b_abnormal = e_b[m_b != 0]

                    p_b = p_b.to(args.device)
                    e_b_normal, e_b_abnormal = e_b_normal.to(args.device), e_b_abnormal.to(args.device)

                    z_normal, log_jac_det_normal = normalizing_flow(e_b_normal, [p_b, ])
                    z_abnormal, log_jac_det_abnormal = normalizing_flow(e_b_abnormal, [p_b, ])

                    z = torch.concat([z_normal, z_abnormal], dim=0)
                    log_jac_det = torch.hstack([log_jac_det_normal, log_jac_det_abnormal])

                    m_b = torch.hstack([torch.zeros(e_b_normal.shape[0]), torch.ones(e_b_abnormal.shape[0])])

            if epoch == 0:
                logps = get_logp(c, z, log_jac_det) / c  # [4*16]

                loss = -log_theta(logps).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loss_count += 1
            else:
                if sub_epoch < args.normal_sub_epoch:
                    logps = get_logp(c, z, log_jac_det)  # Batch_size * 16
                    logps = logps / c
                    if args.focal_weighting:
                        normal_weights = normal_fl_weighting(logps.detach())

                        loss = -log_theta(logps) * normal_weights
                        loss = loss.mean()

                    else:
                        loss = -log_theta(logps).mean()
                else:
                    logps = get_logp(c, z, log_jac_det)

                    logps = logps / c
                    if args.focal_weighting:
                        logps_detach = logps.detach()
                        normal_logps = logps_detach[m_b == 0]
                        anomaly_logps = logps_detach[m_b == 1]
                        nor_weights = normal_fl_weighting(normal_logps)
                        ano_weights = abnormal_fl_weighting(anomaly_logps)
                        weights = nor_weights.new_zeros(logps_detach.shape)
                        weights[m_b == 0] = nor_weights
                        weights[m_b == 1] = ano_weights
                        loss_ml = -log_theta(logps[m_b == 0]) * nor_weights  # (256, )
                        loss_ml = torch.mean(loss_ml)
                    else:

                        loss_ml = -log_theta(logps[m_b == 0])
                        loss_ml = torch.mean(loss_ml)

                    boundaries = get_logp_boundary(logps, m_b, args.pos_beta, args.margin_abnormal_negative,
                                                   args.margin_abnormal_positive, args.normalizer)

                    boundaries_list.append([x.detach().cpu().item() for x in boundaries])
                    # print(boundaries)  # b_n,b_a_negative,b_a_positive

                    if args.focal_weighting:
                        loss_n_con, loss_a_con_pos, loss_a_con_neg = calculate_bg_spp_loss(logps, m_b, boundaries,
                                                                                           args.normalizer,
                                                                                           weights=weights,
                                                                                           mode=args.mode_loss)
                    else:
                        loss_n_con, loss_a_con_pos, loss_a_con_neg = calculate_bg_spp_loss(logps, m_b, boundaries,
                                                                                           args.normalizer,
                                                                                           mode=args.mode_loss)
                    # print(f"Loss_ml {loss_ml}, loss_n_con {loss_n_con}, loss_a_con {loss_a_con_pos}")
                    loss_nomral_boundary += loss_n_con
                    loss_anomaly_boundary += loss_a_con_pos

                    loss = loss_ml + args.bgspp_lambda * (
                            loss_n_con + loss_a_con_pos + loss_a_con_neg)

                    if args.save_boundary:
                        if not os.path.exists(os.path.join(args.result_path, 'Epoch_' + str(epoch))):
                            os.mkdir(os.path.join(args.result_path, 'Epoch_' + str(epoch)))
                        epoch_path = os.path.join(args.result_path, 'Epoch_' + str(epoch))

                        if not os.path.exists(os.path.join(epoch_path, 'SubEpoch_' + str(sub_epoch))):
                            os.mkdir(os.path.join(epoch_path, 'SubEpoch_' + str(sub_epoch)))
                        sub_epoch_path = os.path.join(epoch_path, 'SubEpoch_' + str(sub_epoch))
                        save_visualization(logps, m_b, os.path.join(sub_epoch_path, f'{i}.png'), boundaries)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if math.isnan(loss_item):
                    total_loss += 0.0
                    loss_count += 0
                else:
                    total_loss += loss.item()
                    loss_count += 1
            logps_list.append(logps)
        # print(f"Debug {total_loss} {loss_count}")
        score_normal = convert_to_anomaly_scores(args, logps_list, get_train_normal_score=True,
                                                 n=len(trainloader.dataset))
        if len(boundaries_list) == 0:
            metric_recoder.update(epoch=epoch, sub_epoch=sub_epoch,
                                  loss=total_loss / loss_count if loss_count != 0 else -1,
                                  score_normal=score_normal.mean().item(), boundary=None)
        else:
            metric_recoder.update(epoch=epoch, sub_epoch=sub_epoch,
                                  loss=total_loss / loss_count if loss_count != 0 else -1,
                                  score_normal=score_normal.mean().item(), boundary=boundaries_list)
        print(metric_recoder, loss_nomral_boundary / len(trainloader) if loss_nomral_boundary != 0 else '',
              loss_anomaly_boundary / len(trainloader) if loss_anomaly_boundary != 0 else '')


def validate(args: CFG, epoch, data_loader, normalizing_flow, POS_EMBED, metric_recoder: MetricRecoder):
    print("Compute loss and scores")
    normalizing_flow.eval()
    total_loss, loss_count = 0.0, 0
    logps_list = []

    with torch.no_grad():
        for i, feature in enumerate(data_loader):
            b, n, dim = feature.shape
            feature = feature.to(args.device).reshape(-1, 1024)
            pos_embed = POS_EMBED(feature.reshape(b, n, dim)).reshape(-1, args.pos_embed_dim).to(args.device)
            if args.flow_arch == 'flow_model':
                z, log_jac_det = normalizing_flow(feature)
            else:
                z, log_jac_det = normalizing_flow(feature, [pos_embed, ])
            logps = get_logp(dim, z, log_jac_det)

            logps = logps / dim
            loss = -log_theta(logps).mean()
            total_loss += loss.item()
            loss_count += 1
            logps_list.append(logps.reshape(b, n))

        mean_loss = total_loss / loss_count
        scores = convert_to_anomaly_scores(args, logps_list).detach().cpu().numpy()

        roc_auc = evaluate_result(scores, args.label_test_path)
        metric_recoder.update(loss=mean_loss, roc_auc=roc_auc, epoch=epoch)
        print(metric_recoder)
    return mean_loss, roc_auc


def train(args: CFG):
    trainloader, test_loader = get_dataloader(args)  # Correct

    normalizing_flow = get_flow_model(args, 1024)  # Correct
    optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=args.lr)  # Correct
    pos_embed = PositionalEncoding1D(args.pos_embed_dim)  # Correct
    train_recoder = MetricRecoder(mode='train')  # Correct
    test_recoder = MetricRecoder(mode='test')  # Correct
    normalizing_flow = normalizing_flow.to(args.device)
    # validate(args, -1, test_loader, normalizing_flow, pos_embed, test_recoder)
    start_epoch = 0
    if args.continue_training:
        trained_infos = torch.load(os.path.join(args.log_path, args.model_saved_path))
        start_epoch = trained_infos['epoch']
        normalizing_flow.load_state_dict(trained_infos['model'])

    for epoch in range(start_epoch, args.num_epochs):

        train_meta_epoch(args, epoch, trainloader, normalizing_flow, optimizer, pos_embed,
                         metric_recoder=train_recoder)
        validate(args, epoch, test_loader, normalizing_flow, pos_embed, test_recoder)
        if args.save_result:
            train_infos = {
                'model': normalizing_flow.state_dict(),
                'epoch': epoch
            }
            torch.save(train_infos,
                       os.path.join(args.log_path, args.model_saved_path))
            with open(os.path.join(args.log_path, args.record_train_saved), "wb") as f:
                pickle.dump(train_recoder, f)
            with open(os.path.join(args.log_path, args.record_test_saved), "wb") as f:
                pickle.dump(test_recoder, f)


# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# def check_result(args, model, checkpoint):
#     model.load_state_dict(torch.load(checkpoint))
#     model.to(args.device)
#     model.eval()
#     _, test_loader = get_dataloader(args)
#     pos_embed = PositionalEncoding1D(args.pos_embed_dim)
#     test_recoder = MetricRecoder(mode='test')
#     _, _, res = validate(args, 0, test_loader, model, pos_embed, test_recoder)
#
#     scores = convert_to_anomaly_scores(args, res).detach().cpu().numpy()
#     evaluate_result(scores, args.label_test_path)


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    #
    # plt.savefig('test.png')
    args = CFG()
    train(args)
