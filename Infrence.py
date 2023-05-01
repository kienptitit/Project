import os

from Phase1.model import Model
from Graph_AutoEncoder.edcoder import PreModel
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import roc_auc_score


def get_score_for_each_snippet(video_feature, model1, pre_model):
    """
    Your Video Have 15 snippets , this function compute score for each snippets
    :param video_feature:[15,98,1024]
    :return:Score (scaler)
    """
    video_feature = video_feature.unsqueeze(0)
    _, feature_phase1 = model1(video_feature)
    feature_phase1 = feature_phase1.squeeze(0)
    scores = []
    scaler = MinMaxScaler()
    for f in feature_phase1:
        score = pre_model.forward_test(f).detach().cpu().numpy()
        score = scaler.fit_transform(score)
        scores.append(score.max())
    return scores


def get_label(label_path):
    label = dict()
    with open(label_path, 'r') as f:
        cnt = 0
        for line in f:
            line = line.strip()
            if 'end' not in line:
                continue
            cnt += 1
            l = line.split(' = ')
            label[cnt] = l[1][:-1]
    return label


def get_score_for_all_frames(score, num_frame, length, time_steps):
    """
    :param score:
    :param num_frame:
    :return:
    """
    s = np.zeros(shape=(num_frame,))
    idx = list(range(0, num_frame - time_steps + 1, time_steps))
    r = np.linspace(0, len(idx), length + 1, dtype=np.int64) * time_steps
    for i in range(length):
        s[r[i]:r[i + 1]] = score[i]
    return s


def get_auc_score(original_feature_folder, feature_path, label, model1, pre_model, length, time_step):
    """
    :param feature_path:that is feature after swin and take avg to make 15 snippet
    :param label: label for each video
    :param model1:
    :param pre_model:
    :return:
    """
    all_auc = []
    for video_name in os.listdir(feature_path):
        if 'abnormal' not in video_name:
            continue
        frame_number = np.load(os.path.join(original_feature_folder, video_name)).shape[0]
        video_th = int(video_name.split('_')[1])
        gt = np.ones(shape=(frame_number,))
        gt[(label[video_th][0] - 1):(label[video_th][1] - 1)] = 0
        score_snippet = get_score_for_each_snippet(torch.load(os.path.join(feature_path, video_name)), model1,
                                                   pre_model)
        score_pred_all_frame = get_score_for_all_frames(score_snippet, frame_number, length, time_step)
        auc = roc_auc_score(gt, score_pred_all_frame)
        all_auc.append(auc)
    return all_auc


def load_all_model(model1_weight_path, preModel_weight_path):
    model1 = Model()
    pre_Model = PreModel(256, 128, 4, 4, 4, "prelu", 0.1, 0.1, 0.1, True, "layernorm", mask_rate=0.1, alpha_l=2)
    model1.load_state_dict(torch.load(model1_weight_path))
    pre_Model.load_state_dict(torch.load(preModel_weight_path))
    model1.eval()
    pre_Model.eval()
    return model1, pre_Model


if __name__ == '__main__':
    a = [torch.randn(1, 98, 1024), torch.randn(1, 98, 1024)]
    print(torch.vstack(a).shape)
# l = get_label(r"E:\UCSD_Anomaly_Dataset.v1p2\UCSDped2\Test\UCSDped2.m")
# print(l)

# print(torch.load(r"E:\Python test Work\HopingProject\Feature_swin3-4\abnormal_0_0.pt").shape)
