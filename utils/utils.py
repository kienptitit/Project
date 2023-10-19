import torch
from config import CFG
from models.fc_flow import *
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

_GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))


def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5 * torch.sum(z ** 2, 1) + logdet_J
    return logp


def get_flow_model(args: CFG, in_channels):
    if args.flow_arch == 'flow_model':
        model = flow_model(args, in_channels)
    elif args.flow_arch == 'conditional_flow_model':
        model = conditional_flow_model(args, in_channels)
    else:
        raise NotImplementedError('{} is not supported Normalizing Flow!'.format(args.flow_arch))

    return model


def save_visualization(logps, labels, name_fig, boundaries):
    """
    :param logps: [Batch_size * 32]
    :param labels: [Batch_size * 32]
    :return:
    """
    logp_normal = logps[labels == 0]
    logp_abnormal = logps[labels != 0]
    #
    # n_idx = int(len(logp_normal) * 0.4)
    # sorted_indices = torch.sort(logp_normal)[1]
    # n_idx = sorted_indices[n_idx]
    # b_n = logp_normal[n_idx]
    plt.figure()
    sns.distplot(logp_normal.detach().cpu().numpy(), label='normal')
    sns.distplot(logp_abnormal.detach().cpu().numpy(), label='abnormal')
    plt.axvline(boundaries[0].detach().cpu() * 10, color='red', linestyle='--')
    plt.axvline(boundaries[1].detach().cpu() * 10, color='red', linestyle='--')
    if boundaries[-1] != 0:
        plt.axvline(boundaries[-1].detach().cpu() * 10, color='red', linestyle='--')
    plt.legend()
    if name_fig is not None:
        plt.savefig(name_fig)
    else:
        plt.show()
    plt.close()


def convert_to_anomaly_scores(args, logps_list, get_train_normal_score=False, n=None):
    if isinstance(logps_list, list):
        logps = torch.cat(logps_list)  # [290,16]
    else:
        logps = logps_list  # [810,64]

    logps -= torch.max(logps)  # -inf 0
    scores = torch.exp(logps)  # 0 1

    scores = scores.max() - scores
    if get_train_normal_score:
        scores = scores.reshape(n, -1)
        if scores.shape[1] != 64:
            scores = scores.reshape(-1)
        else:
            scores = scores[:, :32].reshape(-1)
    return scores


import flowtorch.bijectors as bij
import flowtorch.distributions as dist

torch.distributions.normal.Normal(loc=0, scale=1),
dist_x = torch.distributions.Independent(
    torch.distributions.Normal(torch.zeros(1), torch.ones(1)),
    1
)
bijector = bij.Exp()
dist.Flow(dist_x,bijector)