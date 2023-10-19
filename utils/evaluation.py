import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class Boundary_recoder:
    def __init__(self):
        self.boundary = None
        self.epoch = 0
        self.history = []

    def update(self, boundary):
        self.boundary = boundary
        self.history.append(boundary)


class MetricRecoder:
    def __init__(self, mode='train'):
        self.loss = []
        self.mode = mode
        if mode == 'test':
            self.roc_auc_test_loss = []
        self.epoch = -1
        self.sub_epoch = -1
        self.score_normal = None
        self.boundary = []

    def update(self, **kwargs):
        if self.mode == 'train':
            self.loss.append(kwargs['loss'])
            self.epoch = kwargs['epoch']
            self.sub_epoch = kwargs['sub_epoch']
            self.score_normal = kwargs["score_normal"]
            if kwargs['boundary'] is not None:
                self.boundary.append(np.array(kwargs['boundary']))
        else:
            self.loss.append(kwargs['loss'])
            self.roc_auc_test_loss.append((kwargs['roc_auc']))
            self.epoch = kwargs['epoch']

    def __repr__(self):
        if self.mode == 'train':
            return f"Epoch {self.epoch} Sub_Epoch {self.sub_epoch} Loss_train {str(self.loss[-1])} Normal Score {self.score_normal}"
        else:
            return f"Epoch {self.epoch} Loss_test {self.loss[-1]} ROC_AUC {self.roc_auc_test_loss[-1]}"


def postpress(curve, seg_size=32):
    leng = curve.shape[0]
    window_size = leng // seg_size
    new_curve = np.zeros_like(curve)
    for i in range(seg_size):
        new_curve[window_size * i:window_size * (i + 1)] = np.mean(curve[window_size * i:window_size * (i + 1)])
    if leng > window_size * seg_size:
        new_curve[seg_size * window_size:] = np.mean(curve[seg_size * window_size:])
    return new_curve


def evaluate_result(score, label_path):
    videos = {}
    with open(label_path, 'r') as f:
        for idx, line in enumerate(f):
            video_len = int(line.strip().split(' ')[1])
            sub_video_gt = np.zeros((video_len,), dtype=np.int8)
            anomaly_tuple = line.split(' ')[3:]
            for ind in range(len(anomaly_tuple) // 2):
                start = int(anomaly_tuple[2 * ind])
                end = int(anomaly_tuple[2 * ind + 1])
                if start > 0:
                    sub_video_gt[start:end] = 1
            videos[idx] = sub_video_gt

    GT = []
    ANS = []

    GT_matrix = []
    ANS_matrix = []

    for vid in videos:
        cur_ab = score[vid]
        cur_gt = videos[vid]
        ratio = float(len(cur_gt)) / float(len(cur_ab))
        cur_ans = np.zeros_like(cur_gt, dtype='float32')
        for i in range(len(cur_ab)):
            b = int(i * ratio + 0.5)
            e = int((i + 1) * ratio + 0.5)
            cur_ans[b: e] = cur_ab[i]
        cur_ans = postpress(cur_ans, seg_size=64)
        GT_matrix.append(cur_gt.tolist())
        ANS_matrix.append(cur_ans.tolist())
        GT.extend(cur_gt.tolist())
        ANS.extend(cur_ans.tolist())
    # for i, (gt, ans) in enumerate(tqdm(zip(GT_matrix, ANS_matrix))):
    #     plt.figure()
    #     plt.plot(gt, color='blue')
    #     plt.plot(ans, color='red')
    #     plt.savefig(os.path.join('Result', f"{i}.png"))
    #     plt.close()
    return roc_auc_score(GT, ANS)
