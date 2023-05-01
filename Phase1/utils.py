import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random


def clear(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def process_feature(feat, length):
    """
    :param feat:[44,98,1024]
    :param length:15
    :return:
    """
    new_feature = np.zeros((length, feat.shape[1], feat.shape[2]))
    r = np.linspace(0, feat.shape[0], length + 1, dtype=np.int64)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feature[i] = np.mean(feat[r[i]:r[i + 1]], axis=0)
        else:
            new_feature[i] = feat[r[i]]
    return new_feature


def make_feature(folder_path, new_folder_path):
    for file_path in os.listdir(folder_path):
        feature = torch.load(os.path.join(folder_path, file_path)).detach().cpu().numpy()
        new_feature = process_feature(feature, 15)
        f = file_path.split('.')[0]
        np.save(os.path.join(new_folder_path, f + '.npy'), new_feature)


def get_all_data(folder_path):
    normal_path_list = []
    abnormal_path_list = []

    for file_name in os.listdir(folder_path):
        if "abnormal" not in file_name:
            normal_path_list.append(file_name)
        else:
            abnormal_path_list.append(file_name)

    normal_path_list.sort(key=lambda x: (int(x.split('.')[0].split('_')[1]), int(x.split('.')[0].split('_')[2])))
    abnormal_path_list.sort(key=lambda x: (int(x.split('.')[0].split('_')[1]), int(x.split('.')[0].split('_')[2])))

    res = []
    for idx, (normal, abnormal) in enumerate(zip(normal_path_list, abnormal_path_list)):
        normal = np.load(os.path.join(folder_path, normal))
        abnormal = np.load(os.path.join(folder_path, abnormal))
        feat = np.concatenate((normal, abnormal), axis=0)
        res.append(feat)
    return np.stack(res)


def get_all_data2(folder_path):
    normal_path_list = []
    abnormal_path_list = []

    for file_name in os.listdir(folder_path):
        if "abnormal" not in file_name:
            normal_path_list.append(os.path.join(folder_path, file_name))
        else:
            abnormal_path_list.append(os.path.join(folder_path, file_name))
    res = []
    if len(normal_path_list) > len(abnormal_path_list):
        for i in range(len(abnormal_path_list)):
            normal_feat = np.load(normal_path_list[i])
            abnormal_feat = np.load(abnormal_path_list[i])
            feat = np.concatenate((normal_feat, abnormal_feat), axis=0)
            res.append(feat)
        for i in range(len(normal_path_list) - len(abnormal_path_list)):
            normal_feat = np.load(normal_path_list[len(abnormal_path_list) + i])
            abnormal_feat = np.load(abnormal_path_list[i])
            feat = np.concatenate((normal_feat, abnormal_feat), axis=0)
            res.append(feat)
    else:
        for i in range(len(normal_path_list)):
            normal_feat = np.load(normal_path_list[i])
            abnormal_feat = np.load(abnormal_path_list[i])
            feat = np.concatenate((normal_feat, abnormal_feat), axis=0)
            res.append(feat)
        for i in range(len(abnormal_path_list) - len(normal_path_list)):
            normal_feat = np.load(normal_path_list[i])
            abnormal_feat = np.load(abnormal_path_list[len(normal_path_list) + i])
            feat = np.concatenate((normal_feat, abnormal_feat), axis=0)
            res.append(feat)
    return np.stack(res)


class Mydata(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    print(get_all_data2(r"E:\Python test Work\HopingProject\PED2\feature_snippet").shape)
