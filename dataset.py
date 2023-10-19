import numpy as np
import torch
from torch.utils.data import Dataset


def equal_samples(data_normal, data_abnormal):
    N = len(data_normal)
    A = len(data_abnormal)
    if N > A:
        c = N // A
        r = max(0, N % A)
        data_abnormal = data_abnormal.repeat(c, 1, 1)
        random_idx = torch.randint(0, A, (r,))
        data_abnormal = torch.concat([data_abnormal, data_abnormal[random_idx]], dim=0)  # [N,16,1024]
    else:
        c = A // N
        r = max(0, A % N)
        data_normal = data_normal.repeat(c, 1, 1)
        random_idx = torch.randint(0, N, (r,))
        data_normal = torch.concat([data_normal, data_normal[random_idx]], dim=0)  # [N,16,1024]
    return data_normal, data_abnormal


class MyDataset(Dataset):
    def __init__(self, data, labels, mode='train'):
        super().__init__()
        self.data = data
        self.mode = mode

        if self.mode == 'train':
            self.data_normal, self.data_abnormal = equal_samples(data[labels == 0], data[labels != 0])

    def __len__(self):
        if self.mode == 'train':
            return len(self.data_normal)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            normal = self.data_normal[idx]
            abnormal = self.data_abnormal[idx]
            d = torch.concat([normal, abnormal], dim=0)
            return d
        else:
            return self.data[idx]


if __name__ == '__main__':
    X_train = torch.from_numpy(np.load(r"E:\2023\NaverProject\LastCodingProject\Binary_file\X_train.npy")).reshape(-1,
                                                                                                                   16,
                                                                                                                   1024)
    label_train = torch.load(r"E:\2023\NaverProject\LastCodingProject\Binary_file\label_train.pt")
    mydata = MyDataset(X_train, label_train)
    print(len(mydata))
