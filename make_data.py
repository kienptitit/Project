import os
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from video_swin.video_swin_transformer import SwinTransformer3D
import torch
from collections import OrderedDict
from torchvision import transforms
from Phase1.utils import make_feature


def np_load_frame(filename, resize_height, resize_width):
    img_decoded = cv2.imread(filename)
    img_resized = cv2.resize(img_decoded, (resize_height, resize_width))
    img_resized = img_resized.astype(np.float32)
    image_resized = (img_resized / 127.5) - 1.0
    return image_resized


class Frame_Loader(Dataset):
    def __init__(self, np_frame_videos, transforms=None, resize_height=224, resize_width=224,
                 time_steps=4):
        self.np_frame_videos = np_frame_videos
        self.transforms = transforms
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.num_frames = len(np_frame_videos)
        self.index_sample = range(0, self.num_frames - time_steps + 1, time_steps)
        self.time_steps = time_steps

    def __getitem__(self, idx):
        frame_index = self.index_sample[idx]
        batch_frame = np.zeros(
            (self.time_steps, 3, self.resize_width, self.resize_height))
        for i in range(self.time_steps):
            frame = self.np_frame_videos[frame_index + i]
            if self.transforms:
                frame = self.transforms(frame)
            batch_frame[i] = frame
        return batch_frame

    def __len__(self):
        return len(self.index_sample)


def get_path(normal_folder, anomaly_folder):
    normal_video = [os.path.join(normal_folder, p) for p in os.listdir(normal_folder) if "Train" in p]
    anomaly_video = [os.path.join(anomaly_folder, p) for p in os.listdir(anomaly_folder) if
                     "Test" in p and "_gt" not in p]
    return normal_video, anomaly_video


def convert_to_numpy(Video_path, resize_height, resize_width):
    list_frames = []
    for frame in os.listdir(Video_path):
        if '.tif' in frame:
            frame_path = os.path.join(Video_path, frame)
            list_frames.append(np_load_frame(frame_path, resize_height, resize_width))
    return np.stack(list_frames)


def save_np_video(data_name, video_name, list_frame):
    folder_name = os.path.join(data_name, "_original")
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    np.save(os.path.join(folder_name, video_name), list_frame)


def make_faeture_swin(data_name):
    if not os.path.exists(os.path.join(data_name, "_swin")):
        os.mkdir(os.path.join(data_name, "_swin"))
    swin_folder = os.path.join(data_name, "_swin")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTransformer3D(embed_dim=128,
                              depths=[2, 2, 18, 2],
                              num_heads=[4, 8, 16, 32],
                              patch_size=(2, 4, 4),
                              window_size=(16, 7, 7),
                              drop_path_rate=0.4,
                              patch_norm=True)
    checkpoint = torch.load(r"E:\Python test Work\HopingProject\Weight\swin_base_patch244_window1677_sthv2.pth")
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'backbone' in k:
            name = k[9:]
            new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    del model.layers[2].blocks[:17]
    del model.layers[0].blocks[0]
    del model.layers[1].blocks[0]
    del model.layers[3].blocks[0]
    ori_folder = os.path.join(data_name, '_original')
    model = model.to(device)

    for video_name in os.listdir(ori_folder):
        video_feat = np.load(os.path.join(ori_folder, video_name))
        loader = Frame_Loader(video_feat, transforms=transforms.Compose([
            transforms.ToTensor()
        ]))
        dataloader = DataLoader(loader, batch_size=1, shuffle=False, num_workers=4)
        torch.cuda.empty_cache()
        feat = []
        for x in dataloader:
            x = x.to(device)
            x = x.transpose(2, 1)
            out = model(x.float())
            out = out.reshape(out.shape[0], out.shape[1], -1).transpose(2, 1)
            feat.append(out)
        feat = torch.stack(feat, dim=0)
        torch.save(feat, os.path.join(swin_folder, video_name.split('.')[0] + '.pt'))
        print("Done!!!")


if __name__ == '__main__':
    # tmp = np.random.randn(180, 3, 224, 224)
    # Data = Frame_Loader(tmp)
    # print(len(Data))
    # Config
    data_name = "PED2"
    normal_folder = r"E:\UCSD_Anomaly_Dataset.v1p2\UCSDped2\Train"
    abnormal_folder = r"E:\UCSD_Anomaly_Dataset.v1p2\UCSDped2\Test"
    normal_video, abnormal_video = get_path(normal_folder, abnormal_folder)
    resize_weight, resize_height = 224, 224
    # Make_data
    if not os.path.exists(data_name):
        os.mkdir(data_name)
        for x in tqdm(abnormal_video):
            base_name = os.path.basename(x)
            new_name = "abnormal_" + base_name[-3:]
            save_np_video(data_name, new_name, convert_to_numpy(x, 224, 224))
        for x in tqdm(normal_video):
            base_name = os.path.basename(x)
            new_name = "normal_" + base_name[-3:]
            save_np_video(data_name, new_name, convert_to_numpy(x, 224, 224))
    # make_faeture_swin(data_name)
    make_feature(r"E:\Python test Work\HopingProject\PED2\_swin",
                 r"E:\Python test Work\HopingProject\PED2\feature_snippet")
