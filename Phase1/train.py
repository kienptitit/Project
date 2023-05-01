from utils import *
from model import *
import numpy as np
import matplotlib.pyplot as plt


def RTFM(out2_normal, out2_abnormal, out_normal, out_abnormal, margin=40):
    """
    :param out2_normal:[4,3,3,64]
    :param out2_abnormal:[4,3,3,64]
    :param out_normal:[4,15,98,256]
    :param out_abnormal:[4,15,98,256]
    :return:
    """
    # out2_normal_mag = out2_normal.norm(p=2, dim=-1).reshape(-1)
    # out2_abnormal_mag = out2_abnormal.norm(p=2, dim=-1).reshape(-1)
    # out_nomral_mag = out_normal.norm(p=2, dim=-1)
    # out_abnomral_mag = out_abnormal.norm(p=2, dim=-1)

    l_seperate = F.relu(margin - torch.norm((out2_normal - out2_abnormal), p=2, dim=-1)).mean()

    # l_smooth = ((out_nomral_mag[:, 1:, :] - out_nomral_mag[:, :-1, :]) ** 2).mean() + (
    #         (out_abnomral_mag[:, 1:, :] - out_abnomral_mag[:, :-1, :]) ** 2).mean()
    # l_sparse = out_abnomral_mag.mean()
    l_smooth = (torch.norm(out_normal[:, 1:, :, :] - out_normal[:, :-1, :, :], p=2, dim=-1).mean()
                + torch.norm(out_abnormal[:, 1:, :, :] - out_abnormal[:, :-1, :, :], p=2, dim=-1).mean())

    return l_seperate, l_seperate, l_smooth


def train(epoch, dataloader, model, optimizer, device):
    """
    :param epoch:
    :param dataloader: for each index with shape [4,30,98,1024]
    :param model:
    :param optimizer:
    :param device:
    :return:
    """
    model = model.to(device)
    losses = []
    l_seperates = 0.0
    l_smooths = 0.0
    l_sparses = 0.0
    for idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        data_normal = data[:, :15, :, :]
        data_abnormal = data[:, 15:, :, :]

        out2_normal, out_normal = model(data_normal.float())
        out2_abnormal, out_abnormal = model(data_abnormal.float())

        l, l_seperate, l_smooth = RTFM(out2_normal, out2_abnormal, out_normal, out_abnormal)
        l.backward()
        optimizer.step()
        losses.append(l.detach().cpu().numpy())
        l_seperates += l_seperate.detach().cpu().numpy()
        l_smooths += l_smooth.detach().cpu().numpy()
        print("Epoch {} idx {} Loss {:.4f}".format(epoch, idx, l.detach().cpu().numpy()))

    print("Epoch {} Avg_Loss {}".format(epoch, np.mean(losses)))
    return losses, l_seperates / len(dataloader), l_smooths / len(dataloader), l_sparses / len(dataloader)


def plot_loss(losses, loss_seperates, loss_smooth, loss_sparse):
    plt.plot(losses)
    plt.title("losses")
    plt.savefig(r'E:\Python test Work\HopingProject\Plot\loss.png')
    plt.show()
    plt.plot(loss_seperates)
    plt.title("seperates")
    plt.savefig(r'E:\Python test Work\HopingProject\Plot\seperates.png')
    plt.show()
    plt.plot(loss_smooth)
    plt.title("smooth")
    plt.savefig(r'E:\Python test Work\HopingProject\Plot\smooth.png')
    plt.show()
    plt.plot(loss_sparse)
    plt.savefig(r'E:\Python test Work\HopingProject\Plot\sparse.png')
    plt.title("sparse")
    plt.show()


if __name__ == '__main__':
    # GetData
    data = get_all_data2(r"E:\Python test Work\HopingProject\PED2\feature_snippet")
    my_data = Mydata(data)
    dataloader = DataLoader(my_data, batch_size=2, num_workers=2, shuffle=False)
    # Model Config
    model = Model()
    epochs = 30
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=3e-5)
    losses = []
    device = torch.device("cuda")
    losses_2 = []
    loss_seperate = []
    loss_smooth = []
    loss_sparse = []
    # Train
    print("Start Training")
    for epoch in range(epochs):
        model.train()
        l, l_seperate, l_smooth, l_sparse = train(epoch, dataloader, model, optimizer, device)
        losses.append(np.mean(l))
        losses_2.append(losses)
        loss_seperate.append(l_seperate)
        loss_smooth.append(l_smooth)
        loss_sparse.append(l_sparse)
    plot_loss(losses, loss_seperate, loss_smooth, loss_sparse)
    torch.save(model.state_dict(), r'E:\Python test Work\HopingProject\Weight\model.pt')
