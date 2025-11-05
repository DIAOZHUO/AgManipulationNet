import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class FFT_IZ_Classfier(nn.Module):
    def __init__(self):
        super(FFT_IZ_Classfier, self).__init__()
        self.relu = nn.LeakyReLU()
        self.signal_z_conv = self.conv2d_layer()
        self.signal_z_feature_dense = nn.Sequential(nn.Linear(16, 64),
                                                    nn.LeakyReLU(inplace=False),
                                                    nn.Dropout(0.3),
                                                    nn.Linear(64, 128)
                                                    )

        self.signal_i_conv = self.conv2d_layer()
        self.signal_i_feature_dense = nn.Sequential(nn.Linear(16, 64),
                                                    nn.LeakyReLU(inplace=False),
                                                    nn.Dropout(0.3),
                                                    nn.Linear(64, 128)
                                                    )

        self.combined_features = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 64),
            nn.Linear(64, 1),
            nn.LeakyReLU(inplace=False),
            nn.Sigmoid(),
        )

    def conv2d_layer(self):
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(8, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, Z, I):
        batch = Z.size(0)
        # print(Z.shape, I.shape)
        Z = self.signal_z_conv(Z).view(batch, -1)
        # Z = self.signal_z_feature_dense(Z)
        I = self.signal_i_conv(I).view(batch, -1)
        # I = self.signal_i_feature_dense(I)

        output = self.combined_features(Z)
        return output


class IZ_Classfier(nn.Module):
    def __init__(self):
        super(IZ_Classfier, self).__init__()
        self.relu = nn.LeakyReLU()
        self.signal_z_conv = self.conv1d_layer()
        self.signal_z_feature_dense = nn.Sequential(nn.Linear(3584, 256),
                                                    nn.LeakyReLU(inplace=False),
                                                    nn.Dropout(0.3),
                                                    nn.Linear(256, 128)
                                                    )

        self.signal_i_conv = self.conv1d_layer()
        self.signal_i_feature_dense = nn.Sequential(nn.Linear(3584, 256),
                                                    nn.LeakyReLU(inplace=False),
                                                    nn.Dropout(0.3),
                                                    nn.Linear(256, 128)
                                                    )

        self.combined_features = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=False),
            nn.Linear(256, 32),
            nn.Linear(32, 1),
            nn.ReLU(inplace=False),
            # nn.Sigmoid(),
        )

    def conv1d_layer(self):
        return nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(8, 32, kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.Conv1d(32, 64, kernel_size=7, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool1d(kernel_size=7, stride=2),
        )

    def forward(self, Z, I):
        batch = Z.size(0)
        # print(Z.shape, I.shape, Var.shape, batch)
        Z = self.signal_z_conv(Z).view(batch, -1)
        Z = self.signal_z_feature_dense(Z)
        # I = self.signal_i_conv(I).view(batch, -1)
        # I = self.signal_i_feature_dense(I)
        # print(Z.shape, Var.shape)
        # input = torch.cat((Z, I), 1)
        # input = torch.cat((Z.view(Z.size(0), -1), I.view(I.size(0), -1)), dim=1)

        output = self.combined_features(Z)
        return output


class Tipshaper_Classifier(nn.Module):
    def __init__(self):
        super(Tipshaper_Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.signal_z_conv = self.conv1d_layer()
        self.signal_z_feature_dense = nn.Sequential(nn.Linear(2560, 256),
                                                    nn.ReLU(inplace=False),
                                                    nn.Dropout(0.3),
                                                    nn.Linear(256, 128)
                                                    )

        self.signal_i_conv = self.conv1d_layer()
        self.signal_i_feature_dense = nn.Sequential(nn.Linear(2560, 256),
                                                    nn.LeakyReLU(inplace=False),
                                                    nn.Dropout(0.3),
                                                    nn.Linear(256, 128)
                                                    )

        self.numeric_features = nn.Sequential(
            nn.Linear(17, 32),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(32, 64),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.5)
        )

        self.combined_features = nn.Sequential(
            nn.Linear(320, 640),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(640, 320),
            nn.LeakyReLU(inplace=False),
            nn.Linear(320, 64),
            nn.Linear(64, 2),
            nn.LeakyReLU(inplace=False),
            nn.Sigmoid(),
        )

    def conv1d_layer(self):
        return nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(8, 32, kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.Conv1d(32, 64, kernel_size=7, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool1d(kernel_size=7, stride=2),
            nn.Conv1d(64, 128, kernel_size=9, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool1d(kernel_size=9, stride=2),
        )

    def forward(self, Z, I, Var):
        batch = Z.size(0)
        # print(Z.shape, I.shape, Var.shape, batch)
        Z = self.signal_z_conv(Z).view(batch, -1)
        Z = self.signal_z_feature_dense(Z)
        I = self.signal_i_conv(I).view(batch, -1)
        I = self.signal_i_feature_dense(I)
        Var = self.numeric_features(Var).view(Z.size(0), -1)
        # print(Z.shape, Var.shape)
        input = torch.cat((Z, I, Var), 1)

        # print(input.shape)
        output = self.combined_features(input)
        # output = self.relu(output),
        # output = F.sigmoid(output)
        return output



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Tipshaper_Classifier()
    # model.to(device=device)
    # summary(model, [(1, 512), (1, 512), (1, 17)])

    model = IZ_Classfier()
    model.to(device=device)
    summary(model, [(1, 512), (1, 512)])

    # model = FFT_IZ_Classfier()
    # model.to(device=device)
    # summary(model, [(1, 16, 16), (1, 16, 16)])

