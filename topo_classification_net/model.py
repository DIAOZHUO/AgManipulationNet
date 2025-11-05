import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()
        self.num_classes = num_classes
        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64, dropout_p=0.1)
        self.layer4 = self.conv_module(64, 128, dropout_p=0.2)
        self.layer5 = self.conv_module(128, 256, dropout_p=0.2)
        self.gap = self.global_avg_pool(256, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, self.num_classes)

        return out

    def conv_module(self, in_num, out_num, dropout_p=0.0):
        if dropout_p == 0.0:
            return nn.Sequential(
                nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_num),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            return nn.Sequential(
                nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_num),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=dropout_p, inplace=True))


    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Tipshaper_Classifier()
    # model.to(device=device)
    # summary(model, [(1, 512), (1, 512), (1, 17)])

    model = CustomConvNet(11)
    model.to(device=device)
    summary(model, [(3, 256, 256)])

