import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, in_features, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=in_features, out_features=in_features//reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=in_features//reduction, out_features=in_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        # print("x.shape : ", x.shape)
        n,c,_,_ = x.size()
        x = x.view(n, c)
        # print("x.shape : ", x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(n, c, 1, 1)
        # print("x.shape : ", x.shape)  # x.shape :  torch.Size([10, 64])
        # print("identity.shape : ", identity.shape) # identity.shape :  torch.Size([10, 64, 56, 56])
        x = identity * x

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.channel_equal_flag = True
        if in_channels == out_channels:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1 ,stride=2, bias=False)
            self.bn1x1 = nn.BatchNorm2d(num_features=out_channels)
            self.channel_equal_flag = False

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.selayer = SELayer(in_features=out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.channel_equal_flag == True:
            pass
        else:
            identity = self.conv1x1(identity)
            identity = self.bn1x1(identity)
            identity = self.relu(identity)

        x = self.selayer(x)

        out = identity + x

        return out


class ResNet18_se_My(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18_se_My, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        #conv2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = ResidualBlock(in_channels=64, out_channels=64)
        self.conv2_2 = ResidualBlock(in_channels=64, out_channels=64)

        # conv3
        self.conv3_1 = ResidualBlock(in_channels=64, out_channels=128)
        self.conv3_2 = ResidualBlock(in_channels=128, out_channels=128)

        # conv4_x
        self.conv4_1 = ResidualBlock(in_channels=128, out_channels=256)
        self.conv4_2 = ResidualBlock(in_channels=256, out_channels=256)

        # conv5_x
        self.conv5_1 = ResidualBlock(in_channels=256, out_channels=512)
        self.conv5_2 = ResidualBlock(in_channels=512, out_channels=512)

        # avg_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # fc
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

        # softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # conv2
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        # conv3_x
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        # conv4_x
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        # conv5_x
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        # avgpool + fc + softmax
        # x = self.avg_pool(x)
        # x = x.view(x.size(0), -1)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        # x = self.softmax(x)
        return x


if __name__ == '__main__':
    # [N, C, H, W]  non-singleton dimension 3
    input = torch.randn(10, 3, 224, 224)
    model = ResNet18_se_My(num_classes=7)
    output = model(input)
    print(output.shape)