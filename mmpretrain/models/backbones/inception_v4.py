import torch
import torch.nn as nn

from mmpretrain.registry import MODELS

## 自定义inceptionv2网络结构



class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=(1,1),padding=(0,0)):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StemModel(nn.Module):
    def __init__(self):
        super(StemModel, self).__init__()
        self.conv_1 = BasicConv(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv_2 = BasicConv(in_channels=32, out_channels=32, kernel_size=3)
        self.conv_3 = BasicConv(in_channels=32, out_channels=64, kernel_size=3,padding=(1,1))

        self.branch_1_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch_1_2 = BasicConv(in_channels=64, out_channels=96, kernel_size=3, stride=2)

        self.branch_2_1 = nn.Sequential(
            BasicConv(in_channels=160, out_channels=64, kernel_size=1),
            BasicConv(in_channels=64, out_channels=96, kernel_size=3)
        )

        self.branch_2_2 = nn.Sequential(
            BasicConv(in_channels=160, out_channels=64, kernel_size=1),
            BasicConv(in_channels=64, out_channels=64, kernel_size=(7,1), padding=(3,0)),
            BasicConv(in_channels=64, out_channels=64, kernel_size=(1,7), padding=(0,3)),
            BasicConv(in_channels=64, out_channels=96, kernel_size=3)
        )

        self.branch_3_1 = BasicConv(in_channels=192, out_channels=192, kernel_size=3, stride=2)
        self.branch_3_2 = nn.MaxPool2d(kernel_size=3, stride=2)



    def forward(self,x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        x_1 = self.branch_1_1(x)
        x_2 = self.branch_1_2(x)

        x = torch.cat([x_1, x_2], dim=1)

        x_1 = self.branch_2_1(x)
        x_2 = self.branch_2_2(x)



        x = torch.cat([x_1, x_2], dim = 1)

        x_1 = self.branch_3_1(x)
        x_2 = self.branch_3_2(x)

        # print(x_1.shape)
        # print(x_2.shape)


        x = torch.cat([x_1, x_2], dim=1)

        return x


class Inception_A_Model(nn.Module):
    def __init__(self):
        super(Inception_A_Model, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((35)),
            BasicConv(in_channels=384, out_channels=96, kernel_size=1)
        )
        self.branch_2 = nn.Sequential(
            BasicConv(in_channels=384, out_channels=96, kernel_size=1)
        )
        self.branch_3 = nn.Sequential(
            BasicConv(in_channels=384, out_channels=64, kernel_size=1),
            BasicConv(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        )
        self.branch_4 = nn.Sequential(
            BasicConv(in_channels=384, out_channels=64, kernel_size=1),
            BasicConv(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            BasicConv(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        )


    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_4 = self.branch_4(x)

        # print("x_1.shape = ",x_1.shape)
        # print("x_2.shape = ",x_2.shape)
        # print("x_3.shape = ",x_3.shape)
        # print("x_4.shape = ",x_4.shape)
        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)


        return x


class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()
        # pass
        self.branch_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch_2 = BasicConv(in_channels=384, out_channels=384, kernel_size=3, stride=2)

        self.branch_3 = nn.Sequential(
            BasicConv(in_channels=384, out_channels=192,kernel_size=1),
            BasicConv(in_channels=192, out_channels=224, kernel_size=3),
            BasicConv(in_channels=224 , out_channels=256, kernel_size=3, stride=2, padding=1)
        )


    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)

        # print("x_1.shape = ",x_1.shape)
        # print("x_2.shape = ",x_2.shape)
        # print("x_3.shape = ",x_3.shape)
        x = torch.cat([x_1, x_2, x_3], dim=1)
        return x


class Inception_B(nn.Module):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((17)),
            BasicConv(in_channels=1024, out_channels=128,kernel_size=1)
        )
        self.branch_2 = BasicConv(in_channels=1024, out_channels=384, kernel_size=1)
        self.branch_3 = nn.Sequential(
            BasicConv(in_channels=1024, out_channels=192, kernel_size=1),
            BasicConv(in_channels=192, out_channels=224, kernel_size=(1,7), padding=(0,3)),
            BasicConv(in_channels=224, out_channels=256, kernel_size=(1,7), padding=(0,3))
        )
        self.branch_4 = nn.Sequential(
            BasicConv(in_channels=1024, out_channels=192, kernel_size=1),
            BasicConv(in_channels=192, out_channels=192, kernel_size=(1,7), padding=(0,3)),
            BasicConv(in_channels=192,out_channels=224, kernel_size=(7,1),padding=(3,0)),
            BasicConv(in_channels=224, out_channels=224, kernel_size=(1,7), padding=(0,3)),
            BasicConv(in_channels=224,out_channels=256, kernel_size=(7,1), padding=(3,0))
        )


    def forward(self, x ):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_4 = self.branch_4(x)
        # print("x_1.shape = ",x_1.shape)
        # print("x_2.shape = ",x_2.shape)
        # print("x_3.shape = ",x_3.shape)
        # print("x_4.shape = ",x_4.shape)

        x = torch.cat([x_1, x_2, x_3, x_4],dim=1)

        return x


class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()
        # pass
        self.branch_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch_2 = nn.Sequential(
            BasicConv(in_channels=1024, out_channels=192, kernel_size=1),
            BasicConv(in_channels=192, out_channels=192, kernel_size=3, stride=2)
        )
        self.branch_3 = nn.Sequential(
            BasicConv(in_channels=1024, out_channels=256, kernel_size=1),
            BasicConv(in_channels=256, out_channels= 256, kernel_size=(1,7), padding=(0,3)),
            BasicConv(in_channels=256, out_channels=320, kernel_size=(7,1), padding=(3,0)),
            BasicConv(in_channels=320, out_channels=320, kernel_size=3, stride=2)
        )



    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)

        # print("x_1.shape = ",x_1.shape)
        # print("x_2.shape = ",x_2.shape)
        # print("x_3.shape = ",x_3.shape)
        x = torch.cat([x_1, x_2, x_3],dim=1)
        return x

class InceptionC(nn.Module):
    def __init__(self):
        super(InceptionC, self).__init__()
        # pass
        self.branch_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8)),
            BasicConv(in_channels=1536, out_channels=256, kernel_size=1)
        )
        self.branch_2 = BasicConv(in_channels=1536, out_channels=256 ,kernel_size=1)

        self.b_3_1 = BasicConv(in_channels=1536, out_channels=384, kernel_size=1)
        self.b_3_2_1 = BasicConv(in_channels=384, out_channels=256, kernel_size=(1,3), padding=(0,1))
        self.b_3_2_2 = BasicConv(in_channels=384, out_channels=256, kernel_size=(3,1), padding=(1,0))

        self.b_4_1 = nn.Sequential(
            BasicConv(in_channels=1536, out_channels=384, kernel_size=1),
            BasicConv(in_channels=384, out_channels=448, kernel_size=(1,3), padding=(0,1)),
            BasicConv(in_channels=448, out_channels=512, kernel_size=(3,1), padding=(1,0))
        )

        self.b_4_2_1 = BasicConv(in_channels=512, out_channels=256, kernel_size=(3,1), padding=(1,0))
        self.b_4_2_2 = BasicConv(in_channels=512, out_channels=256, kernel_size=(1,3), padding=(0,1))

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)

        x_3 = self.b_3_1(x)
        x_3_1 = self.b_3_2_1(x_3)
        x_3_2 = self.b_3_2_2(x_3)

        x_4 = self.b_4_1(x)
        x_4_1 = self.b_4_2_1(x_4)
        x_4_2 = self.b_4_2_2(x_4)

        x = torch.cat([x_1, x_2, x_3_1, x_3_2, x_4_1, x_4_2], dim=1)


        return x


@MODELS.register_module()
class InceptionV4(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV4, self).__init__()
        self.stem = StemModel()

        self.ModelA_1 = Inception_A_Model()
        self.ModelA_2 = Inception_A_Model()
        self.ModelA_3 = Inception_A_Model()
        self.ModelA_4 = Inception_A_Model()

        self.reduction_a = Reduction_A()

        self.ModelB_1 = Inception_B()
        self.ModelB_2 = Inception_B()
        self.ModelB_3 = Inception_B()
        self.ModelB_4 = Inception_B()
        self.ModelB_5 = Inception_B()
        self.ModelB_6 = Inception_B()
        self.ModelB_7 = Inception_B()

        self.Reduction_B = Reduction_B()

        self.ModelC_1 = InceptionC()
        self.ModelC_2 = InceptionC()
        self.ModelC_3 = InceptionC()

        self.avg_pool = nn.AdaptiveAvgPool2d((1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1536, out_features=num_classes)


    def forward(self, x):
        x = self.stem(x)
        x = self.ModelA_1(x)
        x = self.ModelA_2(x)
        x = self.ModelA_3(x)
        x = self.ModelA_4(x)

        x = self.reduction_a(x)

        x = self.ModelB_1(x)
        x = self.ModelB_2(x)
        x = self.ModelB_3(x)
        x = self.ModelB_4(x)
        x = self.ModelB_5(x)
        x = self.ModelB_6(x)
        x = self.ModelB_7(x)

        x = self.Reduction_B(x)

        x = self.ModelC_1(x)
        x = self.ModelC_2(x)
        x = self.ModelC_3(x)

        x = self.avg_pool(x)
        x = self.flatten(x)

        x = torch.dropout(x, 0.2,train=True)
        x = self.fc(x)
        # x = torch.softmax(x, dim=1)

        return None,x

if __name__ == '__main__':
    input = torch.randn([10,3,299,299])
    model = InceptionV4(num_classes=5)
    output = model(input)
    # print(output.shape)
    # print(output)