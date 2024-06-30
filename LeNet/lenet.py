import torch.nn as nn
import torch.nn.functional as F
# import torch


class LeNet (nn.Module):
    # 初始化
    def __init__(self):
        super(LeNet,self).__init__()
        # 2d卷积 channel为3, 卷积核1为6个, 卷积核大小5*5
        # N = (W-F+2P)/S + 1    -->   28 = (32-5+2*0)/1 + 1
        # input (3,32,32)   -->    output (16,28,28)
        self.conv1 = nn.Conv2d(3,16,5 )
        # 池化层 池化盒大小2*2
        # -->output (16,14,14)
        self.pool1 = nn.MaxPool2d(2,2)
        # -->output (32,10,10)
        self.conv2 = nn.Conv2d(16,32,5)
        # -->output (32,5,5)
        self.pool2 = nn.MaxPool2d(2,2)
        # 120, 84为论文节点数, 10为‘cifar10’类别数
        self.fc1 = nn.Linear(32*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # 通过view()将特征矩阵-->一维向量
        # -1防止view()将 batch_size 也展开
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 内置 softmax 方法
        x = self.fc3(x)
        return x
    

# inputX = torch.rand([32,3,32,32])
# model = LeNet()
# print(model)
# outputX = model(inputX)
