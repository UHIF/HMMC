from __future__ import print_function
import math
import torch
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import pytorch_lsm

t.manual_seed(9)
np.random.seed(9)
t.cuda.manual_seed(9)
t.cuda.manual_seed_all(9)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



trainset=tv.datasets.CIFAR100(
    root='C:/home/cy/data',
    train=True,
    download=True,
    transform=transform_train
)
trainloader=DataLoader(
    trainset,
    batch_size=100,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)
testset=tv.datasets.CIFAR100(
    'C:/home/cy/data',
    train=False,
    download=True,
    transform=transform_test
)
testloader=DataLoader(
    testset,
    batch_size=100,
    shuffle=False,
    num_workers=0
)

class Net2(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,15,kernel_size=(3,3),stride=(1,1))
        self.conv2=nn.Conv2d(15,75,kernel_size=(4,4),stride=(1,1))
        self.conv3=nn.Conv2d(75,175,kernel_size=(3,3),stride=(1,1))
        self.fc1=nn.Linear(700,200)
        self.fc2=nn.Linear(200,120)
        self.fc3=nn.Linear(120,84)
        self.fc4=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=F.max_pool2d(F.relu(self.conv3(x)),2)

        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,200)
        self.fc2=nn.Linear(200,100)
        self.fc3=nn.Linear(100,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        #        print(out.shape)
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.classifier(out)
        #        print(out.shape)
        return out
def conv_3x3_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True),
    )

def dws_conv_3x3_bn(in_channels, out_channels, dw_stride):
    """
    Depthwise Separable Convolution
    :param in_channels: depthwise conv input channels
    :param out_channels: Separable conv output channels
    :param dw_stride: depthwise conv stride
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=in_channels,
                  kernel_size=3,
                  stride=dw_stride,
                  padding=1,
                  groups=in_channels,
                  bias=False,
                  ),
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU6(inplace=True),

        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=1,
                  stride=1,
                  bias=False,
                  ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True),
    )

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNetV1, self).__init__()
        layers = [conv_3x3_bn(in_channels=3, out_channels=32, stride=1)]  # change stride 2->1 for cifar10
        dws_conv_config = [
            # num, in_channels, out_channels, stride
            [1, 32, 64, 1],
            [1, 64, 128, 1],  # change stride 2->1 for cifar10
            [1, 128, 128, 1],
            [1, 128, 256, 2],
            [1, 256, 256, 1],
            [1, 256, 512, 2],
            [5, 512, 512, 1],
            [1, 512, 1024, 2],
            [1, 1024, 1024, 1]
        ]
        for num, in_channels, out_channels, dw_stride in dws_conv_config:
            for i in range(num):
                layers.append(dws_conv_3x3_bn(in_channels, out_channels, dw_stride))
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        y = self.layers(x)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidualBlock, self).__init__()
        assert stride == 1 or stride == 2
        self.stride = stride
        self.residual = self.stride == 1 and (in_channels == out_channels)
        expansion_channels = in_channels * expansion_factor

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=expansion_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=expansion_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=expansion_channels,
                      out_channels=expansion_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=expansion_channels,
                      bias=False),
            nn.BatchNorm2d(num_features=expansion_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=expansion_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        y = self.block(x)
        if self.residual:
            return y + x
        else:
            return y

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNetV2, self).__init__()
        layers = [conv_3x3_bn(in_channels=3, out_channels=32, stride=1)] # change stride 2->1 for cifar10
        in_channels = 32
        inverted_residual_block_config = [
            # expansion factor, out_channels, stride
            [1, 16, 1],

            [6, 24, 1],  # change stride 2->1 for cifar10
            [6, 24, 1],

            [6, 32, 2],
            [6, 32, 1],
            [6, 32, 1],

            [6, 64, 2],
            [6, 64, 1],
            [6, 64, 1],
            [6, 64, 1],

            [6, 96, 1],
            [6, 96, 1],
            [6, 96, 1],

            [6, 160, 2],
            [6, 160, 1],
            [6, 160, 1],

            [6, 320, 1],
        ]
        for expansion_factor, out_channels, stride in inverted_residual_block_config:
            layers.append(InvertedResidualBlock(in_channels, out_channels, stride, expansion_factor))
            in_channels = out_channels
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=320,
                      out_channels=1280,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=1280),
        ))
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1280, out_features=num_classes)
        )

    def forward(self, x):
        y = self.layers(x)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y

def evaluteTop5(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x=x.cuda()
        y=y.cuda()
        with t.no_grad():
            logits = model(x)
            maxk = max((1,5))
            y_resize = y.view(-1,1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += t.eq(pred, y_resize).sum().float().item()
    return correct / total
top5_test=[]
rate_train=[]
rate_test=[]
net=ResNet18()
net=net.cuda()
R_W=t.ones(50000)
Z=t.zeros(50000)
VW=t.zeros(50000)
w_momentum=0
batch_size = 100
r=0.0001
pp=0
p=0.5
p_L=0
criterion=pytorch_lsm.lsmce(size_average = True)
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)
scheduler=t.optim.lr_scheduler.MultiStepLR(optimizer, [100,150], gamma=0.1)
total_loss_last = 0.0

for epoch in range(200):
    running_loss=0.0
    total_loss=0.0
    for i,data in enumerate(trainloader,0):
        W = R_W[i * batch_size:(i + 1) * batch_size]
        inputs,labels=data
        inputs=inputs.cuda()
        labels=labels.cuda()
        W=W.cuda()
        inputs,labels=Variable(inputs),Variable(labels)
        optimizer.zero_grad()
        outputs=net(inputs)
        outputs=outputs.cuda()
        loss = criterion(outputs, labels, W)
        loss.backward()
        t.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
        running_loss+=loss.data
        if i%100==99:
            print('[%d,%5d]  loss:%.3f' \
                  % ( epoch+1,i+1,running_loss ) )
            total_loss+=running_loss
            running_loss=0.0

    scheduler.step()
    train_correct = 0
    train_correct_p=0
    train_total = 0
    for i,data in enumerate(trainloader,0):
        images, labels = data
        images=images.cuda()
        labels=labels.cuda()
        outputs = net(Variable(images))
        _, predicted = t.max(outputs.data, 1)
        labeloutput=t.softmax(outputs.data,1)
        for p_i in range(labels.shape[0]):
            labeloutput[p_i,labels[p_i]] = labeloutput[p_i,labels[p_i]]-pp
        _, predicted_p = t.max(labeloutput, 1)
        train_total += labels.size(0)
        A = t.nonzero(predicted_p != labels)
        m = A.shape

        for m_i in range(m[0]):
            VW [i * batch_size + A[m_i]] = VW[i * batch_size + A[m_i]] +r
        train_correct += (predicted == labels).sum()
        train_correct_p += (predicted_p == labels).sum()
    if total_loss>total_loss_last:
        R_W = R_W +VW
        VW=VW*w_momentum
    else:
        VW=t.zeros(50000)
    total_loss_last=total_loss
    R_W=R_W/t.mean(R_W)
    Rate_p = 100 * float(train_correct_p) / float(train_total)
    if Rate_p>99.8:
        p_L=p_L+p
        pp = 1- math.exp(-0.5*p_L)
    Rate = 100 * float(train_correct) / float(train_total)
    rate_train.append(Rate)
    print('%.2f %%' % Rate)
    print('%.2f %%' % Rate_p)
    print(pp)

    test_correct = 0
    test_total = 0
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = t.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum()
    Rate = 100 * float(test_correct) / float(test_total)
    rate_test.append(Rate)
    print('%.2f %%' % Rate)

    with open('CIFAR100_Resnet18.txt', 'w') as f:
        f.write(str(rate_train)+'\n')
        f.write(str(rate_test)+'\n')
Model_Resnet_18_cifar100 = net
torch.save(Model_Resnet_18_cifar100, 'Model_Resnet_18_cifar100.pkl')


