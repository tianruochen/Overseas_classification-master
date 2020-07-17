import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import torchvision

class Efficietnet(nn.Module):
    def __init__(self,classes):
        super(Efficietnet, self).__init__()
        self.basemodel1 = EfficientNet.from_pretrained('efficientnet-b4')

        #pytorch 卷积核的shape：[输出通道，输入通道，卷积核的高，卷积核的宽]
        #pytorch 卷积层的输出shape：[batch_size，输出通道，特征图高，特征图宽]
        #opencv 读取图片 BGR   BGR->rgb:  img = img[...,::-1]
        for p in self.basemodel1.named_parameters():
            print("name:"+p[0]+"\t",p[1].size())
        print('===========================================')

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(1792, classes)

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.basemodel1.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        # x = self._dropout(x)
        x = self._fc(x)
        return x


class Efficietnet_b5(nn.Module):
    def __init__(self,classes):
        super(Efficietnet_b5, self).__init__()
        self.basemodel1 = EfficientNet.from_pretrained('efficientnet-b5')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, classes)

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.basemodel1.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        # x = self._dropout(x)
        x = self._fc(x)
        return x


class Efficietnet_b5_(nn.Module):
    def __init__(self,classes):
        super(Efficietnet_b5_, self).__init__()
        self.basemodel1 = EfficientNet.from_pretrained('efficientnet-b5')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, 1000,bias = True)
        self._fc1 = nn.Linear(1000,classes, bias = True)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.basemodel1.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        x = self.relu(x)
        x = self._dropout(x)
        x = self._fc1(x)
        return x


class Efficietnet_b7(nn.Module):
    def __init__(self,classes):
        super(Efficietnet_b7, self).__init__()
        self.basemodel1 = EfficientNet.from_pretrained('efficientnet-b7')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2560, classes)

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.basemodel1.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x


class densenet201(nn.Module):
    def __init__(self,classes):
        self.classes = classes
        super(densenet201, self).__init__()
        self.basemodel1 = torchvision.models.densenet201(pretrained=True)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(1920, classes)
    def forward(self, inputs):
        features = self.basemodel1.features(inputs)
        x = F.relu(features, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(features.size(0), -1)
        # x = self._dropout(x)
        out = self._fc(x)
        return out


class densenet121(nn.Module):
    def __init__(self,classes):
        self.classes = classes
        super(densenet121, self).__init__()
        self.basemodel1 = torchvision.models.densenet121(pretrained=True)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(1024, classes)

    def forward(self, inputs):
        features = self.basemodel1.features(inputs)
        x = F.relu(features, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(features.size(0), -1)
        # x = self._dropout(x)
        out = self._fc(x)
        return out


class Resnet50(nn.Module):
    def __init__(self,classes):
        self.classes = classes
        super(Resnet50, self).__init__()
        self.basemodel1 = torchvision.models.resnet50(pretrained=True)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.basemodel1.conv1(x)
        x = self.basemodel1.bn1(x)
        x = self.basemodel1.relu(x)
        x = self.basemodel1.maxpool(x)

        x = self.basemodel1.layer1(x)
        x = self.basemodel1.layer2(x)
        x = self.basemodel1.layer3(x)
        x = self.basemodel1.layer4(x)
        x = self.basemodel1.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self._fc(x)
        return x


if __name__ == '__main__':
    tfms = transforms.Compose([transforms.Resize(800), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])#这是轮流操作的意思
    # resized_images = []
    img1 = tfms(Image.open('./simple/result3.jpg').resize((800,int(800*1.5)))).unsqueeze(0)
    l1 = []
    for i in range(8):
        l1.append(img1)
    resized_images = l1
    image_tensors = torch.cat(resized_images, 0)

    # densenet101 = densenet101(256)
    # densenet101(image_tensors)


    ## 定义DenseNet实例,加载与训练模型，并更改最后一层
    cnn = Resnet50(4)
    for param in cnn.parameters():
        param.requires_grad = False
    # num_features = cnn.classifier.in_features
    # print(num_features)
    # cnn.classifier = nn.Linear(num_features, 256)
    print(cnn(image_tensors).shape)

    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    efficient1 = Efficietnet(8)
    model = torch.nn.DataParallel(efficient1).to(device)
    model.train()
    # Focal = FocalLoss(3)
    # model = torch.nn.DataParallel(efficient1).to(device)
    tfms = transforms.Compose([transforms.Resize(448), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])#这是轮流操作的意思
    # resized_images = []
    img1 = tfms(Image.open('./simple/result3.jpg').resize((448,448))).unsqueeze(0)
    l1 = []
    for i in range(64):
        l1.append(img1)
    resized_images = l1
    image_tensors = torch.cat(resized_images, 0)
    image_tensors = image_tensors.to(device)
    # image_tensors = img1.to(device)
    print(image_tensors.shape)
    # efficient1.eval()


    # targets = torch.tensor([1,2])
    # ids = targets.view(-1, 1)
    # print(ids)
    for i in range(1000):
        # with torch.no_grad():
        outputs1 = efficient1(image_tensors)
        # print(outputs1)
        # outputs = torch.tensor([[0,1.0,0],[0,0,1.0]])
        # print(Focal(outputs,targets))
        # loss = F.cross_entropy(outputs, targets, ignore_index=-1, reduction='elementwise_mean')
        # print(loss)

    ################################
    efficient1.eval()
    with torch.no_grad():
        for i in tqdm(range(1000)):
            outputs = efficient1(image_tensors)
        # print(outputs.shape)
        # print(torch.topk(outputs, k=5))
        # prob = torch.softmax(outputs, dim=1)
        # print(prob.shape)
    ################################
    '''


