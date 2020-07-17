#coding=utf-8
from torch import nn
from efficientnet_pytorch import EfficientNet

model1 = EfficientNet.from_pretrained('efficientnet-b4')


class Efficietnet_b4(nn.Module):
    def __init__(self,classes):
        super(Efficietnet_b4, self).__init__()
        self.basemodel1 = model1
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(1792, classes)

    def forward(self, inputs):
        bs = inputs.size(0)

        #返回basemodel最后一层卷积层   所以实际上没有用到EfficeintNet最终的全联接层。
        x = self.basemodel1.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x


