#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):  # 初始化
        self.reset()

    def add(self, s,c):  # 因为过好久才查看一下loss函数，所以先积累起来
        self.sum += s
        self.n_count += c

    def reset(self):  # 查看完成后，就置0
        self.n_count = 0
        self.sum = 0

    def val(self):  # 查看的时候就执行这句话
        res = 0
        if self.n_count != 0:
            res = float(self.sum) / float(self.n_count)
        return res


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Labelsmoothing(nn.Module):
    def __init__(self, size, smoothing=0.1):
        super(Labelsmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing#if i=y的公式
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        x = torch.nn.functional.log_softmax(x, dim=-1)  # 先给弄成softmax
        true_dist = x.data.clone()#先深复制过来
        #print true_dist
        true_dist.fill_(self.smoothing / (self.size - 1))#otherwise的公式
        #print true_dist
        #变成one-hot编码，1表示按列填充，
        #target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))/2


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class Equation_loss(nn.Module):
    def __init__(self,class_num):
        super(Equation_loss, self).__init__()
        self.class_num = class_num
    def forward(self,x, target,eql_w):
        eql_w = eql_w.cuda()
        pred_exp = torch.exp(x)
        pred_exp = pred_exp.mul(eql_w)
        sum = torch.sum(pred_exp, 1)
        pred_softmax = pred_exp / sum.view(sum.size(0), -1)

        ##############
        ones = torch.sparse.torch.eye(self.class_num)
        one_hot =  ones.index_select(0, target)
        pred_softmax = -torch.log(pred_softmax)
        one_hot = one_hot.cuda()
        pred_softmax_log = pred_softmax.mul(one_hot)
        loss_sum = torch.sum(pred_softmax_log, 1)
        ##############

        # nll_loss = pred_softmax.gather(dim=-1, index=target.unsqueeze(1))#搜集指定位置target的数值
        # nll_loss = nll_loss.squeeze(1)#化成一纬的
        # loss_sum = -torch.log(nll_loss)
        return loss_sum.mean()



