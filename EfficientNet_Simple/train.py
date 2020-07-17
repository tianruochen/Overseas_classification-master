import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from EfficientNet_Simple.network import Efficietnet,densenet121,Efficietnet_b5,Efficietnet_b7,Resnet50
from EfficientNet_Simple.dataset import Data_form_list,AlignCollate,AlignCollate_val
from EfficientNet_Simple.utils import Averager,FocalLoss
import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch import nn
import collections
import sys
import numpy
from sklearn.metrics import roc_auc_score
# from transformers import get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 16
image_size = 380
numer_classes = 11

#训练、测试数据的存储路径
data_path_train = '/data1/wangruihao/cocofun/porn/new_train/train.txt'
data_path_val = '/data1/wangruihao/cocofun/porn/new_train/test.txt'
#最佳模型保存路径
save_best_auc_path = '/data/wangruihao/centernet/Centernet_2/EfficientNet/models/models_porn/best_accuracy_11_class_b4_auc_adl_380.pth'
save_best_accuracy_path = '/data/wangruihao/centernet/Centernet_2/EfficientNet/models/models_porn/best_accuracy_11_class_b4_accuracy_adl_380.pth'
load_path = '/data/wangruihao/centernet/Centernet_2/EfficientNet/models/models_porn/best_accuracy_11_class_b4_accuracy_adl_380_0.901.pth'
optim_type = 'Adadelta'


class train_data(object):
    def __init__(self,l_data):
        #self.AlignCollate 初始化了两个变量  一个是imgSize，另一个是transform对象
        self.AlignCollate_ = AlignCollate(imgSize=image_size)
        self.l_data = l_data


    def get_circle_data(self):
        data_ = Data_form_list(self.l_data)
        self.train_dataset1 = torch.utils.data.DataLoader(
            data_, batch_size=batch_size,
            shuffle=True,
            num_workers=max(int(batch_size/4),2),
            collate_fn=self.AlignCollate_, pin_memory=False)
        return self.train_dataset1


def validation(model,val_dataset):
    count = 0
    eq_count = 0
    l = []
    l_preds = []
    l_labels = []
    for image_tensors, labels,_,_ in val_dataset:
        #将标签tensor转换为tensor
        labels_list = labels.tolist()
        images = image_tensors.to(device)  # 数据转化为batch
        labels = labels.to(device)
        model_results = model(images)
        #获取模型输出类别
        outputs = torch.softmax(model_results, dim=1)
        outputs_label = torch.argmax(outputs, dim=1)

        #count来统计验证图片数量
        count += outputs_label.data.numel()
        #eq_count统计所有匹配正确的图片数量
        eq = torch.eq(labels, outputs_label)
        eq = eq.cpu().tolist()
        eq_count += sum(eq)

        outputs_list = outputs.cpu().tolist()
        for j in range(len(eq)):
            l_preds.append(outputs_list[j])
            l_label = [0]*len(outputs_list[j])
            l_label[labels_list[j]] = 1
            l_labels.append(l_label)
            if eq[j] == 0:
                l.append(labels_list[j])
    print(collections.Counter(l))
    np_pred = numpy.array(l_preds)
    np_label = numpy.array(l_labels)
    roc_auc_score_ = roc_auc_score(np_label,np_pred)
    return float(eq_count)/float(count),l,roc_auc_score_


def train_val_data_from_dictory(train_dictory,val_dictory):
    l_data_train = []
    with open(train_dictory,'r') as f:
        for i in f.readlines():
            l_data_train.append(i.strip().split('\t'))

    l_data_val = []
    with open(val_dictory,'r') as f:
        for i in f.readlines():
            l_data_val.append(i.strip().split('\t'))

    return l_data_train,l_data_val


def train():
    ############################
    #train_list,val_list
    #l_data_train : 列表的列表  每一个内部列表中保留两个元素：[[图片路径，类别标签]...]
    l_data_train,l_data_val = train_val_data_from_dictory(data_path_train,data_path_val)
    ############################


    ############################
    #dataset
    train_dataset1 = train_data(l_data_train)
    ############################

    ############################
    #loadnet
    net = Efficietnet(numer_classes)

    #打印网络结构
    for p in net.named_parameters():
        print("name:" + p[0] + "\t", p[1].size())
    model = torch.nn.DataParallel(net).to(device)
    #已经训练好的最佳的模型
    if load_path != '':
        model.load_state_dict(torch.load(load_path))
    model.train()
    ############################

    ############################
    #创建优化器
    filtered_parameters = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):#对需要back的添加一下
        filtered_parameters.append(p)
    if optim_type == 'Adadelta':
        optimizer = optim.Adadelta(filtered_parameters, lr=0.1, rho=0.9, eps=1e-6)
    elif optim_type == 'SGD':
        optimizer = optim.SGD(filtered_parameters, lr=0.01, momentum=0.9)
    elif optim_type == 'adam':
        optimizer = optim.Adam(filtered_parameters, lr=0.001)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(l_data_train) / BATCH_SIZE * 5,
    #                                             num_training_steps=num_train_steps)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 50, eta_min=0.0001)
    # criterion = FocalLoss(256).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    ############################

    i = 0
    best_accuracy = 0
    best_auc = 0
    average = Averager()
    #对于训练数据，获得一个DataLoder对象
    train_images = train_dataset1.get_circle_data()

    #######################
    data_ = Data_form_list(l_data_val)
    AlignCollate_va = AlignCollate_val(imgSize=image_size)
    val_dataset = torch.utils.data.DataLoader(
        data_, batch_size=batch_size,
        shuffle=True,
        num_workers=int(batch_size/4),
        collate_fn=AlignCollate_va, pin_memory=False)  # batch = self.collate_fn([self.dataset[i] for i in indices])
    #######################


    for epoch in range(300):
        print('********************'+str(epoch)+'*******************')
        for image_tensors, labels_, image_dics,difficult_degree in train_images:
            #########################
            #计算损失
            images = image_tensors.to(device)#数据转化为batch
            labels = labels_.to(device)
            model_results = model(images)
            outputs = model_results
            outputs_label = torch.argmax(outputs, dim=1)
            equal = torch.eq(labels, outputs_label)
            average.add(equal.sum(),outputs_label.data.numel())
            cost = criterion(outputs, labels)
            #########################



            #########################
            if i % 50 == 0:
                accuracy = average.val()
                print('training epoch:'+str(i)+' accuracy:',accuracy,'best_accuracy:', best_accuracy)
                average.reset()
            #########################



            #########################
            #if i == 0:
            if i % 100 == 0 and i > 0:
                model.eval()
                with torch.no_grad():
                    current_accuracy,l,current_auc = validation(model,val_dataset)
                model.train()
                sys.stdout.flush()
                # for
                if current_accuracy >= best_accuracy:
                    best_accuracy = current_accuracy
                    # torch.save(model.state_dict(),save_best_accuracy_path)
                if  current_auc >= best_auc:
                    best_auc = current_auc
                    # torch.save(model.state_dict(), save_best_auc_path)
                print('current_accuracy:', current_accuracy, 'best_accuracy:', best_accuracy)
                print('current_auc:', current_auc, 'best_auc:', best_auc)
            #########################


            #########################
            #backward
            model.zero_grad()
            cost.backward()
            optimizer.step()
            i += 1


if __name__ == '__main__':
    train()
