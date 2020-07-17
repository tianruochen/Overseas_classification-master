import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


class Data_form_list(Dataset):
    def __init__(self,l):
        images = []
        for l_append in l:
            l_append.append(0.0)   #??
            images.append(l_append)
        #<class 'list'>: [['/data1/zhaoshiyu/porn_train_data_1018/train/sex_toy/00177701.png', '0', 0.0], ['/data1/zhaoshiyu/porn_train_data_1018/train/sex_toy/sex_toy_add_000786.png', '0', 0.0]]
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


class AlignCollate(object):

    def __init__(self, imgSize=224):
        self.imgSize = imgSize

        #transforms.ToTensor()会把图片通道换到最前面。（H，W，C）-》（C，H，W）
        self.tfms = transforms.Compose([transforms.Resize(int(self.imgSize*1.06)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomCrop(self.imgSize),
                                        transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])  # 这是轮流操作的意思


    def __call__(self, batch):
        #输入是batch的list，输出是batch
        images = []#batch是个list
        labels = []
        defficty = []
        images_dictorys = []
        for i in batch:
            [image,label_,deffict_degree] = i
            try:
                #PIL获得的图像是RGB格式的   通过img.size属性获得图片的（宽，高）
                #cv2获得的图像是BGR格式的   通过img.shpae属性获得图片的（高，宽）
                img = self.tfms(Image.open(image).convert("RGB").resize((self.imgSize,self.imgSize))).unsqueeze(0)
                images.append(img)
                labels.append(torch.tensor([int(label_)]))
                images_dictorys.append(image)
                defficty.append(torch.tensor([deffict_degree]))
            except Exception as ex:
                print(ex)
                print(image)
                continue
        defficty_tensors = torch.cat(defficty, 0)
        image_tensors = torch.cat(images, 0)
        label_tensors = torch.cat(labels,0)
        return image_tensors,label_tensors,images_dictorys,defficty_tensors


class AlignCollate_val(object):
    def __init__(self, imgSize=456):
        self.imgSize = imgSize
        self.tfms = transforms.Compose([transforms.Resize(int(self.imgSize)),
                                        # transforms.Pad((0, int(self.imgSize * 0.25))),
                                    transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])  # 这是轮流操作的意思
    def __call__(self, batch):
        #输入是batch的list，输出是batch
        images = []#batch是个list
        labels = []
        defficty = []
        images_dictorys = []
        for i in batch:
            [image,label_,deffict_degree] = i
            try:
                img = self.tfms(Image.open(image).convert("RGB").resize((self.imgSize,self.imgSize))).unsqueeze(0)
                # img =  torch.tensor(self.input_x(image))
                images.append(img)
                labels.append(torch.tensor([int(label_)]))
                images_dictorys.append(image)
                defficty.append(torch.tensor([deffict_degree]))
            except Exception as ex:
                print(ex)
                print(image)

                continue
        defficty_tensors = torch.cat(defficty, 0)
        image_tensors = torch.cat(images, 0)
        label_tensors = torch.cat(labels,0)
        return image_tensors,label_tensors,images_dictorys,defficty_tensors








