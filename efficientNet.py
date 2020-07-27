import model_config
import pickle
import torch
from keras import backend as K
from network import Efficietnet_b4
from efficientnet import EfficientNetB5, EfficientNetB4
from keras import models, layers
from keras.models import load_model
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PornModel(object):
    def __init__(self, config_name):
        self.config = model_config.ALL_CONFIG[config_name]
        self.network = self.config['network_type']
        self.model_path = self.config['model_path']
        self.suffix_model_path = self.config['suffix_model_path']
        self.classnum = self.config['class_num']
        self.normal_axis = self.config['normal_axis']
        self.low_threshold = self.config['low_threshold']
        self.high_threshold = self.config['high_threshold']
        self.max_frames = self.config['max_frames']
        self.hw_rate = self.config['hw_rate']

        with open(self.suffix_model_path, 'rb') as f:
            self.suf_model = pickle.load(f)

        if self.network == "B5":
            dropout_rate = 0.0
            self.input_shape = (456, 456)
            self.network_input = (456, 456, 3)
            second_stage_open_layer = 'add_10'
            K.reset_uids()
            #conv_base = EfficientNetB5(weights='imagenet', include_top=False, input_shape=self.network_input)
            conv_base = EfficientNetB5(weights=None, include_top=False, input_shape=self.network_input)
            model = models.Sequential()
            model.add(conv_base)
            model.add(layers.GlobalAveragePooling2D(name="gap"))
            model.add(layers.Dropout(dropout_rate, name="dropout_out"))
            model.add(layers.Dense(1024, activation='relu', name="fc1"))
            model.add(layers.Dropout(dropout_rate, name="dropout_out_2"))
            model.add(layers.Dense(self.classnum, activation='softmax', name="fc_out"))

            conv_base.trainable = True
            set_trainable = False
            for layer in conv_base.layers:
                if layer.name == second_stage_open_layer:
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False

            model.load_weights(self.model_path)
            self.model = model

        if self.network == "B4":
            dropout_rate = 0.0
            self.input_shape = (380, 380)
            self.network_input = (380, 380, 3)
            second_stage_open_layer = 'add_2'
            K.reset_uids()
            #conv_base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=self.network_input)
            conv_base = EfficientNetB5(weights=None, include_top=False, input_shape=self.network_input)
            model = models.Sequential()
            model.add(conv_base)
            model.add(layers.GlobalAveragePooling2D(name="gap"))
            model.add(layers.Dropout(dropout_rate, name="dropout_out"))
            model.add(layers.Dense(1024, activation='relu', name="fc1"))
            model.add(layers.Dropout(dropout_rate, name="dropout_out_2"))
            model.add(layers.Dense(self.classnum, activation='softmax', name="fc_out"))

            conv_base.trainable = True
            set_trainable = False
            for layer in conv_base.layers:
                if layer.name == second_stage_open_layer:
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False

            model.load_weights(self.model_path)
            self.model = model

        else:
            self.model = load_model(self.model_path)
            self.input_shape = (299, 299)

    def preprocess(self, image):
        image = image.resize(self.input_shape)
        image_array = np.asarray(image).astype(np.float32)
        image_array *= 1.0 / 255.0
        return image_array

    def pil2bgr(self, im):
        im.thumbnail((512, 512))
        rgb_img = np.array(im)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return bgr_img

    def predict_img_api(self, im):
        try:
            img_w, img_h = im.size
            img_w = float(img_w)
            img_h = float(img_h)
            h_w = img_h / img_w
        except:
            return -1, 0

        img_list = []
        if h_w > 2.0:
            split_len = int(img_w * self.hw_rate)
            h_div_w = img_h / split_len
            split_num = int(min(self.max_frames, np.ceil(h_div_w)))

            split_stride = int((img_h - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((0, split_stride * i, img_w, split_stride * i + split_len))
                img_list.append(self.preprocess(t_img))

        elif h_w < 0.5:
            split_len = int(img_h * self.hw_rate)
            h_div_w = img_w / split_len
            split_num = int(min(self.max_frames, np.ceil(h_div_w)))

            split_stride = int((img_w - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((split_stride * i, 0, split_stride * i + split_len, img_h))
                img_list.append(self.preprocess(t_img))
        else:
            img_list.append(self.preprocess(im))

        pred = self.model.predict(np.array(img_list))

        suf_score = np.max(self.suf_model.predict_proba(pred)[:, 1])
        # print(suf_score)
        suf_ret = int(suf_score > self.low_threshold) + int(suf_score > self.high_threshold)
        return suf_score,pred.tolist()
        # return suf_ret, pred.tolist()

    def predict_webp_api(self, im_list):
        num_im = len(im_list)
        img_array = []
        if num_im <= self.max_frames:
            for i in range(num_im):
                img_array.append(self.preprocess(im_list[i]))
        else:
            gif_stride = (num_im - 1) / (self.max_frames - 1)
            for i in range(self.max_frames):
                img_array.append(self.preprocess(im_list[int(np.round(i * gif_stride))]))
        img_array = np.array(img_array)
        pred = self.model.predict(img_array)

        suf_score = np.max(self.suf_model.predict_proba(pred)[:, 1])
        suf_ret = int(suf_score > self.low_threshold) + int(suf_score > self.high_threshold)
        return suf_ret, pred.tolist()

    def predict_gif_api(self, frames):
        num_frame = len(frames)
        img_array = []
        if num_frame <= self.max_frames:
            for i in range(num_frame):
                img_array.append(self.preprocess(Image.fromarray(frames[i])))
        else:
            gif_stride = (num_frame - 1) / (self.max_frames - 1)
            for i in range(self.max_frames):
                img_array.append(self.preprocess(Image.fromarray(frames[int(np.round(i * gif_stride))])))
        img_array = np.array(img_array)
        pred = self.model.predict(img_array)

        suf_score = np.max(self.suf_model.predict_proba(pred)[:, 1])
        suf_ret = int(suf_score > self.low_threshold) + int(suf_score > self.high_threshold)
        return suf_ret, pred.tolist()

    def predict_im_list_api(self, im_list):
        img_list = []
        for im in im_list:
            uni_img = self.preprocess(im)
            img_list.append(uni_img)
        pred = self.model.predict(np.array(img_list))

        suf_score = np.max(self.suf_model.predict_proba(pred)[:, 1])
        print(suf_score)
        suf_ret = int(suf_score > self.low_threshold) + int(suf_score > self.high_threshold)
        return suf_ret, pred.tolist()



class UnPorn_Model(object):
    def __init__(self, config_name):
        self.config = model_config.ALL_CONFIG[config_name]
        self.network = self.config['network_type']
        self.model_path = self.config['model_path']
        self.classnum = self.config['class_num']
        self.normal_axis = self.config['normal_axis']
        self.low_threshold = self.config['low_threshold']
        self.high_threshold = self.config['high_threshold']
        self.hw_rate = self.config['hw_rate']
        self.max_frames = self.config['max_frames']
        self.input_shape = self.config['input_shape']
        self.tfms = transforms.Compose([transforms.Resize(self.input_shape[0]), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])  # 这是轮流操作的意思


        if self.network == "B4":
            dropout_rate = 0.0
            self.input_shape = self.input_shape
            self.network_input = (self.input_shape[0],self.input_shape[1],3)
            model = Efficietnet_b4(self.classnum)
            self.model = torch.nn.DataParallel(model).cuda()
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        else:
            self.model = load_model(self.model_path)
            self.input_shape = (380, 380)

    def preprocess(self, image):
        image_array = self.tfms(image.resize((380, 380))).unsqueeze(0)
        return image_array

    def pil2bgr(self, im):
        im.thumbnail((512, 512))
        rgb_img = np.array(im)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return bgr_img

    def predict_img_api(self, im):
        try:
            img_w, img_h = im.size
            img_w = float(img_w)
            img_h = float(img_h)
            h_w = img_h / img_w
        except:
            return -1, 0

        img_list = []
        if h_w > 2.0:
            split_len = int(img_w * self.hw_rate)
            h_div_w = img_h / split_len
            split_num = int(min(self.max_frames, np.ceil(h_div_w)))
            split_stride = int((img_h - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((0, split_stride * i, img_w, split_stride * i + split_len))
                img_list.append(self.preprocess(t_img))

        elif h_w < 0.5:
            split_len = int(img_h * self.hw_rate)
            h_div_w = img_w / split_len
            split_num = int(min(self.max_frames, np.ceil(h_div_w)))
            split_stride = int((img_w - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((split_stride * i, 0, split_stride * i + split_len, img_h))
                img_list.append(self.preprocess(t_img))
        else:
            img_list.append(self.preprocess(im))

        with torch.no_grad():
            pred = self.model(torch.cat(img_list, 0))
            print(type(pred))
            print(pred.shape)
        risk_rate = 1 - np.min(np.sum(pred[:, self.normal_axis].cpu().numpy(), axis=1))
        # risk_rate = 1 - np.min(np.sum(pred[:, self.normal_axis]))
        return int(risk_rate > self.low_threshold) + int(risk_rate > self.high_threshold), pred.tolist()

    def predict_webp_api(self, im_list):
        num_im = len(im_list)
        img_array = []
        if num_im <= self.max_frames:
            for i in range(num_im):
                img_array.append(self.preprocess(im_list[i]))
        else:
            gif_stride = (num_im - 1) / (self.max_frames - 1)
            for i in range(self.max_frames):
                img_array.append(self.preprocess(im_list[int(np.round(i * gif_stride))]))
        with torch.no_grad():
            pred = self.model(torch.cat(img_array, 0))

        risk_rate = 1 - np.min(np.sum(pred[:, self.normal_axis], axis=1))
        return int(risk_rate > self.low_threshold) + int(risk_rate > self.high_threshold), pred.tolist()

    def predict_gif_api(self, frames):
        num_frame = len(frames)
        img_array = []
        if num_frame <= self.max_frames:
            for i in range(num_frame):
                img_array.append(self.preprocess(Image.fromarray(frames[i])))
        else:
            gif_stride = (num_frame - 1) / (self.max_frames - 1)
            for i in range(self.max_frames):
                img_array.append(self.preprocess(Image.fromarray(frames[int(np.round(i * gif_stride))])))
        with torch.no_grad():
            pred = self.model(torch.cat(img_array, 0))

        risk_rate = 1 - np.min(np.sum(pred[:, self.normal_axis], axis=1))
        return int(risk_rate > self.low_threshold) + int(risk_rate > self.high_threshold), pred.tolist()

    def predict_im_list_api(self, im_list):
        img_list = []
        for im in im_list:
            uni_img = self.preprocess(im)
            img_list.append(uni_img)
        with torch.no_grad():
            pred = self.model(torch.cat(img_list, 0))
        risk_rate = 1 - np.min(np.sum(pred[:, self.normal_axis], axis=1))
        return int(risk_rate > self.low_threshold) + int(risk_rate > self.high_threshold), pred.tolist()