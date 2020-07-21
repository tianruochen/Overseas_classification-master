import codecs
import os
import json
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torch.nn as nn
from keras.models import load_model, Model
import efficientnet.model as eff_model
from tqdm import tqdm
import imageio
from PIL import ImageFile, Image
import webp
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"


def trans_array(input_array):
    result = []
    for line in input_array:
        result.append(line)
    return np.array(result)


class ModelsClassify(object):
    def __init__(self, model_path, network=None):
        if network == "B5":
            self.model = load_model(model_path,
                                    custom_objects={'ConvKernalInitializer': eff_model.ConvKernalInitializer})
            self.input_shape = (456, 456)

        elif network == "B4":
            self.model = load_model(model_path,
                                    custom_objects={'ConvKernalInitializer': eff_model.ConvKernalInitializer})
            self.input_shape = (380, 380)

        elif network == 'xcep':
            self.model = load_model(model_path)
            self.input_shape = (299, 299)

    def preprocess(self, image):
        image = image.resize(self.input_shape)
        image_array = np.asarray(image).astype(np.float32)
        image_array *= 1.0 / 255.0

        return image_array

    def predict_img_with_split_distribution(self, img_path, hw_rate=1.0, max_frame=5):
        if hw_rate >= 2.0:
            raise ValueError("invalid parameter of [hw_rate], it should in [1, 2)")
        try:
            im = Image.open(img_path)
            if im.format == "GIF":
                return self.predict_gif(img_path)
            elif im.format == "WEBP":
                return self.predict_webp(img_path)
            im = im.convert("RGB")
            img_w, img_h = im.size
        except:
            return None

        h_w = img_h / img_w
        img_list = []
        if h_w >= 2.0:
            split_len = int(img_w * hw_rate)
            h_div_w = img_h / split_len
            split_num = int(min(max_frame, np.ceil(h_div_w)))

            split_stride = int((img_h - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((0, split_stride * i, img_w, split_stride * i + split_len))
                img_list.append(self.preprocess(t_img))

        elif h_w <= 0.5:
            split_len = int(img_h * hw_rate)
            h_div_w = img_w / split_len
            split_num = int(min(max_frame, np.ceil(h_div_w)))

            split_stride = int((img_w - split_len - 1) // (split_num - 1))
            for i in range(split_num):
                t_img = im.crop((split_stride * i, 0, split_stride * i + split_len, img_h))
                img_list.append(self.preprocess(t_img))
        else:
            uni_img = self.preprocess(im)
            img_list.append(uni_img)
        pred = self.model.predict(np.array(img_list))

        return pred

    def predict_webp(self, img_path, max_sample=5):
        im_list = webp.load_images(img_path, "RGB")
        num_im = len(im_list)
        img_array = []
        if num_im <= max_sample:
            for i in range(num_im):
                img_array.append(self.preprocess(im_list[i]))
        else:
            gif_stride = (num_im - 1) / (max_sample - 1)
            for i in range(max_sample):
                img_array.append(self.preprocess(im_list[int(np.round(i * gif_stride))]))
        img_array = np.array(img_array)
        pred = self.model.predict(img_array)
        return pred

    def predict_gif(self, img_path, max_sample=5):
        try:
            frames = np.array(imageio.mimread(img_path, memtest=False))
            frame_shape_axis = len(np.shape(frames))
            if frame_shape_axis < 4:
                frames = np.expand_dims(frames, axis=-1)
                frames = np.concatenate([frames, frames, frames], axis=-1)
            elif frame_shape_axis > 4:
                raise IOError("strange file {}".format(img_path))
            else:
                channel_num = np.shape(frames)[-1]
                if channel_num == 4:
                    frames = frames[:, :, :, :3]
                elif channel_num == 1:
                    frames = np.concatenate([frames, frames, frames], axis=-1)
                elif channel_num != 3:
                    raise IOError("strange file {}".format(img_path))
        except:
            return None
        num_frame = len(frames)
        img_array = []
        if num_frame <= max_sample:
            for i in range(num_frame):
                img_array.append(self.preprocess(Image.fromarray(frames[i])))

        else:
            gif_stride = (num_frame - 1) / (max_sample - 1)
            for i in range(max_sample):
                img_array.append(self.preprocess(Image.fromarray(frames[int(np.round(i * gif_stride))])))
        img_array = np.array(img_array)
        pred = self.model.predict(img_array)

        return pred


if __name__ == "__main__":
    # [running_mode]
    # [1] for test
    # [2] for train
    # [3] for experiment of eval
    # [4] prepare train data
    # [5] train in haiyong way
    #
    running_mode = 2
    if running_mode == 1:
        suffix_model_path = '/data1/zhaoshiyu/temp/porn_c11_lr_0229_py3.pickle'
        result_file = '/data1/zhaoshiyu/porn_result_temp/porn0229_pornhub_result.txt'
        raw_threshold = 0.4681

        f = open(suffix_model_path, 'rb')
        clf = pickle.load(f)

        fres = codecs.open(result_file, 'r', 'utf-8').read().split('\n')
        score_list = []

        look_file = '/data1/zhaoshiyu/temp/pornhub_look_0512.txt'
        flook = codecs.open(look_file, 'w', 'utf-8')
        for line in tqdm(fres):
            tline = line.strip()
            if tline == "":
                continue
            img_name, pred_str = tline.split('\t')
            temp_x = np.array(json.loads(pred_str))
            temp_proba = clf.predict_proba(temp_x)[:, 1]
            temp_score = np.max(temp_proba)
            if temp_score <= raw_threshold:
                flook.write("{}\n".format(img_name))
            score_list.append(temp_score)
        score_list = np.array(score_list)
        recall_count = np.sum(score_list > raw_threshold)
        print("recall_count = {}".format(recall_count))

    elif running_mode == 4:
        model_path_list = [
            "/data/zhaoshiyu/efficientNet_data_5/models/06-23_B4_4/2stage_ep215-loss0.207-val_acc0.911.h5",
            "/data/zhaoshiyu/efficientNet_data_5/models/06-23_B4_5/2stage_ep164-loss0.236-val_acc0.913.h5"
        ]

        backbone = 'B4'
        if_pol = False

        for model_path in model_path_list:
            date = model_path.split('/')[5].split('_')[0].replace('-', '')
            epoch = model_path.split('/')[6].split('_ep')[1].split('-')[0]
            model_name = "porn{}_ep{}".format(date, epoch)

            # ======= initialization ========
            temp_model = ModelsClassify(model_path, network=backbone)

            # ===================================
            # eval suffix dataset
            # ===================================
            eval_dir = "/data1/zhaoshiyu/suffix_train_data"
            eval_res = "/data1/zhaoshiyu/porn_result_temp/{}_suffix_result.txt".format(model_name)
            fw_res = codecs.open(eval_res, 'w', 'utf-8')

            cls_list = os.listdir(eval_dir)
            for cls in cls_list:
                cls_dir = os.path.join(eval_dir, cls)
                img_list = os.listdir(cls_dir)
                for img_name in tqdm(img_list):
                    img_path = os.path.join(cls_dir, img_name)
                    pred_prob = temp_model.predict_img_with_split_distribution(img_path)
                    if pred_prob is None:
                        print("sth wrong with {}".format(img_name))
                        continue

                    prob_str = json.dumps(pred_prob.tolist())

                    new_line = "{}\t{}\t{}\n".format(cls, img_name, prob_str)
                    fw_res.write(new_line)

            fw_res.close()

    elif running_mode == 2:
        # file_path = "/data1/zhaoshiyu/porn_result_temp/porn0229_suffix_result.txt"
        # 后接分类器的训练数据路径  （EfficientNet的输出）
        suffix_file_path = "/data1/zhaoshiyu/porn_result_temp/porn_0620/softmax/porn0620_softmax_suffix_train_data_result.txt"
        root_dir = os.path.dirname(suffix_file_path)
        model_name = suffix_file_path.split('/')[-1].split('_suffix')[0]     # porn0620
        pickle_path = '/home/changqing/workspaces/Overseas_classification-master/EfficientNet_Simple/{}_lr.pickle'.format(model_name)

        porn_labels = ['common_porn', 'cartoon_porn']
        sexy_labels = ['cartoon_sexy', 'female_sexy']

        false_recall_rate = 0.018      #误召率

        fr = codecs.open(suffix_file_path, 'r', 'utf-8').read().split('\n')
        print(fr[:5])
        # print(json.loads(fr[22393].split("\t")[2]))
        print("Trianing data : ",len(fr))        #44720

        train_rate = 0.7
        porn_weight = 0.5
        sexy_weight = 0.2
        normal_weight = 0.5
        rand_seed = 11
        # raw_threshold = 0.339


        x_array = []
        y_array = []
        # i = 0
        sample_weight = []
        for line in fr:
            # i = i+1
            # print(i)       #22394
            tline = line.strip()
            if tline == "":
                continue
            #类型，图片名称，训练数据[[11维的向量]..]
            items = tline.split('\t')
            temp_x = np.max(json.loads(items[2]), axis=0).tolist()
            label = items[0]
            x_array.append(temp_x)
            if label != 'normal':
                if label in porn_labels:
                    sample_weight.append(porn_weight)
                elif label in sexy_labels:
                    sample_weight.append(sexy_weight)
                else:
                    continue
                y_array.append(1)
            else:
                y_array.append(0)
                sample_weight.append(normal_weight)

        total_array = np.array(list(zip(x_array, y_array, sample_weight)))
        print(total_array.size)
        np.random.seed(rand_seed)
        np.random.shuffle(total_array)

        split_point = int(len(total_array) * train_rate)
        train_array = total_array[: split_point]       #30603
        test_array = total_array[split_point:]
        train_x = train_array[:, 0]
        train_x = trans_array(train_x)
        train_y = train_array[:, 1].astype(int)
        train_sample = train_array[:, 2]
        test_x = test_array[:, 0]
        test_x = trans_array(test_x)
        test_y = test_array[:, 1].astype(int)
        test_sample = test_array[:, 2]

        LR = LogisticRegression()
        LR.fit(train_x, train_y, sample_weight=train_sample)
        result = LR.score(test_x, test_y, sample_weight=test_sample)
        print(result)
        train_res = LR.score(train_x, train_y, sample_weight=train_sample)
        print(train_res)

        y_pred = LR.predict(test_x)
        y_proba = LR.predict_proba(test_x)[:, 1]

        test_len = len(y_pred)       # 13116

        raw_stat = [0, 0, 0]   # 预测的违规图片中porn的数量，预测的违规图片中sexy的数量,预测的违规图片数，
        new_stat = [0, 0, 0]   # 测试集中porn的数量，测试集中sexy的数量,测试集总量
        for i in range(test_len):
            temp_sample = test_sample[i]
            new_pred = y_pred[i]
            if new_pred > 0:
                new_stat[2] += 1        #预测的违规数量
                if temp_sample == porn_weight:
                    new_stat[0] += 1        #预测的违规图片中porn的数量
                elif temp_sample == sexy_weight:
                    new_stat[1] += 1        #预测的违规图片中sexy的数量

            raw_stat[2] += 1
            if temp_sample == porn_weight:
                raw_stat[0] += 1
            elif temp_sample == sexy_weight:
                raw_stat[1] += 1

        print(raw_stat)
        print(new_stat)

        with open(pickle_path, 'wb') as f:
            # pickle.dump(LR, f, protocol=2)
            pickle.dump(LR, f)

        with open(pickle_path, 'rb') as f:
            clf2 = pickle.load(f)
            print(clf2.coef_)

        # /data1/zhaoshiyu/porn_result_temp/porn_0620/softmax/porn0620_softmax_im_eval_dataset_result.txt
        # model_name = 'porn0620'
        # root_dir = '/data1/zhaoshiyu/porn_result_temp/porn_0620/raw/'
        im_normal_file = "{}/{}_im_normal_dataset_result.txt".format(root_dir, model_name)
        im_sense_file = "{}/{}_im_eval_dataset_result.txt".format(root_dir, model_name)
        common_porn_file = "{}/{}_common_porn_0512_result.txt".format(root_dir, model_name)
        cartoon_porn_file = "{}/{}_cartoon_porn_0512_result.txt".format(root_dir, model_name)
        porn_hub_file = "{}/{}_porn_hub_dataset_result.txt".format(root_dir, model_name)


        normal_pred = []
        fnorm = codecs.open(im_normal_file, 'r', 'utf-8').read().split('\n')
        for line in fnorm:
            tline = line.strip()
            if tline == "":
                continue
            img_name, pred_str = tline.split('\t')
            pred = np.max(json.loads(pred_str), axis=0)
            normal_pred.append(pred)
        print('='*60)
        print("normal pred example:")
        print(normal_pred[:2])
        normal_pred = np.array(normal_pred)
        normal_scores = clf2.predict_proba(normal_pred)[:, 1].tolist()
        normal_scores.sort(reverse=True)
        print("sorted normal scores:")
        print(normal_scores[:50])
        threshold = normal_scores[int(np.round(len(normal_scores) * false_recall_rate))]
        print("threshold = {}".format(threshold))

        # im dataset
        sense_pred = []
        fsense = codecs.open(im_sense_file, 'r', 'utf-8').read().split('\n')
        for line in fsense:
            tline = line.strip()
            if tline == "":
                continue
            img_name, pred_str = tline.split('\t')
            pred = np.max(json.loads(pred_str), axis=0)
            sense_pred.append(pred)
        sense_pred = np.array(sense_pred)
        sense_scores = clf2.predict_proba(sense_pred)[:, 1]

        recall_num = np.sum(sense_scores > threshold)
        recall_rate = recall_num / len(sense_scores)

        print("im recall_num = {}".format(recall_num))
        print("im recall_rate = {:.2f} %".format(recall_rate * 100))

        # common_porn dataset
        sense_pred = []
        fsense = codecs.open(common_porn_file, 'r', 'utf-8').read().split('\n')
        for line in fsense:
            tline = line.strip()
            if tline == "":
                continue
            img_name, pred_str = tline.split('\t')
            pred = np.max(json.loads(pred_str), axis=0)
            sense_pred.append(pred)
        sense_pred = np.array(sense_pred)
        sense_scores = clf2.predict_proba(sense_pred)[:, 1]

        recall_num = np.sum(sense_scores > threshold)
        recall_rate = recall_num / len(sense_scores)

        print("common_porn recall_num = {}".format(recall_num))
        print("common_porn recall_rate = {:.2f} %".format(recall_rate * 100))

        # cartoon_porn dataset
        sense_pred = []
        fsense = codecs.open(cartoon_porn_file, 'r', 'utf-8').read().split('\n')
        for line in fsense:
            tline = line.strip()
            if tline == "":
                continue
            img_name, pred_str = tline.split('\t')
            pred = np.max(json.loads(pred_str), axis=0)
            sense_pred.append(pred)
        sense_pred = np.array(sense_pred)
        sense_scores = clf2.predict_proba(sense_pred)[:, 1]

        recall_num = np.sum(sense_scores > threshold)
        recall_rate = recall_num / len(sense_scores)

        print("cartoon_porn recall_num = {}".format(recall_num))
        print("cartoon_porn recall_rate = {:.2f} %".format(recall_rate * 100))

        # pornhub dataset
        sense_pred = []
        fsense = codecs.open(porn_hub_file, 'r', 'utf-8').read().split('\n')
        for line in fsense:
            tline = line.strip()
            if not tline:
                continue
            img_name, pred_str = tline.split('\t')
            pred = np.max(json.loads(pred_str), axis=0)
            sense_pred.append(pred)
        sense_pred = np.array(sense_pred)
        sense_scores = clf2.predict_proba(sense_pred)[:, 1]

        recall_num = np.sum(sense_scores > threshold)
        recall_rate = recall_num / len(sense_scores)

        print("pornhub recall_num = {}".format(recall_num))
        print("pornhub recall_rate = {:.2f} %".format(recall_rate * 100))


        # 色情模型的正常数据集  max_mode
        porn_norm_file = '/data/wangruihao/centernet/Centernet_2/EfficientNet/results/cocofun_porn_new.txt'
        sense_pred = []
        fsense = codecs.open(porn_norm_file, 'r','utf-8').read().split('\n')
        print(len(fsense))   # 4999
        print(fsense[3027])
        # print(fsense[0].split("\t")[2])
        # fs_data = np.array(json.loads(fsense[0].split("\t")[2]))
        # print(fs_data)
        # soft = nn.Softmax()
        # pred_soft = soft(torch.from_numpy(np.array(json.loads(fsense[0].split("\t")[2]))))
        # print(pred_soft)
        for line in fsense:
            # print(i)
            tline = line.strip()
            if len(tline) == 0:
                continue
            img_name, data_type, pred_str = tline.split('\t')
            pred = json.loads(pred_str)

            if len(np.array(pred).shape)>1:
                pred = np.max(json.loads(pred_str), axis=0)
                if len(pred)==11:
                    sense_pred.append(pred)
            elif len(pred)==11:
                sense_pred.append(pred)
            else:
                continue

            i = i+1
        print('='*60)
        print("海外 normal examples：")
        print(sense_pred[:2])
        sense_pred = np.array(sense_pred)
        print('numbers of test porn_normal : ',len(sense_pred))
        # softmax = nn.Softmax(dim=1)
        # sense_pred = softmax(torch.from_numpy(sense_pred)).numpy()
        print("after softmax examples：")
        print(sense_pred[:2])
        print(clf2.predict_proba(sense_pred)[:2000, :])
        sense_scores = clf2.predict_proba(sense_pred)[:, 1]

        injudge_num = np.sum(sense_scores > threshold)
        injudge_rate = injudge_num / len(sense_scores)

        print("porn normal injudge_num = {}".format(injudge_num))
        print("porn normal injudge_rate = {:.2f} %".format(injudge_rate * 100))
        print('='*100)


        #===============================================================
        # 色情模型的正常数据集   solo模式
        porn_norm_file = '/data/wangruihao/centernet/Centernet_2/EfficientNet/results/cocofun_porn_new.txt'
        sense_pred = []
        fsense = codecs.open(porn_norm_file, 'r', 'utf-8').read().split('\n')
        print(len(fsense))  # 4999

        for line in fsense:
            tline = line.strip()
            if len(tline) == 0:
                continue
            img_name, data_type, pred_str = tline.split('\t')
            pred = json.loads(pred_str)
            pred = np.array(pred)

            if len(pred.shape) > 1:
                sense_pred.append(pred)
            elif len(pred) == 11:
                sense_pred.append(pred[None,:])
            elif len(pred) < 11:
                continue
        sense_pred = np.concatenate(sense_pred,axis = 0)
        print('=' * 60)

        print('numbers of test porn_normal : ', (sense_pred.shape[0]))
        # softmax = nn.Softmax(dim=1)
        # sense_pred = softmax(torch.from_numpy(sense_pred))
        # print(clf2.predict_proba(sense_pred)[:2000, :])
        sense_scores = clf2.predict_proba(sense_pred)[:, 1]

        injudge_num = np.sum(sense_scores > threshold)
        injudge_rate = injudge_num / len(sense_scores)

        print("porn normal injudge_num = {}".format(injudge_num))
        print("porn normal injudge_rate = {:.2f} %".format(injudge_rate * 100))

        # ===============================================================
        # 色情模型的正常数据集   帖子召回模式
        porn_norm_file = '/data/wangruihao/centernet/Centernet_2/EfficientNet/results/cocofun_porn_new.txt'
        sense_pred = []
        fsense = codecs.open(porn_norm_file, 'r', 'utf-8').read().split('\n')
        print(len(fsense))  # 4999

        sense_count = 0
        injudge_count = 0
        invalid_sense = 0
        video_count = 0
        image_count = 0
        injudge_img = 0
        for i,line in enumerate(fsense):
            tline = line.strip()
            if len(tline) == 0:
                continue
            img_name, data_type, pred_str = tline.split('\t')
            pred = json.loads(pred_str)
            pred = np.array(pred)

            if len(pred.shape) > 1:
                video_count += 1

                sense_scores = clf2.predict_proba(pred)[:, 1]
                temp_sum = np.sum(sense_scores>threshold)
                if temp_sum:
                    # print(sense_scores>threshold)
                    injudge_img += temp_sum
                    injudge_count +=1
                    # print(line)
            elif len(pred) == 11:
                image_count += 1
                sense_scores = clf2.predict_proba(pred)
                if sense_scores>threshold:
                    injudge_count += 1
            elif len(pred) < 11:
                # print(line)
                invalid_sense +=1
                continue
            sense_count += 1

        print('=' * 60)

        print("total sense: ",len(fsense))  # 4999
        print("video count: ",video_count)
        print("total injudge images:", injudge_img)
        print("image count: ",image_count)
        print("invalid sense count:",invalid_sense)

        # print('numbers of test porn_normal : ', (sense_pred.shape[0]))
        # softmax = nn.Softmax(dim=1)
        # sense_pred = softmax(torch.from_numpy(sense_pred))
        # print(clf2.predict_proba(sense_pred)[:2000, :])
        # sense_scores = clf2.predict_proba(sense_pred)[:, 1]


        injudge_rate = injudge_count / sense_count

        print("porn normal injudge_num = {}".format(injudge_count))
        print("porn normal injudge_rate = {:.2f} %".format(injudge_rate * 100))

    elif running_mode == 3:
        with open('/data1/zhaoshiyu/temp/porn_c11_lr_0229_py3.pickle', 'rb') as f:
            clf2 = pickle.load(f)
            print(clf2.coef_)

        raw_recall_count = 353
        high_expose_recall_set = 200
        huohua_raw_recall = 8
        huohua_new_recall = 4

        score_list = []
        norm_result_file = "/data1/zhaoshiyu/porn_result_temp/porn0229_normal_result.txt"
        fr = codecs.open(norm_result_file, 'r', 'utf-8').read().split('\n')

        cmp_score_list = []
        ctp_score_list = []
        for line in fr:
            tline = line.strip()
            if tline == "":
                continue
            img_name, pred_str = tline.split('\t')
            temp_x = np.array(json.loads(pred_str))

            cmp_score_list.append(np.max(temp_x[:, 6]))
            ctp_score_list.append(np.max(temp_x[:, 2]))

            temp_proba = clf2.predict_proba(temp_x)[:, 1]
            temp_score = np.max(temp_proba)
            score_list.append(temp_score)

        score_list.sort(reverse=True)
        new_threshold = score_list[raw_recall_count]
        high_exp_thres = score_list[high_expose_recall_set]

        huohua_new_threshold = score_list[huohua_new_recall]

        sense_file = "/data1/zhaoshiyu/porn_result_temp/porn0229_sensetive_result.txt"
        raw_hit_count = 2940
        new_hit_count = 0
        huohua_hit_count = 0

        porn_score_list = []
        fs = codecs.open(sense_file, 'r', 'utf-8').read().split('\n')
        for line in fs:
            tline = line.strip()
            if tline == "":
                continue

            items = tline.split('\t')
            label = items[0]
            img_name = items[1]
            pred_str = items[2]
            temp_x = np.array(json.loads(pred_str))
            temp_proba = clf2.predict_proba(temp_x)[:, 1]
            temp_score = np.max(temp_proba)
            if 'porn' in label:
                porn_score_list.append(temp_score)

            if temp_score > new_threshold:
                if 'porn' in label:
                    new_hit_count += 1

        print("进审量不变，严重色情case召回量对比")
        print("new threshold = {} / {}".format(new_threshold, high_exp_thres))
        print("new_hit_count = {}".format(new_hit_count))
        print("raw_hit_count = {}\n".format(raw_hit_count))
        print("huohua_hit_count = {}".format(huohua_hit_count))

        porn_score_list.sort(reverse=True)
        equal_threshold = porn_score_list[raw_hit_count]
        print(len(porn_score_list))
        # high_equal_threshold = porn_score_list[raw_high_hit_stat[0]]
        # print(high_equal_threshold)

        after_recall = np.sum(score_list > equal_threshold)
        print("\n维持召回严重色情case数量不变，进审量对比")
        print("equal_threshold = {}".format(equal_threshold))
        print("raw: {}, new: {}".format(raw_recall_count, after_recall))

        # 火花阈值确定
        cmp_low = 0.0025
        ctp_low = 0.0005
        cmp_high = 0.0003
        ctp_high = 0.0001
        total_normal = len(score_list)
        cmp_score_list.sort(reverse=True)
        ctp_score_list.sort(reverse=True)

        print("common_porn high threshold = {}".format(cmp_score_list[int(np.round(total_normal * cmp_high))]))
        print("cartoon_porn high threshold = {}".format(ctp_score_list[int(np.round(total_normal * ctp_high))]))
        print("common_porn low threshold = {}".format(cmp_score_list[int(np.round(total_normal * cmp_low))]))
        print("cartoon_porn low threshold = {}".format(ctp_score_list[int(np.round(total_normal * ctp_low))]))






        # 视频进审量评估
        # video_normal_file = "/Users/zuiyou/Downloads/pull_video_normal_0102.txt"
        # fv = codecs.open(video_normal_file, 'r', 'utf-8').read().split('\n')
        #
        # temp_input = []
        # temp_id = -1
        # raw_video_recall = 0
        # new_video_recall = 0
        # idset = set()
        # for line in fv:
        #     tline = line.strip()
        #     if tline == "":
        #         continue
        #     items = tline.split('\t')
        #     id = int(items[0])
        #     idset.add(id)
        #     pred_str = items[2]
        #     if id != temp_id:
        #         if len(temp_input) > 0:
        #             temp_input = np.array(temp_input)
        #             temp_proba = clf2.predict_proba(temp_input)[:, 1]
        #             temp_score = np.max(temp_proba)
        #             raw_score = 1 - np.min(np.sum(temp_input[:, [2, 7]], axis=1))
        #             if raw_score > 0.339:
        #                 raw_video_recall += 1
        #             if temp_score > equal_threshold:
        #                 new_video_recall += 1
        #
        #         temp_input = []
        #         temp_id = id
        #
        #     temp_x = json.loads(pred_str)
        #     temp_input.append(temp_x)
        #
        # print(len(idset))
        # print(raw_video_recall)
        # print(new_video_recall)

    elif running_mode == 5:
        suffix_file_path = "/data1/zhaoshiyu/porn_result_temp/porn0623_ep164_suffix_result.txt"
        model_name = suffix_file_path.split('/')[-1].split('_suffix')[0]
        pickle_path = '/data1/zhaoshiyu/temp/{}_lr.pickle'.format(model_name)

        sexy_labels = ['cartoon_sexy', 'female_sexy', 'cartoon_porn', 'common_porn']
        x = []
        y = []
        fsuffix = codecs.open(suffix_file_path, 'r', 'utf-8').read().split('\n')
        for line in fsuffix:
            tline = line.strip()
            if tline == "":
                continue
            cls, img_name, pred_str = tline.split('\t')
            pred = np.max(json.loads(pred_str), axis=0)

            if cls == 'normal':
                x.append(pred)
                y.append(0)
            elif cls in sexy_labels:
                x.append(pred)
                y.append(1)

        x = np.array(x)
        y = np.array(y)

        trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.3, random_state=4)
        print("Train X: ", trainX.shape)
        print("Train Y: ", trainY.shape)
        print("Test X: ", testX.shape)
        print("Test Y: ", testY.shape)

        classifier = LogisticRegression(class_weight={0:0.5, 1:0.5})
        classifier.fit(trainX, trainY)

        score = classifier.score(testX, testY)
        print("Test score: {0:.2f}%".format(100*score))
        Y = classifier.predict(x)

        # save the LR classifier
        pickle.dump(classifier, open(pickle_path, 'wb'))

        with open(pickle_path, 'rb') as f:
            clf = pickle.load(f)
            print(clf.coef_)

        im_normal_file = "/data1/zhaoshiyu/porn_result_temp/{}_im_normal_result.txt".format(model_name)
        im_sense_file = "/data1/zhaoshiyu/porn_result_temp/{}_im_result.txt".format(model_name)

        normal_pred = []
        fnorm = codecs.open(im_normal_file, 'r','utf-8').read().split('\n')
        for line in fnorm:
            tline = line.strip()
            if tline == "":
                continue
            img_name, pred_str = tline.split('\t')
            pred = np.max(json.loads(pred_str), axis=0)
            normal_pred.append(pred)
        normal_pred = np.array(normal_pred)
        normal_scores = clf.predict_proba(normal_pred)[:, 1].tolist()
        normal_scores.sort(reverse=True)

        threshold = normal_scores[int(np.round(len(normal_scores) * 0.0005))]
        print("threshold = {}".format(threshold))

        sense_pred = []
        fsense = codecs.open(im_sense_file, 'r', 'utf-8').read().split('\n')
        for line in fsense:
            tline = line.strip()
            if tline == "":
                continue
            img_name, pred_str = tline.split('\t')
            pred = np.max(json.loads(pred_str), axis=0)
            sense_pred.append(pred)
        sense_pred = np.array(sense_pred)
        sense_scores = clf.predict_proba(sense_pred)[:, 1]

        recall_num = np.sum(sense_scores > threshold)
        recall_rate = recall_num / len(sense_scores)

        print("recall_num = {}".format(recall_num))
        print("recall_rate = {:.2f} %".format(recall_rate * 100))





# im recall_num = 5060
# im recall_rate = 98.44 %
# common_porn recall_num = 1389
# common_porn recall_rate = 97.27 %
# cartoon_porn recall_num = 1411
# cartoon_porn recall_rate = 95.14 %
# pornhub recall_num = 8278
# pornhub recall_rate = 95.83 %
# porn normal injudge_rate = 4.60 %
# porn normal injudge_rate = 6.25 %
# porn normal injudge_rate = 21.80 %