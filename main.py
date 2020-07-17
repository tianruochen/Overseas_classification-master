import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from efficientNet import PornModel,UnPorn_Model
# import cv2
from PIL import Image


if __name__ == '__main__':
    # testimg = cv2.imread("testimg.jpeg")
    testimg = Image.open("testimg.jpeg")
    # Porn_classification = PornModel('porn_cocofun')
    # risk,pred = Porn_classification.predict_img_api(testimg)

    # Unporn_classification = UnPorn_Model('unpron_cocofun')
    Unporn_classification = UnPorn_Model('porn_cocofun')
    for i in range(1000):
        risk,pred = Unporn_classification.predict_img_api(testimg)
        print(risk,pred)

