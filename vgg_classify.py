# vgg_classify.py
import numpy as np
import rasterio
import torch
import argparse
import vgg_model

from PIL import Image
from torchvision.transforms import ToTensor

# class_names = {"neutral": 0, "angry": 1, "depressed": 2, "drunk": 3, "happy": 4, "heavy": 5, "hurried": 6, "lazy" : 7, "old": 8, "proud": 9, "robot": 10, "sneaky": 11, "soldier": 12, "strutting": 13, "zombie": 14}
class_names = ["neutral", "angry", "depressed", "happy", "heavy", "old", "proud","strutting"]

class Classify:
    def __init__(self, model_path, img_size, num_class, use_gpu=False):
        self.model_path = model_path
        self.img_size = img_size
        self.num_class = num_class
        self.use_gpu = use_gpu
        self.init_model()

    def init_model(self):
        # initialize model
        if self.use_gpu:
            self.net = vgg_model.VGG(img_size=self.img_size, input_channel=3, num_class=self.num_class).cuda()
        else:
            self.net = vgg_model.VGG(img_size=self.img_size, input_channel=3, num_class=self.num_class)

        # load model data
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.eval()

    def classify(self, image_path):
        with rasterio.open(image_path) as image:
            image_array = image.read()
        img = ToTensor()(image_array)
        img = np.array(img, np.float32).transpose(1, 2, 0)
        img = torch.tensor(img)
        img = img.unsqueeze(0)

        if self.use_gpu:
            img = img.cuda()
        output = self.net(img)

        _, indices = torch.max(output, 1)
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        precision = percentage[int(indices)].item()
        result = class_names[indices]

        print(f"Input File: {image_path} \npredicted: {result}\n percentage: {precision}% ")

        print("====================")
        i = 0
        result_text = ""
        for output_element in output[0]:
            percentage_element = torch.nn.functional.softmax(output, dim=1)[0] * 100
            perc_element = percentage[i].item()
            class_name = class_names[i]
            result_text+=(f"{class_name}: {perc_element}% \n")
            i += 1

        print(result_text)

        return precision, result, result_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, default='motion_data/angry-200.tiff', help='images path')
    parser.add_argument('-model_path', type=str, default='checkpoint/epoch_42-best-acc_0.9312499761581421.pth', help='model path')
    parser.add_argument('-img_size', type=int, default=64, help='the size of image, mutiple of 32')
    parser.add_argument('-num_class', type=int, default=8, help='the number of class')
    parser.add_argument('-gpu', default= False, help='use gpu or not')

    opt = parser.parse_args()

    classify = Classify(opt.model_path,opt.img_size,opt.num_class,opt.gpu)
    classify.init_model()
    classify.classify(opt.img_path)



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-img_path', type=str, default='motion_data/angry-200.tiff', help='images path')
#     parser.add_argument('-model_path', type=str, default='checkpoint/epoch_42-best-acc_0.9312499761581421.pth', help='model path')
#     parser.add_argument('-img_size', type=int, default=64, help='the size of image, mutiple of 32')
#     parser.add_argument('-num_class', type=int, default=8, help='the number of class')
#     parser.add_argument('-gpu', default= False, help='use gpu or not')
#
#     opt = parser.parse_args()
#
#     # initialize vgg
#     if opt.gpu:
#         net = vgg_model.VGG(img_size=opt.img_size, input_channel=3, num_class=opt.num_class).cuda()
#     else:
#         net = vgg_model.VGG(img_size=opt.img_size, input_channel=3, num_class=opt.num_class)
#
#     # load model data
#     net.load_state_dict(torch.load(opt.model_path))
#     net.eval()
#
#     # img = Image.open(opt.img_path)
#     # if len(img.split()) == 1:
#     #     img = img.convert("RGB")
#     # img = img.resize((opt.img_size, opt.img_size))
#     # image_to_tensor = ToTensor()
#     # img = image_to_tensor(img)
#
#     with rasterio.open(opt.img_path) as image:
#         image_array = image.read()
#     img = ToTensor()(image_array)
#     img = np.array(img, np.float32).transpose(1, 2, 0)
#     img = torch.tensor(img)
#     img = img.unsqueeze(0)
#
#     if opt.gpu:
#         img = img.cuda()
#     output = net(img)
#
#     _, indices = torch.max(output, 1)
#     percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
#     perc = percentage[int(indices)].item()
#     result = class_names[indices]
#     print(f"Input File: {opt.img_path} \npredicted: {result}, percentage: {perc}% ")
#
#     print("====================")
#     i = 0
#     for output_element in output[0]:
#         percentage_element = torch.nn.functional.softmax(output, dim=1)[0] * 100
#         perc_element = percentage[i].item()
#         result = class_names[i]
#         print(f"{result}: {perc_element}% ")
#         i+=1
