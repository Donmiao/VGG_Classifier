from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # "OMP: Error #15: Initializing libiomp5md.dll"
###################################################################################
def image_loader(image_name):
    """Loading Image"""
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def Image_trans(tensor):
    image = tensor.cpu().clone()            # 为避免因image修改影响tensor的值，这里采用clone
    image = image.squeeze(0)                # 去掉一个维度
    unloader = transforms.ToPILImage()      # reconvert into PIL image
    image = unloader(image)
    return image


class ContentLoss(nn.Module):
    """内容损失函数"""
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # 必须要用detach来分离出target，这时候target不再是一个Variable
        # 这是为了动态计算梯度，否则forward会出错，不能向前传播
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    """格拉姆矩阵"""
    a, b, c, d = input.size()               # a is batch size. b is number of channels. c is height and d is width.
    features = input.view(a * b, c * d)     # x矩阵
    G = torch.mm(features, features.t())    # 计算内积（矩阵 * 转置矩阵）
    return G.div(a * b * c * d)             # 除法（格拉姆矩阵 - 归一化处理）


class StyleLoss(nn.Module):
    """风格损失函数"""
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    """标准化处理"""
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)    # self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)      # self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


###################################################################################
# 卷积网络不同层学到的图像特征是不一样的。研究发现，使用靠近底层但不能太靠近的层来衡量图像内容比较理想。
#       （1）靠近底层的卷积层（输入端）：学到的是图像局部特征。如：位置、形状、颜色、纹理等。
#       （2）靠近顶部的卷积层（输出端）：学到的图像特征更全面、更抽象，但也会丢失图像详细信息。
###################################################################################
def get_style_model_and_losses(cnn,                                     # VGG19网络主要用来做内容识别
                               normalization_mean, normalization_std, style_img, content_img,
                               content_layers=['conv_4'],               # 为计算内容损失和风格损失，指定使用的卷积层
                               # 研究发现：使用前三层已经能够达到比较好的内容重建工作，而后两层保留一些比较高层的特征。
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)                    # 深复制
    normalization = Normalization(normalization_mean, normalization_std).to(device)  # 标准化
    content_losses = []                         # 初始化（内容）损失值
    style_losses = []                           # 初始化（风格）损失值
    model = nn.Sequential(normalization)        # 使用sequential方法构建模型

    i = 0                                       # 每次迭代增加1
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):        # isinstance(object, classinfo)：判断两个对象类型是否相同。
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)          # 添加指定的网络层（conv、relu、pool、bn）到模型中

        if name in content_layers:                          # 累加内容损失
            target = model(content_img).detach()            # 前向传播
            content_loss = ContentLoss(target)              # 内容损失
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:                            # 累加风格损失
            target_feature = model(style_img).detach()      # 前向传播
            style_loss = StyleLoss(target_feature)          # 风格损失
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 我们需要对在内容损失和风格损失之后的层进行修剪
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses


def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=600, style_weight=1000000, content_weight=1):
    """风格迁移"""
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])       # requires_grad_()：需要对输入图像进行梯度计算。采用LBFGS优化方法
    run = [0]
    while run[0] <= num_steps:                                  # 批次数
        def closure():
            input_img.data.clamp_(0, 1)                         # 将输入张量的每个元素收紧到区间内，并返回结果到一个新张量。
            optimizer.zero_grad()                               # 梯度清零
            model(input_img)                                    # 前向传播

            # 计算当前批次的损失
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss                          # （叠加）风格得分
            for cl in content_losses:
                content_score += cl.loss                        # （叠加）内容得分
            style_score *= style_weight                         # 风格权重系数：1000000
            content_score *= content_weight                     # 内容权重系数：1
            loss = style_score + content_score                  # 总损失
            loss.backward()                                     # 反向传播

            # 打印损失值
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()
            return style_score + content_score

        optimizer.step(closure)                                 # 梯度更新
    input_img.data.clamp_(0, 1)
    return input_img


###################################################################################
# （1）图像预处理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                   # 可用设备
# torchvision.transforms.Normalize		# Normalize的作用是用均值和标准差对Tensor进行标准化处理。
loader = transforms.Compose([transforms.Resize((512, 600)), transforms.ToTensor()])     # 数据预处理
style_img = image_loader("oilpainting.jpg")                                      # 风格图像
content_img = image_loader("cat.jpg")                                  # 内容图像
print("style size:", style_img.size())
print("content size:", content_img.size())
assert style_img.size() == content_img.size(), "we need to import style and content images of the same size"
###################################################################################
# （2）风格迁移
cnn = models.vgg19(pretrained=True).features.to(device).eval()                          # 下载预训练模型
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)                 # 标准化（均值）
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)                  # 标准化（标准差）
input_img = content_img.clone()                                                         # 复制图像
# input_img = torch.randn(content_img.data.size(), device=device)       				# 随机添加白噪声
output_img = run_style_transfer(cnn=cnn, normalization_mean=cnn_normalization_mean,
                                normalization_std=cnn_normalization_std,
                                content_img=content_img, style_img=style_img, input_img=input_img,
                                num_steps=500, style_weight=1000000, content_weight=1)
###################################################################################
# （3）画图
style_img = Image_trans(style_img)          # 格式转换
content_img = Image_trans(content_img)      # 格式转换
output_img = Image_trans(output_img)        # 格式转换
plt.subplot(131), plt.imshow(style_img, 'gray'),    plt.title('style_img')
plt.subplot(132), plt.imshow(content_img, 'gray'),  plt.title('content_img')
plt.subplot(133), plt.imshow(output_img, 'gray'),   plt.title('style_img + content_img')
plt.show()

