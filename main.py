import os
import sys
import time
import torch
import random
import argparse
import numpy as np
from tqdm import *
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torchvision import models, transforms
from utils import get_truth_info, imgnormalize, gkern, DI, get_gaussian_kernel

# NES 偏差生成，用于矫正梯度方向
variance = np.random.uniform(0.5,1.5)
neg_perturbations = - variance
variance.extend(neg_perturbations)
variance.append([0])
liner_interval = variance


def parse_arguments():
    parser = argparse.ArgumentParser(description='transfer_attack')
    parser.add_argument('--source_model', nargs="+", default=['resnet50'])   # 替代模型
    parser.add_argument('--batch_size', type=int, default=41)
    parser.add_argument('--img_size', type=int, default=500)
    parser.add_argument('--max_iterations', type=int, default=50)
    parser.add_argument('--lr', type=eval, default=1.0/255.)
    parser.add_argument('--linf_epsilon', type=float, default=32)
    parser.add_argument('--di', type=eval, default="True")
    parser.add_argument('--input_path', type=str, default='inputdata')
    parser.add_argument('--result_path', type=str, default='results')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    result_folder = args.result_path     # 结果存放的目录
    input_folder = os.path.join(args.input_path,'images/')    # 输入图片的文件夹
    adv_img_folder = os.path.join(result_folder, 'images')    # 对抗样本保存文件夹
    if not os.path.exists(adv_img_folder):
        os.makedirs(adv_img_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)       # 如果不存在目录则创建目录
    norm = imgnormalize()              # 标准化处理类
    Totensor = transforms.Compose([transforms.ToTensor()])                        # tensor化
    device = torch.device("cuda:0")                 # GPU ID
    source_model_names = args.source_model          # 替代模型
    num_source_models = len(source_model_names)     # 替代模型的数量
    source_models = []                              # 根据替代模型的名称分别加载对应的网络模型
    for model_name in source_model_names:
        print("Loading: {}".format(model_name))
        source_model = models.__dict__[model_name](pretrained=True).eval()
        for param in source_model.parameters():
            param.requires_grad = False             # 不可导
        source_model.to(device)                     # 计算环境
        source_models.append(source_model)          # ensemble
    seed_num=1                                      # 随机种子
    random.seed(seed_num)                           # random设置随机种子
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)                     # torch随机种子
    torch.backends.cudnn.deterministic = True
    # TI 参数设置
    channels = 3                                      # 3通道
    kernel_size = 5                                   # kernel大小
    kernel = gkern(kernel_size, 1).astype(np.float32)      # 3表述kernel内元素值得上下限
    gaussian_kernel = np.stack([kernel, kernel, kernel])   # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)   # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False, padding=7)
    gaussian_filter.weight.data = gaussian_kernel          # 高斯滤波，高斯核的赋值
    # 划分数据，多卡攻击
    image_id_list, label_ori_list = get_truth_info(os.path.join(args.input_path,'dev.csv'))
    num_batches = np.int(np.ceil(len(image_id_list)/args.batch_size))                                  # 总共待攻击目标必须可以被BS整除
    gaussian_smoothing = get_gaussian_kernel(kernel_size=5, sigma=1, channels=3, use_cuda=True)        # 高斯核（过滤部分高频信息） 5，1
    print('start atttacking....')
    for k in tqdm(range(0,num_batches)):
        time.sleep(0.1)
        X_ori = torch.zeros(args.batch_size, 3, args.img_size, args.img_size).to(device)     # 输入大小的初始化
        delta = torch.zeros_like(X_ori,requires_grad=True).to(device)                        # 噪声大小的初始化
        for i in range(args.batch_size):
            X_ori[i] = Totensor(Image.open(input_folder + image_id_list[k * args.batch_size + i]))  # 输入大小的赋值
        X_ori = gaussian_smoothing(X_ori)                                                           # 对输入图片进行高斯滤波
        # 获取真实的label信息
        labels_gt = torch.tensor(label_ori_list[k*args.batch_size:(k*args.batch_size+args.batch_size)]).to(device)
        grad_momentum = 0    # 梯度动量法
        for t in range(args.max_iterations):
            g_temp = []
            for tt in range(len(liner_interval)):
                if args.di:
                    X_adv = X_ori + delta
                    X_adv = DI(X_adv)         # di操作
                    X_adv = nn.functional.interpolate(X_adv, (224, 224), mode='bilinear', align_corners=False)        # 插值到224
                else:
                    c = liner_interval[tt]
                    X_adv = X_ori + c * delta  # 如果使用了DI，则不用顶点浮动
                    X_adv = nn.functional.interpolate(X_adv, (224, 224), mode='bilinear', align_corners=False)  # 插值到224
                logits = 0
                for source_model_n, source_model in zip(source_model_names, source_models):
                    logits += source_model(norm(X_adv))               # ensemble操作
                logits /= num_source_models
                loss = -nn.CrossEntropyLoss()(logits,labels_gt)       # 交叉熵
                loss.backward()                                       # 梯度回传
                # MI + TI 操作
                grad_c = delta.grad.clone()                           # 同时使用MI和TI
                grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
                #grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+0.5*grad_momentum   # 1
                grad_a = grad_c
                grad_momentum = grad_a
                g_temp.append(grad_a)
            g0 = 0.0
            for j in range(7):
                g0 += g_temp[j]      # 求均值，抵消噪声【多次DI随机，消除噪声，保留有效信息】
            g0 = g0 / 7.0
            delta.grad.zero_()                                      # 梯度清零
            # 无穷范数攻击
            delta.data=delta.data-args.lr * torch.sign(g0)
            delta.data=delta.data.clamp(-args.linf_epsilon/255.,args.linf_epsilon/255.)
            delta.data=((X_ori+delta.data).clamp(0,1))-X_ori       # 噪声截取操作

        for i in range(args.batch_size):
            adv_final = (X_ori+delta)[i].cpu().detach().numpy()
            adv_final = (adv_final*255).astype(np.uint8)
            file_path = os.path.join(adv_img_folder, image_id_list[k * args.batch_size + i])
            adv_x_255 = np.transpose(adv_final, (1, 2, 0))
            im = Image.fromarray(adv_x_255)
            im.save(file_path,quality=99)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
