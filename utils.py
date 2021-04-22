import csv
import math
import torch
import numpy as np
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F


# 加载图片路径命名和对应的类别
def get_truth_info(csv_file):
    img_paths = []   # 记录攻击图片的path
    labels = []   # 每个id对应的真实类别
    with open(csv_file) as f:
        lines = csv.DictReader(f, delimiter=',')
        for line in lines:
            img_paths.append(line['ImageId'])    # 图片名称
            labels.append(line['TrueLabel'])     # 该图片对应的label
    return img_paths, labels


# 图片标准化处理(不针对对抗训练模型)
class imgnormalize(nn.Module):
    def __init__(self):
        super(imgnormalize, self).__init__()
        self.mean = [0.485, 0.456, 0.406]  # 均值
        self.std = [0.229, 0.224, 0.225]   # 标准差

    '''
    return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]
    '''
    def forward(self, x):
        for i in range(len(self.mean)):
            x[i] = (x[i]-self.mean[i])/self.std[i]
        return x


# 高斯核
def get_gaussian_kernel(kernel_size=15, sigma=2, channels=3, use_cuda=False):
    x_coord = torch.arange(kernel_size)   # x坐标系
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()  # kernel_size*kernel_size*2
    mean = (kernel_size - 1)/2.   # 均值
    variance = sigma**2.          # 方差
    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))  # 二维高斯函数
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)                           # 复制三次
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, padding=(kernel_size-1)//2, bias=False)
    if use_cuda:
        gaussian_filter.weight.data = gaussian_kernel.cuda()   # 使用cuda
    else:
        gaussian_filter.weight.data = gaussian_kernel          # 利用高斯函数值更新卷积核的权重
    gaussian_filter.weight.requires_grad = False               # 不更新高斯核的参数
    return gaussian_filter


# TI核 定义
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def DI(x, resize_rate=1.15, diversity_prob=0.7):
    assert resize_rate >= 1.0                                   # 随机放大的尺度上限
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0      # 执行DI的概率
    img_size = x.shape[-1]                        # 获取输入图片的尺度
    img_resize = int(img_size * resize_rate)      # DI最大缩放尺度
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)    # 随机尺度
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)  # 双线性插值
    h_rem = img_resize - rnd                      # 需要填充的边界
    w_rem = img_resize - rnd                      # 需要填充的边界
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)     # 顶部填充
    pad_bottom = h_rem - pad_top                                                        # 底部填充
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)    # 左边填充
    pad_right = w_rem - pad_left                                                        # 右边填充
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)  # 填充
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret


# 高斯平滑操作（TI使用，增加迁移性）
class GaussianSmooth(nn.Module):
    def __init__(self, kernlen=21, nsig=3):
        super(GaussianSmooth, self).__init__()
        self.kernlen = kernlen   # 核大小
        self.nsig = nsig         # 核内数值初始化大小
        kernel = self.gkern().astype(np.float32)  # 2D 高斯核
        self.stack_kernel = np.stack([kernel, kernel, kernel])
        self.stack_kernel = np.expand_dims(self.stack_kernel, 0)
        self.stack_kernel = torch.from_numpy(self.stack_kernel).cuda()   # 3D
        print('using TI with kernel size ', self.stack_kernel.size(-1))
        self.stride = 1

    # 返回二维高斯核
    def gkern(self):
        x = np.linspace(-self.nsig, self.nsig, self.kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    # 噪声重复添加
    def forward(self, x):
        padding = int((((self.stride - 1) * x.size(-1) - self.stride + self.stack_kernel.size(-1)) / 2) + 0.5)
        noise = F.conv2d(x, self.stack_kernel, stride=self.stride, padding=padding)
        noise = noise / torch.mean(torch.abs(noise), [1, 2, 3], keepdim=True)
        x = x + noise
        return x