import hashlib
import json
import string
import random
import numpy
from model import SRCNN
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import numpy as np
import os, sys
import PIL.Image as pil_image
import h5py
import glob
import cv2 as cv
import PIL.Image as Image
import datetime


# RGB 图像转换为 YCbCr 颜色空间
def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


# YCbCr 图像转换为 RGB 颜色空间
def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


# 计算两个图像的均方误差来得到它们之间的差异性
def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


# 获取该目录下的所有文件名
def get_filenames(path: str) -> list:
    filenames = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            filenames.append(file)
    return filenames


# 检测指定目录里是否有指定目录，若没有则创建爱你
def create_dir(path: str, filename: str) -> None:
    if not os.path.exists(os.path.join(path, filename)):
        os.mkdir(os.path.join(path, filename))


# 将 opencv 的图像转换为 PIL.Image
def cv2_to_pil(cv2_img) -> Image:
    cv2_im_rgb = cv.cvtColor(cv2_img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_im_rgb)
    return pil_img


# 将 PIL.Image 转换为 opencv 的图像
def pil_to_cv2(pil_img):
    cv2_im = cv.cvtColor(numpy.array(pil_img), cv.COLOR_RGB2BGR)
    return cv2_im


# 判断该值是否能被转换为 int 类型
# 有一说一，这个方法很野233 建议使用 isnumeric() 方法
def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


# 判断该字符串是否能转为 float 类型
# 垃圾方法233 但是很无脑，很好用
def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# 获取随机生成的哈希值
def get_hash(length=10) -> str:
    # 生成随机字符串
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    # 计算哈希值
    hash_object = hashlib.sha256(random_string.encode())
    return hash_object.hexdigest()


# 获得当前时间，精确到秒
def get_time() -> str:
    now = datetime.datetime.now()  # 获取当前时间
    return now.strftime("%Y-%m-%d %H:%M:%S")  # 将时间调整为字符串格式


# 将当前字典写入本息 JSON
def write_dict_to_json(path: str, d: dict) -> None:
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(d, outfile, ensure_ascii=False)


# dict to str
def dict_to_str(d: dict) -> str:
    return json.dumps(
        d,
        # sohrt_keys=True,
        indent=4,
        separators=('', ': '),
        ensure_ascii=False
    ).replace('{', '').replace('}', '').replace('\"', '')


# 读取json文件
def read_json(file: str) -> dict:
    with open(file, 'r', encoding='utf-8') as data:
        return json.load(data)


# 是否存在该路径，若不存在则创建
def check_path(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} 已创建")
    return path
