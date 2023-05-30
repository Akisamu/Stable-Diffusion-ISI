"""make variations of input image"""
import os
import sys
import torch
import numpy
import math
import PIL
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm, trange
from itertools import islice
from einops import repeat
from torch import autocast
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from utils import *
import cv2 as cv

HOME_DIR = '.'


# stable diffusion 加载流程的
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


# sd_process 中的图像加载
# https://github.com/CompVis/stable-diffusion/blob/main/scripts/img2img.py
def load_img(image: Image):
    image = image.convert("RGB")
    w, h = image.size
    print(f"loaded input image.")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = numpy.array(image).astype(numpy.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


class StableDiffusion:
    def __init__(self, config_path: str, model_path: str):
        """ 以下为不可变参数 """
        self.C = 4
        # 高
        self.H = 512
        # 宽
        self.W = 512
        # 下采样因子
        self.f = 8
        # 采样数量
        self.n_samples = 1
        # 确定性因子
        self.ddim_eta = 0.0
        # 迭代次数
        self.n_iter = 1
        # 起始编码 (没用)
        self.start_code = None
        # 加载配置文件
        self.config = OmegaConf.load(config_path)
        # 从加载的配置文件创建模型
        self.model = load_model_from_config(self.config, model_path)
        # 设备
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # 将模型送到CUDA设备 或者 CPU
        self.model = self.model.to(self.device)
        # 创建ddim 采样器
        self.sampler = DDIMSampler(self.model)
        """ 以下为可变参数 """
        self.batch_size = 1
        # CFG 缩放
        self.scale = 7
        # 采样步数
        self.ddim_steps = 100
        # 去除噪声强度
        self.strength = 0.5
        # 提示词
        self.p_prompt = [""]
        self.n_prompt = [""]
        self.data = [""]

    # 输入图片大小缩放
    @classmethod
    def input_image_resize(cls, img: Image) -> Image:
        # 获取图片大小
        width, height = img.size
        resize_image = img = pil_to_cv2(img)
        if width == height:
            # 方形非512x512缩放
            resize_image = cv.resize(img, (512, 512)) if width != 512 else img
        else:
            # 长方形图片的缩放与裁剪
            if width <= height:
                img = cv.resize(img, (512, math.ceil(height*512/width)))
                middle = int(height * 512 / width / 2)
                resize_image = img[middle-256:middle+256, 0:512]
            else:
                img = cv.resize(img, (math.ceil(width*512/height), 512))
                middle = int(width*512/height/2)
                resize_image = img[0:512, middle-256:middle+256]
        return cv2_to_pil(resize_image)

    # 返回模型路径
    @classmethod
    def get_model(cls, folder: str, file: str) -> str:
        return f"{folder}\\{file}"

    # 将句子划分为以prompt为单位的list
    @classmethod
    def sec_to_list(cls, sec: str) -> list:
        if sec == "":
            return [""]
        return sec.split(';')

    # 图像 RGB 均值化
    @classmethod
    def sd_equalize_hist(cls, image: Image) -> Image:
        """没有独立 VAE 的良好方法，可以不让图像发灰"""
        img = pil_to_cv2(image)
        (b, g, r) = cv.split(img)
        equal_b = cv.equalizeHist(b)
        equal_g = cv.equalizeHist(g)
        equal_r = cv.equalizeHist(r)
        dst = cv.merge((equal_b, equal_g, equal_r))
        return cv2_to_pil(dst)

    # 解析输入tag，确定其权重
    @classmethod
    def get_prompt_and_weight(cls, prompt: str) -> dict:
        re = {
            'prompt': prompt,
            'weight': 1,
        }
        if prompt.startswith(' '):
            prompt = prompt[1::]
        if (':' in prompt) and not prompt.endswith(':'):
            ls = prompt.replace('(', '').replace(')', '').split(':')
            re['prompt'] = ls[0]
            re['weight'] = float(ls[1]) if is_float(ls[1]) else 1
        elif ('(' in prompt) and (')' in prompt):
            left, right = [prompt.count('('), prompt.count(')')]
            re['prompt'] = prompt.replace('(', '').replace(')', '')
            re['weight'] = math.pow(1.1, left) if left == right else 1
        print(f'ENCODING: prompt = {re["prompt"]} and weight = {re["weight"]}')
        re['prompt'] = re['prompt'].replace('\n', '')
        if re['prompt'][0] == ',':
            re['prompt'] = re['prompt'][1::]
        return re

    # 返回预览图，以方图的形式
    @classmethod
    def get_preview(cls, imgs: list) -> Image:

        # 获取基本数据，创建画布
        imgs_num = len(imgs)
        if imgs_num == 1:
            return imgs[0]
        elif imgs_num == 0:
            return None
        raws = math.ceil(math.sqrt(imgs_num))
        long = 512 * raws + 16 * (raws - 1)
        img = Image.new("RGB", (long, long), "white")
        print(f'image size: {img.size}')

        # 绘制
        i = 0
        for y in range(1, raws + 1):
            for x in range(1, raws + 1):
                begin_x = (x - 1)*(512 + 16)
                begin_y = (y - 1)*(512 + 16)
                end_x = begin_x + 512
                end_y = begin_y + 512
                square_area = (begin_x, begin_y, end_x, end_y)
                img.paste(imgs[i], square_area)
                print(f'begin to paint:{x}/{y}: ({begin_x}, {begin_y}, {end_x}, {end_y})')
                # canvas.rectangle(square_area, fill=imgs[i], outline=None)
                if i == imgs_num - 1:
                    if raws * raws - imgs_num >= raws and imgs_num != 1:
                        img = img.crop((0, 0, long, long-(512+16)))
                    if long > 2096:
                        img = img.resize((2096, 2096))
                    return img
                i = i + 1

    # 刷新 data 数据
    def flash_data_isi(self, glo: list, loc: list) -> str:
        return ''.join(glo) + ',' + ''.join(loc)

    def flash_data_sd(self) -> None:
        self.data = self.batch_size * [self.p_prompt]

    # 更换模型权重
    def sd_change_model(self, folder: str, file: str) -> None:
        self.model = load_model_from_config(self.config, StableDiffusion.get_model(folder, file))
        print(StableDiffusion.get_model(folder, file))
        self.model = self.model.to(self.device)
        self.sampler = DDIMSampler(self.model)

    # t2i 流程
    def sd_t2i_process(self,
                       seed: int,
                       batch_size=1,
                       scale=7,
                       ddim_steps=100,
                       p_p="",
                       n_p="",
                       ) -> list:
        # 设置必要参数
        self.batch_size = batch_size
        self.scale = scale
        self.ddim_steps = ddim_steps
        self.p_prompt = p_p
        self.n_prompt = n_p
        self.flash_data_sd()
        re = []

        # 设置随机数种子
        seed_everything(seed)

        # 一些杂事
        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(self.data, desc="data", colour="green"):
                            print(f'This step: {prompts}')
                            uc = None
                            # cfg scale
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(self.batch_size * self.n_prompt)
                            # 元组 转 列表
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            # c 就是embedding
                            c = self.model.get_learned_conditioning(prompts)
                            print("embedding shape:", c.shape)

                            # 采样数据 shape (4, 512/8, 512/8) => (4, 64, 64)
                            shape = [self.C, self.H // self.f, self.W // self.f]
                            # 采样
                            samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                             conditioning=c,
                                                             batch_size=self.n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=uc,
                                                             eta=self.ddim_eta,
                                                             x_T=self.start_code)
                            print("samples shape:", samples_ddim.shape)
                            # VAE 解码器 输出最终图像
                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            # 把图像 值域 缩放到 0-1之间
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().numpy()

                            print("image shape:", x_samples_ddim.shape)
                            # 图像 值域从0-1 缩放到 0-255
                            x_sample = 255. * x_samples_ddim

                            img1 = np.stack([x_sample[0][0, :, :], x_sample[0][1, :, :], x_sample[0][2, :, :]], axis=2)
                            re.append(Image.fromarray(img1.astype(np.uint8)))
                    return re

    # sd 的 i2i 流程
    def sd_i2i_process(self,
                       img: Image,
                       seed: int,
                       scale=7,
                       ddim_steps=100,
                       strength=0.5,
                       batch_size=1,
                       p_p="",
                       n_p=""
                       ) -> list:
        # 设置随机数种子
        seed_everything(seed)

        re_image = []

        # 参数设置
        self.batch_size = batch_size
        self.scale = scale
        self.ddim_steps = ddim_steps
        self.strength = strength

        # 提示词
        self.p_prompt = p_p
        self.n_prompt = n_p
        self.data = [batch_size * [p_p]]
        print(f'data === {self.data}')

        # 加载 像素空间 图片
        init_image = load_img(img).to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=self.batch_size)

        # 像素空间 转换到 潜在空间
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        # 初始化采样器
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

        # 取噪强度 * 采样步数
        assert 0. <= self.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(self.strength * self.ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(self.data, desc="data", colour="green"):
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(self.batch_size * [self.n_prompt])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)

                            # encode (scaled latent) Aplha混合 潜在空间图像 和 噪声
                            z_enc = self.sampler.stochastic_encode(init_latent,
                                                              torch.tensor([t_enc] * self.batch_size).to(self.device))
                            # decode it 采样器采样
                            samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=self.scale,
                                                     unconditional_conditioning=uc, )

                            # VAE 解码器 将潜在空间转化为像素空间
                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples = x_samples.cpu().numpy()

                            print("image shape:", x_samples.shape)
                            # 图像 值域从0-1 缩放到 0-255
                            x_sample = 255. * x_samples

                            # 保存成PNG文件
                            img1 = np.stack([x_sample[0][0, :, :], x_sample[0][1, :, :], x_sample[0][2, :, :]], axis=2)
                            img = Image.fromarray(img1.astype(np.uint8))
                            re_image.append(img)
        return re_image

    # ISI 的 i2i 流程
    def isi_i2i_process(self,
                        img: Image,
                        seed: int,
                        scale=7,
                        ddim_steps=100,
                        strength=0.5,
                        p_p="",
                        n_p="") -> list:
        # 设置必要参数
        self.batch_size = 1
        self.scale = scale
        self.ddim_steps = ddim_steps
        self.strength = strength
        self.p_prompt = p_p
        self.n_prompt = n_p

        global_prompt = []
        local_prompt = []
        re_image = []

        if '}' in self.p_prompt:
            global_prompt.append(
                self.p_prompt.replace('{', '').split('}')[0]
            )
            local_prompt = StableDiffusion.sec_to_list(
                self.p_prompt.replace('{', '').split('}')[1]
            )
        else:
            local_prompt = StableDiffusion.sec_to_list(
                self.p_prompt
            )

        # 设置随机数种子
        seed_everything(seed)
        # 加载 像素空间 图片
        init_image = load_img(img).to(self.device)

        # 像素空间 转换到 潜在空间
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        # 初始化采样器
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

        # 取噪强度 * 采样步数
        assert 0. <= self.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(self.strength * self.ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(local_prompt, desc="data", colour="green"):
                            prompts = self.flash_data_isi(global_prompt, prompts)
                            p, w = [StableDiffusion.get_prompt_and_weight(prompts)['prompt'],
                                    StableDiffusion.get_prompt_and_weight(prompts)['weight']]
                            print(f'prompt = {p}, weight = {self.scale*w}')
                            uc = None
                            if self.scale != 1.0:
                                uc = self.model.get_learned_conditioning(self.batch_size * self.n_prompt)
                            if isinstance(p, tuple):
                                prompts = list(p)
                            c = self.model.get_learned_conditioning(prompts)

                            # encode (scaled latent) Aplha混合 潜在空间图像 和 噪声
                            z_enc = self.sampler.stochastic_encode(init_latent,
                                                              torch.tensor([t_enc] * self.batch_size).to(self.device))
                            # decode it 采样器采样
                            samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=self.scale*w,
                                                     unconditional_conditioning=uc, )

                            # VAE 解码器 将潜在空间转化为像素空间
                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples = x_samples.cpu().numpy()

                            print("image shape:", x_samples.shape)
                            # 图像 值域从0-1 缩放到 0-255
                            x_sample = 255. * x_samples

                            # 保存成PNG文件
                            img1 = numpy.stack([x_sample[0][0, :, :], x_sample[0][1, :, :], x_sample[0][2, :, :]], axis=2)
                            img = Image.fromarray(img1.astype(numpy.uint8))
                            re_image.append(img)

                            init_image = load_img(img).to(self.device)
                            # 像素空间 转换到 潜在空间
                            init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))
                    return re_image









