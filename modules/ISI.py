"""make variations of input image"""
import os
import sys
import torch
import numpy
import math
import PIL
from modules.utils import *
import cv2 as cv
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tqdm import tqdm, trange
from itertools import islice
from einops import repeat
from torch import autocast
from pytorch_lightning import seed_everything

sys.path.append('stable_diffusion')
from stable_diffusion.ldm.util import instantiate_from_config
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler


# stable diffusion 加载流程
# https://github.com/CompVis/stable-diffusion/blob/main/scripts/img2img.py
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
    def __init__(self, config_path: str, model_path: str,
                 is_init_model=True, h=512, w=512):
        """ 以下为不可变参数 """
        self.C = 4
        # 高
        self.H = h
        # 宽
        self.W = w
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
        # 设备
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # 从加载的配置文件创建模型
        self.model = load_model_from_config(self.config, model_path) if is_init_model else None
        # 将模型送到CUDA设备 或者 CPU
        self.model = self.model.to(self.device) if is_init_model else None
        # 创建ddim 采样器
        self.sampler = DDIMSampler(self.model) if is_init_model else None

    # 输入图片大小缩放
    @classmethod
    def input_image_resize(cls, img: Image, w=512, h=512) -> Image:
        # 获取图片大小
        width, height = img.size
        resize_image = img = pil_to_cv2(img)
        if w / h == width / height:
            rate = w / width
            resize_image = cv.resize(img, (w*rate, h*rate)) if width != h else img
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
        return os.path.join(folder, file)

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
        print('Start Equalize')
        img = pil_to_cv2(image)
        (b, g, r) = cv.split(img)
        equal_b = cv.equalizeHist(b)
        equal_g = cv.equalizeHist(g)
        equal_r = cv.equalizeHist(r)
        dst = cv.merge((equal_b, equal_g, equal_r))
        print('Equalize success')
        return cv2_to_pil(dst)

    # 解析输入tag，确定其权重
    @classmethod
    def get_prompt_and_weight(cls, prompt: str) -> dict:
        re = {
            'prompt': prompt,
            'weight': 1,
        }
        if prompt.count('(') != prompt.count(')'):
            re['prompt'] = re['prompt'].replace('(', '').replace(')', '').replace(':', '')
            return re
        if prompt.startswith(' '):
            prompt = prompt[1::]
        if (':' in prompt) and not prompt.endswith(':') and prompt.endswith(')') and ',' not in prompt.split(':')[1]:
            index = prompt.rfind(":")
            re['prompt'] = prompt[:index][1:]
            re['weight'] = float(prompt[index+1:].replace(')', '')) \
                if is_float(prompt[index+1:].replace(')', '')) else 1
        elif prompt.startswith('(') and prompt.endswith(')'):
            count = 0
            for char in reversed(prompt):
                if char == ")":
                    count = count + 1
                else:
                    break
            re['prompt'] = prompt[count: -count]
            re['weight'] = math.pow(1.1, count)
        print(f'ENCODING: prompt = {re["prompt"]} and weight = {re["weight"]}')
        re['prompt'] = re['prompt'].replace('\n', '')
        re['weight'] = round(re['weight'], 5)
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
        data = batch_size * [p_p]
        re_image = []

        # 设置随机数种子
        seed_everything(seed)

        # 一些杂事
        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data", colour="red"):
                            print(f'This step: {prompts}')
                            uc = None
                            # cfg scale
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * n_p)
                            # 元组 转 列表
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            # c 就是embedding
                            c = self.model.get_learned_conditioning(prompts)
                            print("embedding shape:", c.shape)

                            # 采样数据 shape (4, 512/8, 512/8) => (4, 64, 64)
                            shape = [self.C, self.H // self.f, self.W // self.f]
                            # 采样
                            samples_ddim, _ = self.sampler.sample(S=ddim_steps,
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
                            re_image.append(Image.fromarray(img1.astype(np.uint8)))
                    return re_image

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
        data = [batch_size * [p_p]]

        # 加载 像素空间 图片
        init_image = load_img(img).to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

        # 像素空间 转换到 潜在空间
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        # 初始化采样器
        self.sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

        # 取噪强度 * 采样步数
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data", colour="green"):
                            print(f'This step: {prompts}')
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [n_p])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)

                            # encode (scaled latent) Aplha混合 潜在空间图像 和 噪声
                            z_enc = self.sampler.stochastic_encode(init_latent,
                                                              torch.tensor([t_enc] * batch_size).to(self.device))
                            # decode it 采样器采样
                            samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
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
        global_prompt = ''
        local_prompt = []
        re_image = []

        if '}' in p_p:
            global_prompt = p_p.replace('{', '').split('}')[0]
            local_prompt = StableDiffusion.sec_to_list(
                p_p.replace('{', '').split('}')[1]
            )
        else:
            local_prompt = StableDiffusion.sec_to_list(
                p_p
            )

        # 设置随机数种子
        seed_everything(seed)
        # 加载 像素空间 图片
        init_image = load_img(img).to(self.device)

        # 像素空间 转换到 潜在空间
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        # 初始化采样器
        self.sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

        # 取噪强度 * 采样步数
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(local_prompt, desc="data", colour="blue"):
                            p, w = [StableDiffusion.get_prompt_and_weight(prompts)['prompt'],
                                    StableDiffusion.get_prompt_and_weight(prompts)['weight']]
                            prompts = global_prompt + ',' + str(p)
                            print(f'prompt = {prompts}, weight = {scale*w}')
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(n_p)
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)

                            # encode (scaled latent) Aplha混合 潜在空间图像 和 噪声
                            z_enc = self.sampler.stochastic_encode(init_latent,
                                                              torch.tensor([t_enc]).to(self.device))
                            # decode it 采样器采样
                            samples = self.sampler.decode(z_enc, c, t_enc,
                                                          unconditional_guidance_scale=scale*w,
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

