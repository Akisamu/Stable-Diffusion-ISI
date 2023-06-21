import os, sys
import ffmpeg
import numpy
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt

sys.path.append('../../stable_diffusion')
from modules.ISI import StableDiffusion
from modules.utils import *
from modules.super_resolution.image_super_resolution import upscale, Para

# 超分模型路径改变，，，python环境变量就是一坨
Para.set_model_path('../../models/iss/best.pth')

# 打开视频文件
cap = cv2.VideoCapture('bbl.mp4')
# 获取视频分辨率
width, ow = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))] * 2
height, oh = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))] * 2
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'fps = {fps}')
print('视频分辨率: {} x {}'.format(width, height))
cap.release()

limit_time = 10
width_remainder = width % 64
height_remainder = height % 64

if width > 640 or height > 640:
    if width > height:
        # more_time = width_times - 10
        # (width - width_remainder) / 64 * (10 - more_time) / 10
        width = 640
        height = (height - height_remainder) * 640 / (width - width_remainder)
    elif height > width:
        width = (width - width_remainder) * 640 / (height - height_remainder)
        height = 640
else:
    width = width - width_remainder
    height = height - height_remainder

width = int(width)
height = int(height)

print('修改后视频分辨率: {} x {}'.format(width, height))
print('super_r后视频分辨率: {} x {}'.format(width * 2, height * 2))

sd = StableDiffusion(w=width, h=height,
                     config_path='../../stable_diffusion/configs/stable-diffusion/v1-inference.yaml',
                     model_path='../../models/stable diffusion/pastelmix.ckpt'
                     )


# sd.sd_change_model('../../models/stable diffusion/', 'pastelmix.ckpt')


def sd_process(img: Image, sc, st):
    sd_img = sd.sd_i2i_process(
        img=img.resize((width, height)),
        seed=42,
        scale=sc,
        ddim_steps=100,
        strength=st,
        batch_size=1,
        p_p='animation, masterpieces,detail face,1girl,yellow hair, blue eyes,white sailor uniform with blue collar',
        n_p='watermark,longbody,lowres,bad anatomy,bad hands,missing fingers,pubic hair,extra digit,fewer digits,cropped,worst quality,low quality'
    )[0]
    equalize_img = sd.sd_equalize_hist(sd_img)
    up = upscale(2, img=sd_img)
    # img.show()
    re = numpy.array(
        up.resize((1280, 920))
    )
    torch.cuda.empty_cache()
    plt.imshow(re)
    plt.show()
    print(re.shape)
    return re


if __name__ == "__main__":
    i = 1
    scale = range(5, 9, 1)
    denoise = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    s = 8
    d = 0.3
    process1 = (
        ffmpeg
        .input(filename='.\\bbl.mp4')
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(1280, 920))
        .output(f'babala-remake.mp4', pix_fmt='yuv444p', r=fps, video_size=f'1280x920')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    print(f'this cycle: scale={s}, strength={d}')
    while True:
        in_bytes = process1.stdout.read(ow * oh * 3)
        if not in_bytes:
            break
        print(f'正在处理第 {i} 帧数, 已完成 {i / frame_count * 100}%')
        frame = np.array(
            sd_process(
                Image.fromarray(
                    np.frombuffer(in_bytes, np.uint8).reshape([oh, ow, 3])
                ), sc=s, st=d
            )
        )
        process2.stdin.write(
            # out_frame
            frame
            .astype(np.uint8)
            .tobytes()
        )
        i += 1
    i = 1

    process2.stdin.close()
    process1.wait()
    process2.wait()
