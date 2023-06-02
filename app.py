import gradio as gr
from img2img import StableDiffusion
from image_super_resolution import upscale
from PIL import Image
from utils import *

# 加载webui的配置
config = read_json('config.json')
server_config = config['server_config']
webui_config = config['webui_config']
sd_config = config['stable_diffusion_config']


# 初始化加载模型名称列表
MODEL_DIR = check_path(sd_config['model_folder'])
model_name_list = [item for item in get_filenames(MODEL_DIR) if item.endswith(".ckpt")]
model_name = sd_config['default_model']

# 获取sd流程的io路径
io_configs = config["io_config"]
cut = check_path(io_configs["cut"])
sd = check_path(io_configs["sd"])
equalize = check_path(io_configs["equalize"])
pre = check_path(io_configs["pre"])
fin = check_path(io_configs["fin"])
his = check_path(io_configs["history"])


# 建立 Stable Diffusion 对象
sd_instance = StableDiffusion(
    config_path=sd_config['model_config_path'],
    model_path=os.path.join(MODEL_DIR, model_name),
    is_init_model=sd_config["is_init_model"]
)


# gradio 的 callback 函数：切换模型
def change_model(model: str) -> str:
    global model_name
    if model == model_name:
        return f'已经是 {model_name}'
    if model in model_name_list:
        sd_instance.sd_change_model(MODEL_DIR, model)
        model_name = model
        return f'已重新加载model：{model}'
    return f'该模组不在 {MODEL_DIR} 目录下'


# gradio 的 callback 函数：整个图像处理逻辑流程
def stable_diffusion_logic(img,
                           p_p: str,
                           n_p: str,
                           bs: int,
                           scale: float,
                           steps: int,
                           strength: float,
                           seed: str,
                           up: int,
                           eq: bool
                           ) -> list:
    # 获取随机哈希码，作为图像名字
    png_hash = get_hash(24)

    seed = int(seed) if is_int(seed) else random.randint(1, 2147483647)
    is_i2i = False if img is None else True
    re_image = None
    is_isi = True if ';' in p_p else False
    p_p = p_p.replace('\n', '')

    # 裁剪图片
    resize_image = StableDiffusion.input_image_resize(img) if img is not None else None
    if resize_image is not None:
        resize_image.save(Op.join(cut, f"{png_hash}.png"))

    # 扩散模型处理图片
    sd_image = []
    if is_isi and is_i2i:
        bs = 1
        sd_image = sd_instance.isi_i2i_process(
            img=resize_image,
            seed=seed,
            scale=scale,
            ddim_steps=steps,
            strength=strength,
            p_p=p_p,
            n_p=n_p,
        )
    elif is_i2i:
        sd_image = sd_instance.sd_i2i_process(
            img=resize_image,
            seed=seed,
            scale=scale,
            ddim_steps=steps,
            strength=strength,
            batch_size=bs,
            p_p=p_p,
            n_p=n_p,
        )
    else:
        sd_image = sd_instance.sd_t2i_process(
            seed=seed,
            batch_size=bs,
            scale=scale,
            ddim_steps=steps,
            p_p=p_p,
            n_p=n_p,
        )
    if len(sd_image) == 1:
        sd_image[0].save(Op.join(sd, f"{png_hash}.png"))
    else:
        create_dir(sd, png_hash)
        i = 1
        for item in sd_image:
            item.save(Op.join(sd, png_hash, f'batch={i}.png'))
            i = i + 1

    # 图像均值化
    equalize_image = sd_image
    equalize_image_cache = []
    if eq:
        if len(equalize_image) == 1:
            equalize_image[0] = StableDiffusion.sd_equalize_hist(equalize_image[0])
            equalize_image[0].save(Op.join(equalize, f"{png_hash}.png"))
        else:
            create_dir(equalize, png_hash)
            i = 1
            for item in sd_image:
                item = StableDiffusion.sd_equalize_hist(item)
                item.save(Op.join(f'{equalize}', f'{png_hash}', f'batch={i}.png'))
                equalize_image_cache.append(item)
                i = i + 1
            equalize_image = equalize_image_cache
        # equalize_image = StableDiffusion.sd_equalize_hist(sd_image)
        # equalize_image.save(put_png_path('outputs\\equalize_cache', png_hash))

    # SRCNN 图像放大
    upscale_image = equalize_image
    upscale_image_cache = []
    if len(upscale_image) == 1:
        re_image = upscale_image[0]
        if up == 1:
            upscale_image[0].save(Op.join(fin, f"{png_hash}.png"))
        else:
            upscale_image[0] = upscale(scale=up, img=upscale_image[0])
            upscale_image[0].save(Op.join(fin, f'[X{up}] {png_hash}.png'))
    else:
        re_image = StableDiffusion.get_preview(upscale_image)
        re_image.save(Op.join(pre, f'{png_hash}.png'))
        i = 1
        if up == 1:
            create_dir(fin, f'{png_hash}')
            for item in upscale_image:
                item.save(Op.join(f'{fin}', f'{png_hash}', f'batch={i}.png'))
                i = i + 1
        else:
            create_dir(fin, f'[X{up}] {png_hash}')
            for item in upscale_image:
                item = upscale(scale=up, img=item)
                item.save(Op.join(f'{fin}', f'[X{up}] {png_hash}', f'batch={i}.png'))
                upscale_image_cache.append(item)
                i = i + 1
            upscale_image = upscale_image_cache

    # 保存本次运行配置
    log = {
        "本次完成时间": get_time(),
        "图片哈希": png_hash,
        "模式": 'Image to Image' if is_i2i else 'Text to Image',
        "stable diffusion 参数": {
            "模型名称": model_name,
            "seed": seed,
            "批量": bs,
            "提示词相关性": scale,
            "步数": steps,
            "重绘幅度": strength if img is not None else '无',
            "正相关提示词": p_p,
            "反相关提示词": n_p,
        },
        "放大倍率": 'X' + str(up),
        "是否使用均值化": '是' if eq else '否'
    }
    # 保存该次配置到 history 文件夹中
    write_dict_to_json(Op.join(his, f'{png_hash}.json'), log)
    print("已将本次运行配置保存在 history 文件夹中")
    # 清除显存中缓存的张量
    torch.cuda.empty_cache()
    return [re_image, dict_to_str(log)]


# 建立 gradio 的 demo，在后续程序中 lunch()。
def create_ui():
    with gr.Blocks(css=webui_config['css'],
                   theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")) as demo:
        with gr.Row():
            # 左半部分 UI
            with gr.Column(scale=5):
                with gr.Row():
                    model = gr.Dropdown(model_name_list, label='Models', info="请在此处选择您的模型",
                                        value=model_name_list[0] if len(model_name_list) > 0 else "没有模型",
                                        interactive=True)
                    change_model_ins = gr.Button('确定更改模型', interactive=True)
                with gr.Column(variant="panel"):
                    with gr.Row():
                        p_prompt = gr.Textbox(label='正向提示词',
                                              placeholder='请输入正向提示词，每个提示词请用英文逗号分割。', lines=5,
                                              value=webui_config['default_p_prompt'], interactive=True)
                    with gr.Row():
                        n_prompt = gr.Textbox(label='反向提示词',
                                              placeholder='请输入反向提示词，每个提示词请用英文逗号分割。', lines=5,
                                              value=webui_config['default_n_prompt'], interactive=True)
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=5, variant="panel"):
                            with gr.Row():
                                bs = gr.Slider(label='批量', minimum=1, maximum=webui_config['max_batch_size'], step=1,
                                               value=1, interactive=True)
                                scale = gr.Slider(label='提示词相关性', minimum=1, maximum=webui_config['max_scale'],
                                                  step=0.1, value=7, interactive=True, elem_classes='gr_slider')
                            with gr.Row():
                                steps = gr.Slider(label='步数', minimum=10, maximum=100, step=1, value=80,
                                                  interactive=True)
                            with gr.Row():
                                strength = gr.Slider(label='重绘幅度', minimum=0, maximum=1, step=0.01, value=0.75,
                                                     interactive=True)
                                eq_cb = gr.Checkbox(label='均值化', value=False, interactive=True)
                            with gr.Row():
                                seed = gr.Textbox(label='SEED', value='随机', max_lines=1, interactive=True)
                        with gr.Column(scale=5):
                            with gr.Row():
                                intput_img = gr.Image(type='pil', label='输入图片')
                                # intput_img = gr.Image(type='filepath')
                                pass
                            with gr.Row():
                                up = gr.Slider(label='SRCNN 放大倍率', minimum=1, maximum=4, value=1, step=1,
                                               interactive=True)
                            with gr.Row():
                                submit_btn = gr.Button('点击生成', interactive=True, variant='primary')
            # 右半部分 UI
            with gr.Column(scale=5):
                with gr.Row():
                    gr.Markdown('''
                    <h1><center>Stable Diffusion ISI</center></h1>
                    <h5><center>及 t2i，i2i 一体的 ai 绘画框架。</center></h5>
                    ''')
                with gr.Row():
                    with gr.Accordion(label='说明', open=False):
                        gr.Markdown(webui_config['intro'])
                with gr.Row():
                    output_img = gr.Image(type='pil', label='输出图片')
                with gr.Row():
                    d_value = f'已加载模型：{sd_config["default_model"]}' if 'model.ckpt' in model_name_list else '无默认模型'

                    logs = gr.Textbox(label='模型状态',
                                      value=d_value if sd_config["is_init_model"] else '不加载默认模型，请切换模型',
                                      interactive=False, lines=1)
                with gr.Row():
                    with gr.Accordion(label='本次运行数据', open=False):
                        run_data = gr.Textbox(value='', interactive=False, lines=10)

            # ****************** 事件处理 ******************
            change_model_ins.click(change_model, inputs=[
                model
            ], outputs=[
                logs,
            ])
            submit_btn.click(stable_diffusion_logic, inputs=[
                intput_img,
                p_prompt,
                n_prompt,
                bs,
                scale,
                steps,
                strength,
                seed,
                up,
                eq_cb
            ], outputs=[
                output_img,
                run_data
            ])
    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.queue(1).launch(
        server_name="0.0.0.0" if server_config['IS_PUBLIC_IP'] else "127.0.0.1",
        server_port=server_config['PORT'],
        share=server_config['IS_SHARE']
    )
    # test()
    pass
