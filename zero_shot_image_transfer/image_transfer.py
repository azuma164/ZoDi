import os
import random

import cv2
import einops
import numpy as np
import torch
from einops import repeat
from PIL import Image
from pytorch_lightning import seed_everything

import config
from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *

preprocessor = None

model_name = "control_v11p_sd15_seg"
model = create_model(f"./models/{model_name}.yaml").cpu()
model.load_state_dict(
    load_state_dict("./models/v1-5-pruned.ckpt", location="cuda"), strict=False
)
model.load_state_dict(
    load_state_dict(f"./models/{model_name}.pth", location="cuda"), strict=False
)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(
    det,
    input_image,
    seg_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    detect_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
):
    global preprocessor

    with torch.no_grad():
        input_image = HWC3(input_image)

        detected_map = resize_image(seg_image, image_resolution)
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
                # model.get_learned_conditioning([prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
           # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )

        batch_size = 1
        device = "cuda"

        init_image = img.astype(np.float32) / 255.0
        init_image = init_image[None].transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image).to(device)
        init_image = 2.0 * init_image - 1.0
        init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(init_image)
        )  # move to latent space

        # stochastic inversion
        strength_enc = 0.60
        t_enc = int(strength_enc * 1000)
        x_noisy = model.q_sample(
            x_start=init_latent,
            t=torch.tensor([t_enc] * batch_size).to(device),
        )
        model_output = model.apply_model(
            x_noisy, torch.tensor([t_enc] * batch_size).to(device), cond
        )

        ddim_sampler.make_schedule(
            ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False
        )
        z_enc = ddim_sampler.stochastic_encode(
            init_latent,
            torch.tensor([t_enc] * batch_size).to(device),
            noise=model_output,
            use_original_steps=True,
        )

        t_enc = int(strength_enc * ddim_steps)

        # decode it
        samples = ddim_sampler.decode(
            z_enc,
            cond,
            t_enc,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation training and evaluation')
    parser.add_argument('--cs_dir', type=str, required=True)
    parser.add_argument('--domain', required=True)

    args = parser.parse_args()

    prompt_dict = {
        "night": "driving at night",
        "snow": "driving in snow",
        "rain": "driving under rain",
        "fog": "driving in fog",
        "game": "driving in a game"
    }

    det = "Seg_OFADE20K"
    prompt = prompt_dict[args.domain]
    a_prompt = "best quality"
    n_prompt = "lower quality, worst quality"
    num_samples = 1
    image_resolution = 512
    detect_resolution = 512
    ddim_steps = 20
    guess_mode = False
    strength = 1.0
    scale = 9.0
    seed = 42
    eta = 1.0


    img_dir = f"{args.cs_dir}/leftImg8bit/train/"
    seg_dir = f"{args.cs_dir}/gtFine/train/"
    save_dir = f"{args.cs_dir}/{prompt}/train/"

    for city in os.listdir(img_dir):
        city_dir = os.path.join(img_dir, city)
        save_city_dir = os.path.join(save_dir, city)
        if not os.path.exists(save_city_dir):
            os.makedirs(save_city_dir)
        for img in os.listdir(city_dir):
            img_path = os.path.join(city_dir, img)
            seg_name = "_".join(img.split("_")[:3]) + "_gtFine_color.png"
            seg_path = os.path.join(seg_dir, city, seg_name)

            save_name = os.path.join(save_city_dir, img)
            if os.path.exists(save_name):
                continue

            input_image = Image.open(img_path)
            input_image = np.asarray(input_image)
            seg_image = Image.open(seg_path)
            seg_image = np.asarray(seg_image)

            ips = [
                det,
                input_image,
                seg_image,
                prompt,
                a_prompt,
                n_prompt,
                num_samples,
                image_resolution,
                detect_resolution,
                ddim_steps,
                guess_mode,
                strength,
                scale,
                seed,
                eta,
            ]
            seg_image, gen_image = process(*ips)
            seg_image, gen_image = Image.fromarray(seg_image), Image.fromarray(gen_image)

            gen_image.save(os.path.join(save_city_dir, img))
