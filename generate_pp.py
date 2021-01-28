import argparse

import paddle
import numpy as np
from model_pp import Generator
from tqdm import tqdm
from PIL import Image


def generate(args, g_ema, mean_latent):

    with paddle.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = paddle.randn((args.sample, args.latent))

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            
            sample = np.uint8((sample*0.5+0.5).clip(0,1).numpy() * 255)
            sample = [t for t in sample]
            sample = np.concatenate(sample, 1)
            sample = sample.transpose((1,2,0))
            sample = Image.fromarray(sample)
            sample.save(f"sample/{str(i).zfill(6)}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=512, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="network-tadne.g_ema",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 1024
    args.n_mlp = 4

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    )
    state_dict = paddle.load(args.ckpt)

    g_ema.set_state_dict(state_dict)

    if args.truncation < 1:
        with paddle.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, mean_latent)
