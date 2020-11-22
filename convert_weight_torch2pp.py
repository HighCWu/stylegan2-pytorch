import argparse
import os
import sys
import pickle
import math

import torch
import paddle
import numpy as np
from torchvision import utils

from model import Generator, Discriminator, EqualLinear
from model_pp import Generator as G_pp, Discriminator as D_pp

    
@paddle.no_grad()
def torch2pp(torch_model, pp_model):
    from model import EqualLinear
    torch_layers = { k: v for k, v in torch_model.named_modules() }
    pp_layers = { k: v for k, v in pp_model.named_sublayers()}
    pp_layers[''] = pp_model
    
    for layer_name in torch_layers.keys():
        torch_layer = torch_layers[layer_name]
        pp_layer = pp_layers[layer_name]
        if isinstance(torch_layer, EqualLinear):
            pp_layer.weight[:] = paddle.to_tensor(torch_layer.weight.detach().cpu().numpy()).transpose((1,0))
            if pp_layer.bias is not None:
                pp_layer.bias[:] = paddle.to_tensor(torch_layer.bias.detach().cpu().numpy())
        else:
            for param_name, param in torch_layer._parameters.items():
                pp_layer._parameters[param_name] = paddle.to_tensor(param.detach().cpu().numpy())


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="PyTorch to paddle model checkpoint converter"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="output image size of the generator"
    )
    parser.add_argument("path", metavar="PATH", help="path to the pytorch weights")

    args = parser.parse_args()
    
    torch_state_dicts = torch.load(args.path)

    size = args.size
    name = os.path.splitext(os.path.basename(args.path))[0]

    state_dict = torch_state_dicts['g_ema']
    g = g_ema = Generator(size, 512, 8, channel_multiplier=args.channel_multiplier)
    g.load_state_dict(state_dict)
    g_pp = g_ema_pp = G_pp(size, 512, 8, channel_multiplier=args.channel_multiplier)
    torch2pp(g, g_pp)
    paddle.save(g_pp.state_dict(), name + '.g_ema')
        
    latent_avg = torch_state_dicts['latent_avg']
    latent_avg_pp = paddle.to_tensor(latent_avg.detach().cpu().numpy())
    latent_avg_layer = paddle.nn.Layer()
    latent_avg_layer.register_buffer('latent_avg', latent_avg_pp)
    paddle.save(latent_avg_layer.state_dict(), name + '.latent_avg')
        
    if 'g' in torch_state_dicts:
        state_dict = torch_state_dicts['g']
        g = Generator(size, 512, 8, channel_multiplier=args.channel_multiplier)
        g.load_state_dict(state_dict)
        g_pp = G_pp(size, 512, 8, channel_multiplier=args.channel_multiplier)
        torch2pp(g, g_pp)
        paddle.save(g_pp.state_dict(), name + '.g')

    if 'd' in torch_state_dicts:
        state_dict = torch_state_dicts['d']
        d = Discriminator(size, channel_multiplier=args.channel_multiplier)
        d.load_state_dict(state_dict)
        d_pp = D_pp(size, channel_multiplier=args.channel_multiplier)
        torch2pp(d, d_pp)
        paddle.save(d_pp.state_dict(), name + '.d')

    batch_size = {256: 16, 512: 9, 1024: 4}
    n_sample = batch_size.get(size, 25)

    g = g_ema.to(device)

    z = np.random.RandomState(0).randn(n_sample, 512).astype("float32")

    with torch.no_grad():
        img_pt, _ = g(
            [torch.from_numpy(z).to(device)],
            truncation=0.5,
            truncation_latent=latent_avg.to(device),
            randomize_noise=False,
        )
        
    with paddle.no_grad():
        img_pp, _ = g_ema_pp(
            [paddle.to_tensor(z)],
            truncation=0.5,
            truncation_latent=latent_avg_pp,
            randomize_noise=False,
        )

    img_pp = torch.from_numpy(img_pp.numpy()).to(device)

    img_diff = ((img_pt + 1) / 2).clamp(0.0, 1.0) - ((img_pp.to(device) + 1) / 2).clamp(
        0.0, 1.0
    )

    img_concat = torch.cat((img_pp, img_pt, img_diff), dim=0)

    print(img_diff.abs().max())

    utils.save_image(
        img_concat, name + ".png", nrow=n_sample, normalize=True, range=(-1, 1)
    )
