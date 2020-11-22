import argparse
import os
import gc
import sys
import pickle
import math

import paddle
import numpy as np

from psp_pp import pSp


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Pixel2Pixel paddle model generate"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default='doc/pSp_input_img.jpg',
        help="input image for convert",
    )
    parser.add_argument("path", metavar="PATH", help="path to the paddle weights")

    args = parser.parse_args()
    
    state_dict = paddle.load(args.path)

    opts = state_dict.pop('opts')
    opts = AttrDict(opts)
    if opts.input_nc != 3:
        args.input_image = ''

    net = pSp(opts)
    net.set_state_dict(state_dict)
    net.eval()

    if args.input_image != '':
        from PIL import Image
        img = Image.open(args.input_image).resize((256,256), Image.BILINEAR).convert('RGB')
        img = np.asarray(img) / 255.0 * 2 - 1
        with paddle.no_grad():
            img = paddle.to_tensor(img.transpose([2,0,1])[None,...].astype('float32'))
            img_rev = net(img)[0]
        img = np.uint8((img_rev.numpy().transpose([1,2,0]).clip(-1,1) + 1) / 2 * 255)
        path = args.input_image
        suffix = path.split('.')[-1]
        path = path.replace(f'.{suffix}', f'.rev.{suffix}')
        Image.fromarray(img).save(path)
        print(f'Save converted image to {path}')