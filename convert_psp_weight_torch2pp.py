import argparse
import os
import gc
import sys
import pickle
import math

import torch
import paddle
import numpy as np

from psp_pp import pSp


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt
    
@paddle.no_grad()
def torch2pp(torch_state_dict, pp_state_dict):
    for param_name, param in torch_state_dict.items():
        be_found = False
        if '.num_batches_tracked' in param_name:
            continue
        for _param_name, _param in pp_state_dict.items():
            if param_name == _param_name:
                if len(param.shape) != 2:
                    _param.set_value(paddle.to_tensor(param.cpu().numpy()))
                else:
                    _param.set_value(paddle.to_tensor(param.cpu().numpy()).transpose((1,0)))
                be_found = True
                print(f'Convert from {param_name}({list(param.shape)}) to {_param_name}({list(_param.shape)})')
                break
            if ('.weight' in param_name and param_name.split('.weight')[0] == _param_name.split('._weight')[0]) or \
              ('.bias' in param_name and param_name.split('.bias')[0] == _param_name.split('._bias')[0]) or \
              ('.running_mean' in param_name and param_name.split('.running_mean')[0] == _param_name.split('._mean')[0]) or \
              ('.running_var' in param_name and param_name.split('.running_var')[0] == _param_name.split('._variance')[0]):
                if len(param.shape) != 2:
                    _param.set_value(paddle.to_tensor(param.cpu().numpy()))
                else:
                    _param.set_value(paddle.to_tensor(param.cpu().numpy()).transpose((1,0)))
                be_found = True
                print(f'Convert from {param_name}({list(param.shape)}) to {_param_name}({list(_param.shape)})')
                break
        assert be_found, f'Parameter `{param_name}` needed.`'


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Pixel2Style2Pixel PyTorch to paddle model checkpoint converter"
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
    parser.add_argument("path", metavar="PATH", help="path to the pytorch weights")

    args = parser.parse_args()
    
    ckpt = torch.load(args.path, map_location='cpu')

    name = os.path.splitext(os.path.basename(args.path))[0]

    opts = ckpt['opts']
    opts['checkpoint_path'] = name + '.pdparams'
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    opts = AttrDict(opts)
    if opts.input_nc != 3:
        args.input_image = ''

    net = pSp(opts)
    if 'latent_avg' in ckpt:
        if opts.learn_in_w:
            net.latent_avg[:,:] = paddle.to_tensor(ckpt['latent_avg'].cpu().numpy())
        else:
            net.latent_avg[0,:,:] = paddle.to_tensor(ckpt['latent_avg'].cpu().numpy())
    
    pp_ckpt = net.state_dict()
    torch2pp(get_keys(ckpt, 'encoder'), get_keys(pp_ckpt, 'encoder'))
    torch2pp(get_keys(ckpt, 'decoder'), get_keys(pp_ckpt, 'decoder'))
    net.set_state_dict(pp_ckpt)
    del pp_ckpt

    net.eval()
    pp_ckpt = net.state_dict()
    pp_ckpt['opts'] = dict(opts)

    del ckpt
    gc.collect()
    
    print('Saving...')
    paddle.save(pp_ckpt, name + '.pdparams')
    print('Saved.')

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
        print(f'Save convert image to {path}')