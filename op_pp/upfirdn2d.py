import paddle
from paddle.nn import functional as F
 
 
def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )
 
    return out
 
 
def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape((-1, in_h, in_w, 1))
 
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
 
    out = input.reshape((-1, in_h, 1, in_w, 1, minor))
    out = out.transpose((0,1,3,5,2,4))
    out = out.reshape((-1,1,1,1))
    out = F.pad(out, [0, up_x - 1, 0, up_y - 1])
    out = out.reshape((-1, in_h, in_w, minor, up_y, up_x))
    out = out.transpose((0,3,1,4,2,5))
    out = out.reshape((-1, minor, in_h * up_y, in_w * up_x))
 
    out = F.pad(
        out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,:,
        max(-pad_y0, 0) : out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[3] - max(-pad_x1, 0),
    ]
 
    out = out.reshape((
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    ))
    w = paddle.flip(kernel, [0, 1]).reshape((1, 1, kernel_h, kernel_w))
    out = F.conv2d(out, w)
    out = out.reshape((
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    ))
    out = out.transpose((0, 2, 3, 1))
    out = out[:, ::down_y, ::down_x, :]
 
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
 
    return out.reshape((-1, channel, out_h, out_w))