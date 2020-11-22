import paddle
from paddle import nn
import psp_encoders_pp as psp_encoders
from model_pp import Generator


class pSp(nn.Layer):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.set_opts(opts)
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(1024, 512, 8)
        self.face_pool = paddle.nn.AdaptiveAvgPool2D((256, 256))
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                self.register_buffer('latent_avg', paddle.zeros([1, self.decoder.style_dim]))
            else:
                self.register_buffer('latent_avg', paddle.zeros([1, 18, self.decoder.style_dim]))

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w:
                    codes = codes + self.latent_avg.tile([codes.shape[0], 1])
                else:
                    codes = codes + self.latent_avg.tile([codes.shape[0], 1, 1])


        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts