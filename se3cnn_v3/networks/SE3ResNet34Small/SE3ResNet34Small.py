import torch.nn as nn

from functools import partial

from se3cnn_v3.util.arch_blocks import *
from se3cnn.image import kernel


class network(ResNet):
    def __init__(self,
                 n_input,
                 n_output,
                 latent_size,
                 args):

        features = [[[(2,  2,  2,  2)]],          #  32 channels
                    [[(2,  2,  2,  2)] * 2] * 3,  #  32 channels
                    [[(4,  4,  4,  4)] * 2] * 4,  #  64 channels
                    [[(8,  8,  8,  8)] * 2] * 6,  # 128 channels
                    # [[(8, 8, 8, 8)] * 2] * 2 + [[(8, 8, 8, 8), (128, 0, 0, 0)]]]  # 256 channels
                    [[(16, 16, 16, 16)] * 2] * 2 + [[(16, 16, 16, 16), (256, 0, 0, 0)]]]  # 256 channels
        common_params = {
            'radial_window': partial(kernel.gaussian_window_wrapper,
                                     mode=args.bandlimit_mode, border_dist=0, sigma=0.6),
            'batch_norm_momentum': 0.01,
            # TODO: probability needs to be adapted to capsule order
            'capsule_dropout_p': args.p_drop_conv,  # drop probability of whole capsules
            'normalization': args.normalization,
            'downsample_by_pooling': args.downsample_by_pooling,
        }
        if args.SE3_nonlinearity == 'gated':
            res_block = SE3GatedResBlock
        else:
            res_block = SE3NormResBlock
        global OuterBlock
        OuterBlock = partial(OuterBlock,
                             res_block=partial(res_block, **common_params))
        super().__init__(
            OuterBlock((n_input,),          features[0], size=7),
            OuterBlock(features[0][-1][-1], features[1], size=args.kernel_size, stride=1),
            OuterBlock(features[1][-1][-1], features[2], size=args.kernel_size, stride=2),
            OuterBlock(features[2][-1][-1], features[3], size=args.kernel_size, stride=2),
            OuterBlock(features[3][-1][-1], features[4], size=args.kernel_size, stride=2),
            AvgSpacial(),
            nn.Dropout(p=args.p_drop_fully,
                       inplace=True) if args.p_drop_fully is not None else None,
            VAE(features[4][-1][-1][0], latent_size, n_output)
        )