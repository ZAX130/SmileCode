'''
VoxelMorph

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Haiqiao Wang
2110246069@email.szu.edu.cn
Shenzhen University
'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, factor, ndims):
        super().__init__()
        self.factor = factor
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.2):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        x = self.upconv(x)
        x = x[:, :, 1:-1, 1:-1, 1:-1]
        return self.actout(x)

class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvBlock(in_channel, c, 3, 2, 1)
        self.conv1 = ConvBlock(c, 2*c, 3, 2, 1)
        self.conv2 = ConvBlock(2*c, 4*c, 3, 2, 1)
        self.conv3 = ConvBlock(4*c, 8*c, 3, 2, 1)

    def forward(self, x):
        out0 = self.conv0(x)  # 1/2
        out1 = self.conv1(out0)  # 1/4
        out2 = self.conv2(out1)  # 1/8
        out3 = self.conv3(out2)  # 1/16

        return [out0, out1, out2, out3]


class Estimator(nn.Module):
    def __init__(self, channel, alpha=0.1):
        super(Estimator, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            nn.Conv3d(c, c, 3, 1, 1),
            nn.Conv3d(c, c, 3, 1, 1),
            nn.Conv3d(c, c, 3, 1, 1),
            nn.LeakyReLU(alpha),
            nn.Conv3d(c, 3, 3, 1, 1)
        )
        self.conv[-1].weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv[-1].weight.shape))
        self.conv[-1].bias = nn.Parameter(torch.zeros(self.conv[-1].bias.shape))

    def forward(self, fixed_fm, float_fm):
        x = torch.cat([fixed_fm, float_fm], dim=1)
        x = self.conv(x)
        return x


class RDN_diff_share(nn.Module):
    def __init__(self, inshape=(160, 192, 160), in_channel=1, channels=16,stage_recursion=1, level_recursion=[1,1,1,1]):
        super(RDN_diff_share, self).__init__()

        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.stages = stage_recursion
        self.levels = level_recursion
        # dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)  # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.est3 = Estimator(2 * 8 * c)
        self.est2 = Estimator(2 * 4 * c)
        self.est1 = Estimator(2 * 2 * c)
        self.est0 = Estimator(2 * c)

        self.transformer = nn.ModuleList()
        self.integrate = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s//2**i for s in inshape]))
            self.integrate.append(VecInt([s//2**i for s in inshape]))

    def forward(self, moving, fixed):
        # encode stage
        conv0_float, conv1_float, conv2_float, conv3_float = self.encoder(moving)
        conv0_fixed, conv1_fixed, conv2_fixed, conv3_fixed = self.encoder(fixed)
        # c, 2c, 4c, 8c  # 1/2, 1/4, 1/8, 1/16

        flow = 0
        # moves = []
        # sflows = []
        svs = []
        # stage wise recursion
        for i in range(self.stages):
            if i==0 :
                Fm0, Fm1, Fm2, Fm3 = conv0_float, conv1_float, conv2_float, conv3_float
            else :
                Fm3 = self.transformer[4](conv3_float, 0.125*nnf.interpolate(flow, scale_factor=0.125, mode='trilinear',
                                                                       align_corners=True))
                Fm2 = self.transformer[3](conv2_float, 0.25*nnf.interpolate(flow, scale_factor=0.25, mode='trilinear',
                                                                       align_corners=True))
                Fm1 = self.transformer[2](conv1_float, 0.5*nnf.interpolate(flow, scale_factor=0.5, mode='trilinear',
                                                                       align_corners=True))
                Fm0 = self.transformer[1](conv0_float, flow)

            sflow = 0
            sv = 0
            # level 4  level wise recusion
            for j in range(self.levels[3]):
                if j == 0:
                    w = self.est3(conv3_fixed, Fm3)
                    sflow = self.integrate[4](w)
                    sv = w

                else :
                    w = self.est3(conv3_fixed, self.transformer[4](Fm3, sflow))
                    sv = self.transformer[4](sv, w) + w
                    w = self.integrate[4](w)
                    sflow = self.transformer[4](sflow, w) + w

            # level 3
            sflow = self.upsample_trilin(2*sflow)
            sv = self.upsample_trilin(2*sv)
            for j in range(self.levels[2]):
                w = self.est2(conv2_fixed, self.transformer[3](Fm2, sflow))
                sv = self.transformer[3](sv, w) + w
                w = self.integrate[3](w)
                sflow = self.transformer[3](sflow, w) + w

            # level 2
            sflow = self.upsample_trilin(2*sflow)
            sv = self.upsample_trilin(2 * sv)
            for j in range(self.levels[1]):
                w = self.est1(conv1_fixed, self.transformer[2](Fm1, sflow))
                sv = self.transformer[2](sv, w) + w
                w = self.integrate[2](w)
                sflow = self.transformer[2](sflow, w) + w

            # level 1
            sflow = self.upsample_trilin(2*sflow)
            sv = self.upsample_trilin(2 * sv)
            for j in range(self.levels[0]):
                w = self.est0(conv0_fixed, self.transformer[1](Fm0, sflow))
                sv = self.transformer[1](sv, w) + w
                w = self.integrate[1](w)
                sflow = self.transformer[1](sflow, w) + w

            if i == 0:
                flow = sflow
            else :
                flow = self.transformer[1](flow, sflow) + sflow
            svs.append(sv)

        flow_out = self.upsample_trilin(2*flow)
        y_moved = self.transformer[0](moving, flow_out)
        # moves.append(y_moved)


        return y_moved, flow_out, *svs#moves, flow_out, sflows

class RDN_diff(nn.Module):
    def __init__(self, inshape=(160, 192, 160), in_channel=1, channels=16,stage_recursion=1, level_recursion=[1,1,1,1]):
        super(RDN_diff, self).__init__()

        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.stages = stage_recursion
        self.levels = level_recursion
        # dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)  # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.est3 = nn.ModuleList()
        self.est2 = nn.ModuleList()
        self.est1 = nn.ModuleList()
        self.est0 = nn.ModuleList()
        for _ in range(self.stages):
            self.est3.append(Estimator(2 * 8 * c))
            self.est2.append(Estimator(2 * 4 * c))
            self.est1.append(Estimator(2 * 2 * c))
            self.est0.append(Estimator(2 * c))

        self.transformer = nn.ModuleList()
        self.integrate = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s//2**i for s in inshape]))
            self.integrate.append(VecInt([s//2**i for s in inshape]))

    def forward(self, moving, fixed):
        # encode stage
        conv0_float, conv1_float, conv2_float, conv3_float = self.encoder(moving)
        conv0_fixed, conv1_fixed, conv2_fixed, conv3_fixed = self.encoder(fixed)
        # c, 2c, 4c, 8c  # 1/2, 1/4, 1/8, 1/16

        flow = 0
        # moves = []
        # sflows = []
        svs = []
        # stage wise recursion
        for i in range(self.stages):
            if i==0 :
                Fm0, Fm1, Fm2, Fm3 = conv0_float, conv1_float, conv2_float, conv3_float
            else :
                Fm3 = self.transformer[4](conv3_float, 0.125*nnf.interpolate(flow, scale_factor=0.125, mode='trilinear',
                                                                       align_corners=True))
                Fm2 = self.transformer[3](conv2_float, 0.25*nnf.interpolate(flow, scale_factor=0.25, mode='trilinear',
                                                                       align_corners=True))
                Fm1 = self.transformer[2](conv1_float, 0.5*nnf.interpolate(flow, scale_factor=0.5, mode='trilinear',
                                                                       align_corners=True))
                Fm0 = self.transformer[1](conv0_float, flow)

            sflow = 0
            sv = 0
            # level 4  level wise recusion
            for j in range(self.levels[3]):
                if j == 0:
                    w = self.est3[i](conv3_fixed, Fm3)
                    sflow = self.integrate[4](w)
                    sv = w

                else :
                    w = self.est3[i](conv3_fixed, self.transformer[4](Fm3, sflow))
                    sv = self.transformer[4](sv, w) + w
                    w = self.integrate[4](w)
                    sflow = self.transformer[4](sflow, w) + w

            # level 3
            sflow = self.upsample_trilin(2*sflow)
            sv = self.upsample_trilin(2*sv)
            for j in range(self.levels[2]):
                w = self.est2[i](conv2_fixed, self.transformer[3](Fm2, sflow))
                sv = self.transformer[3](sv, w) + w
                w = self.integrate[3](w)
                sflow = self.transformer[3](sflow, w) + w

            # level 2
            sflow = self.upsample_trilin(2*sflow)
            sv = self.upsample_trilin(2 * sv)
            for j in range(self.levels[1]):
                w = self.est1[i](conv1_fixed, self.transformer[2](Fm1, sflow))
                sv = self.transformer[2](sv, w) + w
                w = self.integrate[2](w)
                sflow = self.transformer[2](sflow, w) + w

            # level 1
            sflow = self.upsample_trilin(2*sflow)
            sv = self.upsample_trilin(2 * sv)
            for j in range(self.levels[0]):
                w = self.est0[i](conv0_fixed, self.transformer[1](Fm0, sflow))
                sv = self.transformer[1](sv, w) + w
                w = self.integrate[1](w)
                sflow = self.transformer[1](sflow, w) + w

            if i == 0:
                flow = sflow
            else :
                flow = self.transformer[1](flow, sflow) + sflow
            svs.append(sv)

        flow_out = self.upsample_trilin(2*flow)
        y_moved = self.transformer[0](moving, flow_out)
        # moves.append(y_moved)


        return y_moved, flow_out, *svs#moves, flow_out, sflows

class RDN(nn.Module):
    def __init__(self, inshape=(160, 192, 160), in_channel=1, channels=16,stage_recursion=1, level_recursion=[1,1,1,1]):
        super(RDN, self).__init__()

        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.stages = stage_recursion
        self.levels = level_recursion
        # dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)  # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.est3 = nn.ModuleList()
        self.est2 = nn.ModuleList()
        self.est1 = nn.ModuleList()
        self.est0 = nn.ModuleList()
        for _ in range(self.stages):
            self.est3.append(Estimator(2 * 8 * c))
            self.est2.append(Estimator(2 * 4 * c))
            self.est1.append(Estimator(2 * 2 * c))
            self.est0.append(Estimator(2 * c))

        self.transformer = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s//2**i for s in inshape]))

    def forward(self, moving, fixed):
        # encode stage
        conv0_float, conv1_float, conv2_float, conv3_float = self.encoder(moving)
        conv0_fixed, conv1_fixed, conv2_fixed, conv3_fixed = self.encoder(fixed)
        # c, 2c, 4c, 8c  # 1/2, 1/4, 1/8, 1/16

        flow = 0
        # moves = []
        sflows = []
        # svs = []
        # stage wise recursion
        for i in range(self.stages):
            if i==0 :
                Fm0, Fm1, Fm2, Fm3 = conv0_float, conv1_float, conv2_float, conv3_float
            else :
                Fm3 = self.transformer[4](conv3_float, 0.125*nnf.interpolate(flow, scale_factor=0.125, mode='trilinear',
                                                                       align_corners=True))
                Fm2 = self.transformer[3](conv2_float, 0.25*nnf.interpolate(flow, scale_factor=0.25, mode='trilinear',
                                                                       align_corners=True))
                Fm1 = self.transformer[2](conv1_float, 0.5*nnf.interpolate(flow, scale_factor=0.5, mode='trilinear',
                                                                       align_corners=True))
                Fm0 = self.transformer[1](conv0_float, flow)

            sflow = 0
            # level 4  level wise recusion
            for j in range(self.levels[3]):
                if j == 0:
                    sflow = self.est3[i](conv3_fixed, Fm3)

                else :
                    w = self.est3[i](conv3_fixed, self.transformer[4](Fm3, sflow))
                    sflow = self.transformer[4](sflow, w) + w

            # level 3
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[2]):
                w = self.est2[i](conv2_fixed, self.transformer[3](Fm2, sflow))
                sflow = self.transformer[3](sflow, w) + w

            # level 2
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[1]):
                w = self.est1[i](conv1_fixed, self.transformer[2](Fm1, sflow))
                sflow = self.transformer[2](sflow, w) + w

            # level 1
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[0]):
                w = self.est0[i](conv0_fixed, self.transformer[1](Fm0, sflow))
                sflow = self.transformer[1](sflow, w) + w

            if i == 0:
                flow = sflow
            else :
                flow = self.transformer[1](flow, sflow) + sflow
            sflows.append(sflow)

        flow_out = self.upsample_trilin(2*flow)
        y_moved = self.transformer[0](moving, flow_out)

        return y_moved, flow_out, *sflows#moves, flow_out, sflows


class RDN_test(nn.Module):
    def __init__(self, inshape=(160, 192, 160), in_channel=1, channels=16,stage_recursion=1, level_recursion=[1,1,1,1]):
        super(RDN_test, self).__init__()

        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.stages = stage_recursion
        self.levels = level_recursion
        # dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)  # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.est3 = nn.ModuleList()
        self.est2 = nn.ModuleList()
        self.est1 = nn.ModuleList()
        self.est0 = nn.ModuleList()
        for _ in range(self.stages):
            self.est3.append(Estimator(2 * 8 * c))
            self.est2.append(Estimator(2 * 4 * c))
            self.est1.append(Estimator(2 * 2 * c))
            self.est0.append(Estimator(2 * c))

        self.transformer = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s//2**i for s in inshape]))

    def forward(self, moving, fixed):
        # encode stage
        conv0_float, conv1_float, conv2_float, conv3_float = self.encoder(moving)
        conv0_fixed, conv1_fixed, conv2_fixed, conv3_fixed = self.encoder(fixed)
        # c, 2c, 4c, 8c  # 1/2, 1/4, 1/8, 1/16

        flow = 0
        # stage wise recursion
        for i in range(self.stages):
            if i==0 :
                Fm0, Fm1, Fm2, Fm3 = conv0_float, conv1_float, conv2_float, conv3_float
            else :
                Fm3 = self.transformer[4](conv3_float, 0.125*nnf.interpolate(flow, scale_factor=0.125, mode='trilinear',
                                                                       align_corners=True))
                Fm2 = self.transformer[3](conv2_float, 0.25*nnf.interpolate(flow, scale_factor=0.25, mode='trilinear',
                                                                       align_corners=True))
                Fm1 = self.transformer[2](conv1_float, 0.5*nnf.interpolate(flow, scale_factor=0.5, mode='trilinear',
                                                                       align_corners=True))
                Fm0 = self.transformer[1](conv0_float, flow)

            sflow = 0
            # level 4  level wise recusion
            for j in range(self.levels[3]):
                if j == 0:
                    sflow = self.est3[i](conv3_fixed, Fm3)

                else :
                    w = self.est3[i](conv3_fixed, self.transformer[4](Fm3, sflow))
                    sflow = self.transformer[4](sflow, w) + w

            # level 3
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[2]):
                w = self.est2[i](conv2_fixed, self.transformer[3](Fm2, sflow))
                sflow = self.transformer[3](sflow, w) + w

            # level 2
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[1]):
                w = self.est1[i](conv1_fixed, self.transformer[2](Fm1, sflow))
                sflow = self.transformer[2](sflow, w) + w

            # level 1
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[0]):
                w = self.est0[i](conv0_fixed, self.transformer[1](Fm0, sflow))
                sflow = self.transformer[1](sflow, w) + w

            if i == 0:
                flow = sflow
            else :
                flow = self.transformer[1](flow, sflow) + sflow

        flow_out = self.upsample_trilin(2*flow)
        y_moved = self.transformer[0](moving, flow_out)

        return y_moved, flow_out

class RDN_share(nn.Module):
    def __init__(self, inshape=(160, 192, 160), in_channel=1, channels=16,stage_recursion=1, level_recursion=[1,1,1,1]):
        super(RDN_share, self).__init__()

        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.stages = stage_recursion
        self.levels = level_recursion
        # dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)

        self.est3 = Estimator(2 * 8 * c)
        self.est2 = Estimator(2 * 4 * c)
        self.est1 = Estimator(2 * 2 * c)
        self.est0 = Estimator(2 * c)

        self.transformer = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s//2**i for s in inshape]))

    def forward(self, moving, fixed):
        # encode stage
        conv0_float, conv1_float, conv2_float, conv3_float = self.encoder(moving)
        conv0_fixed, conv1_fixed, conv2_fixed, conv3_fixed = self.encoder(fixed)
        # c, 2c, 4c, 8c  # 1/2, 1/4, 1/8, 1/16

        flow = 0
        # moves = []
        sflows = []
        # svs = []
        # stage wise recursion
        for i in range(self.stages):
            if i == 0:
                Fm0, Fm1, Fm2, Fm3 = conv0_float, conv1_float, conv2_float, conv3_float
            else:
                Fm3 = self.transformer[4](conv3_float,
                                          0.125 * nnf.interpolate(flow, scale_factor=0.125, mode='trilinear',
                                                                  align_corners=True))
                Fm2 = self.transformer[3](conv2_float, 0.25 * nnf.interpolate(flow, scale_factor=0.25, mode='trilinear',
                                                                              align_corners=True))
                Fm1 = self.transformer[2](conv1_float, 0.5 * nnf.interpolate(flow, scale_factor=0.5, mode='trilinear',
                                                                             align_corners=True))
                Fm0 = self.transformer[1](conv0_float, flow)

            sflow = 0
            # level 4  level wise recusion
            for j in range(self.levels[3]):
                if j == 0:
                    sflow = self.est3(conv3_fixed, Fm3)

                else:
                    w = self.est3(conv3_fixed, self.transformer[4](Fm3, sflow))
                    sflow = self.transformer[4](sflow, w) + w

            # level 3
            sflow = self.upsample_trilin(2 * sflow)
            for j in range(self.levels[2]):
                w = self.est2(conv2_fixed, self.transformer[3](Fm2, sflow))
                sflow = self.transformer[3](sflow, w) + w

            # level 2
            sflow = self.upsample_trilin(2 * sflow)
            for j in range(self.levels[1]):
                w = self.est1(conv1_fixed, self.transformer[2](Fm1, sflow))
                sflow = self.transformer[2](sflow, w) + w

            # level 1
            sflow = self.upsample_trilin(2 * sflow)
            for j in range(self.levels[0]):
                w = self.est0(conv0_fixed, self.transformer[1](Fm0, sflow))
                sflow = self.transformer[1](sflow, w) + w

            if i == 0:
                flow = sflow
            else:
                flow = self.transformer[1](flow, sflow) + sflow
            sflows.append(sflow)

        flow_out = self.upsample_trilin(2 * flow)
        y_moved = self.transformer[0](moving, flow_out)

        return y_moved, flow_out, *sflows  # moves, flow_out, sflows

class RDN_share_test(nn.Module):
    def __init__(self, inshape=(160, 192, 160), in_channel=1, channels=16,stage_recursion=1, level_recursion=[1,1,1,1]):
        super(RDN_share_test, self).__init__()

        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.stages = stage_recursion
        self.levels = level_recursion
        # dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)

        self.est3 = Estimator(2 * 8 * c)
        self.est2 = Estimator(2 * 4 * c)
        self.est1 = Estimator(2 * 2 * c)
        self.est0 = Estimator(2 * c)

        self.transformer = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s//2**i for s in inshape]))

    def forward(self, moving, fixed):
        # encode stage
        conv0_float, conv1_float, conv2_float, conv3_float = self.encoder(moving)
        conv0_fixed, conv1_fixed, conv2_fixed, conv3_fixed = self.encoder(fixed)
        # c, 2c, 4c, 8c  # 1/2, 1/4, 1/8, 1/16

        flow = 0
        # stage wise recursion
        for i in range(self.stages):
            if i == 0:
                Fm0, Fm1, Fm2, Fm3 = conv0_float, conv1_float, conv2_float, conv3_float
            else:
                Fm3 = self.transformer[4](conv3_float,
                                          0.125 * nnf.interpolate(flow, scale_factor=0.125, mode='trilinear',
                                                                  align_corners=True))
                Fm2 = self.transformer[3](conv2_float, 0.25 * nnf.interpolate(flow, scale_factor=0.25, mode='trilinear',
                                                                              align_corners=True))
                Fm1 = self.transformer[2](conv1_float, 0.5 * nnf.interpolate(flow, scale_factor=0.5, mode='trilinear',
                                                                             align_corners=True))
                Fm0 = self.transformer[1](conv0_float, flow)

            sflow = 0
            # level 4  level wise recusion
            for j in range(self.levels[3]):
                if j == 0:
                    sflow = self.est3(conv3_fixed, Fm3)

                else:
                    w = self.est3(conv3_fixed, self.transformer[4](Fm3, sflow))
                    sflow = self.transformer[4](sflow, w) + w

            # level 3
            sflow = self.upsample_trilin(2 * sflow)
            for j in range(self.levels[2]):
                w = self.est2(conv2_fixed, self.transformer[3](Fm2, sflow))
                sflow = self.transformer[3](sflow, w) + w

            # level 2
            sflow = self.upsample_trilin(2 * sflow)
            for j in range(self.levels[1]):
                w = self.est1(conv1_fixed, self.transformer[2](Fm1, sflow))
                sflow = self.transformer[2](sflow, w) + w

            # level 1
            sflow = self.upsample_trilin(2 * sflow)
            for j in range(self.levels[0]):
                w = self.est0(conv0_fixed, self.transformer[1](Fm0, sflow))
                sflow = self.transformer[1](sflow, w) + w

            if i == 0:
                flow = sflow
            else:
                flow = self.transformer[1](flow, sflow) + sflow

        flow_out = self.upsample_trilin(2 * flow)
        y_moved = self.transformer[0](moving, flow_out)

        return y_moved, flow_out

class RDN_diff_share_test(nn.Module):
    def __init__(self, inshape=(160, 192, 160), in_channel=1, channels=16,stage_recursion=1, level_recursion=[1,1,1,1]):
        super(RDN_diff_share_test, self).__init__()

        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.stages = stage_recursion
        self.levels = level_recursion
        # dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)  # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.est3 = Estimator(2 * 8 * c)
        self.est2 = Estimator(2 * 4 * c)
        self.est1 = Estimator(2 * 2 * c)
        self.est0 = Estimator(2 * c)

        self.transformer = nn.ModuleList()
        self.integrate = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s//2**i for s in inshape]))
            self.integrate.append(VecInt([s//2**i for s in inshape]))

    def forward(self, moving, fixed):
        # encode stage
        conv0_float, conv1_float, conv2_float, conv3_float = self.encoder(moving)
        conv0_fixed, conv1_fixed, conv2_fixed, conv3_fixed = self.encoder(fixed)
        # c, 2c, 4c, 8c  # 1/2, 1/4, 1/8, 1/16

        flow = 0
        # stage wise recursion
        for i in range(self.stages):
            if i==0 :
                Fm0, Fm1, Fm2, Fm3 = conv0_float, conv1_float, conv2_float, conv3_float
            else :
                Fm3 = self.transformer[4](conv3_float, 0.125*nnf.interpolate(flow, scale_factor=0.125, mode='trilinear',
                                                                       align_corners=True))
                Fm2 = self.transformer[3](conv2_float, 0.25*nnf.interpolate(flow, scale_factor=0.25, mode='trilinear',
                                                                       align_corners=True))
                Fm1 = self.transformer[2](conv1_float, 0.5*nnf.interpolate(flow, scale_factor=0.5, mode='trilinear',
                                                                       align_corners=True))
                Fm0 = self.transformer[1](conv0_float, flow)

            sflow = 0
            # level 4  level wise recusion
            for j in range(self.levels[3]):
                if j == 0:
                    w = self.est3(conv3_fixed, Fm3)
                    sflow = self.integrate[4](w)

                else :
                    w = self.est3(conv3_fixed, self.transformer[4](Fm3, sflow))
                    w = self.integrate[4](w)
                    sflow = self.transformer[4](sflow, w) + w

            # level 3
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[2]):
                w = self.est2(conv2_fixed, self.transformer[3](Fm2, sflow))
                w = self.integrate[3](w)
                sflow = self.transformer[3](sflow, w) + w

            # level 2
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[1]):
                w = self.est1(conv1_fixed, self.transformer[2](Fm1, sflow))
                w = self.integrate[2](w)
                sflow = self.transformer[2](sflow, w) + w

            # level 1
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[0]):
                w = self.est0(conv0_fixed, self.transformer[1](Fm0, sflow))
                w = self.integrate[1](w)
                sflow = self.transformer[1](sflow, w) + w

            if i == 0:
                flow = sflow
            else :
                flow = self.transformer[1](flow, sflow) + sflow

        flow_out = self.upsample_trilin(2*flow)
        y_moved = self.transformer[0](moving, flow_out)
        # moves.append(y_moved)


        return y_moved, flow_out #moves, flow_out, sflows

class RDN_diff_test(nn.Module):
    def __init__(self, inshape=(160, 192, 160), in_channel=1, channels=16, stage_recursion=1, level_recursion=[1,1,1,1]):
        super(RDN_diff_test, self).__init__()

        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.stages = stage_recursion
        self.levels = level_recursion
        # dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear',
                                           align_corners=True)  # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.est3 = nn.ModuleList()
        self.est2 = nn.ModuleList()
        self.est1 = nn.ModuleList()
        self.est0 = nn.ModuleList()
        for _ in range(self.stages):
            self.est3.append(Estimator(2 * 8 * c))
            self.est2.append(Estimator(2 * 4 * c))
            self.est1.append(Estimator(2 * 2 * c))
            self.est0.append(Estimator(2 * c))

        self.transformer = nn.ModuleList()
        self.integrate = nn.ModuleList()
        for i in range(5):
            self.transformer.append(SpatialTransformer([s//2**i for s in inshape]))
            self.integrate.append(VecInt([s//2**i for s in inshape]))

    def forward(self, moving, fixed):
        # encode stage
        conv0_float, conv1_float, conv2_float, conv3_float = self.encoder(moving)
        conv0_fixed, conv1_fixed, conv2_fixed, conv3_fixed = self.encoder(fixed)
        # c, 2c, 4c, 8c  # 1/2, 1/4, 1/8, 1/16

        flow = 0

        # stage wise recursion
        for i in range(self.stages):
            if i==0 :
                Fm0, Fm1, Fm2, Fm3 = conv0_float, conv1_float, conv2_float, conv3_float
            else :
                Fm3 = self.transformer[4](conv3_float, 0.125*nnf.interpolate(flow, scale_factor=0.125, mode='trilinear',
                                                                       align_corners=True))
                Fm2 = self.transformer[3](conv2_float, 0.25*nnf.interpolate(flow, scale_factor=0.25, mode='trilinear',
                                                                       align_corners=True))
                Fm1 = self.transformer[2](conv1_float, 0.5*nnf.interpolate(flow, scale_factor=0.5, mode='trilinear',
                                                                       align_corners=True))
                Fm0 = self.transformer[1](conv0_float, flow)
            sflow = 0
            # level 4  level wise recusion
            for j in range(self.levels[3]):
                if j == 0:
                    w = self.est3[i](conv3_fixed, Fm3)
                    sflow = self.integrate[4](w)

                else :
                    w = self.est3[i](conv3_fixed, self.transformer[4](Fm3, sflow))
                    w = self.integrate[4](w)
                    sflow = self.transformer[4](sflow, w) + w

            # level 3
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[2]):
                w = self.est2[i](conv2_fixed, self.transformer[3](Fm2, sflow))
                w = self.integrate[3](w)
                sflow = self.transformer[3](sflow, w) + w

            # level 2
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[1]):
                w = self.est1[i](conv1_fixed, self.transformer[2](Fm1, sflow))
                w = self.integrate[2](w)
                sflow = self.transformer[2](sflow, w) + w

            # level 1
            sflow = self.upsample_trilin(2*sflow)
            for j in range(self.levels[0]):
                w = self.est0[i](conv0_fixed, self.transformer[1](Fm0, sflow))
                w = self.integrate[1](w)
                sflow = self.transformer[1](sflow, w) + w

            if i == 0:
                flow = sflow
            else :
                flow = self.transformer[1](flow, sflow) + sflow

        flow_out = self.upsample_trilin(2*flow)
        y_moved = self.transformer[0](moving, flow_out)

        return y_moved, flow_out

if __name__ == '__main__':
    #     # model = VoxResNet().cuda()
    #     # A = torch.ones((1,1,160,196,160))
    #     # B = torch.ones((1,1,160,196,160))
    #     # output1 = model(A.cuda())
    #     # output2 = model(B.cuda())
    #     # for i in range(len(output2)):
    #     #     print(torch.sum(output1[i]==output2[i]).item())
    #     #     print(output1[i].shape[0]*output1[i].shape[1]*output1[i].shape[2]*output1[i].shape[3]*output1[i].shape[4])
    size = (1, 1, 80, 96, 80)
    model = RDN_diff_test(size[2:]).cuda(2)
    # print(str(model))
    A = torch.ones(size)
    B = torch.ones(size)
    out, flow = model(A.cuda(2), B.cuda(2))
    print(out.shape, flow.shape)