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

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
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

    def __init__(self, ndims, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
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
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)

class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )
    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2*c, kernel_size=3, stride=2, padding=1),#80
            ResBlock(2*c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2*c, 4*c, kernel_size=3, stride=2, padding=1),#40
            ResBlock(4*c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4*c, 8*c, kernel_size=3, stride=2, padding=1),#20
            ResBlock(8*c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8

        return [out0, out1, out2, out3]



class DFIBlock(nn.Module):
    def __init__(self,inshape, level, steps=7, channel=16):
        super(DFIBlock, self).__init__()

        c = channel
        list_num = 5-level
        self.list_num = list_num

        self.conv = nn.Sequential(
            ConvInsBlock(3*list_num, c*list_num, 3, 1),
            ConvInsBlock(c * list_num, c * list_num, 3, 1)
        )

        self.weight_conv = nn.ModuleList([])
        self.upsample = nn.ModuleList([])
        for i in range(list_num):
            self.weight_conv.append(
                nn.Sequential(
                nn.Conv3d(c * list_num, 3, 3, 1, 1),
                nn.Sigmoid()
                )
            )
            self.upsample.append(nn.Upsample(
                scale_factor=2**(list_num-i),
                mode='trilinear',
                align_corners=True
            ))
        self.integrate = VecInt(inshape, steps)
    def forward(self, prediction_list):
        prediction_cache = []
        for i, prediction in enumerate(prediction_list):
            prediction_cache.append(self.upsample[i](prediction))

        x = torch.cat(prediction_cache, dim=1)  # 将采样到相同大小的形变场级联
        x = self.conv(x)

        for i, prediction in enumerate(prediction_cache):
            weight_map = self.weight_conv[i](x)
            prediction_cache[i] = prediction * weight_map  # 每个形变场用一个不同的权重卷积层算权重加权 占用7G
            if i == 0:
                progress_field = prediction_cache[i]
            else:
                progress_field = progress_field + prediction_cache[i]  # 加权后的形变场相加
            # print(progress_field)
        progress_field = self.integrate(progress_field)
        return progress_field

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)  # (1, c)
        y_max = self.max_pool(x).view(b, c)  # (1, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1, 1)

        return self.sigmoid(y_avg+y_max)  # (1, c, 1, 1, 1)

class NFFBlock(nn.Module):
    def __init__(self, channel):
        super(NFFBlock, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            ConvInsBlock(c, c, 3, 1),
            ConvInsBlock(c, c, 3, 1)
        )

        self.weight_conv = nn.Sequential(
            nn.Conv3d(c, 3, 3, 1, 1),
            nn.Softmax(dim=1)
        )

        self.channel_attention = ChannelAttention(c)

    def forward(self, float_fm, fixed_fm, decon_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, decon_fm], dim=1)
        x = self.conv(concat_fm)
        weight_map = self.weight_conv(x)
        concat = torch.cat([
            float_fm * weight_map[:, 0, ...].unsqueeze(1),
            fixed_fm * weight_map[:, 1, ...].unsqueeze(1),
            decon_fm * weight_map[:, 2, ...].unsqueeze(1)
        ], dim=1) # (1, 3*c, h, w, t)
        channel_wise = self.channel_attention(concat)
        return concat*channel_wise

class PCNet(nn.Module):
    def __init__(self, inshape=(160,192,160), flow_multiplier=1.,in_channel=1, channels=16):
        super(PCNet, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        dims = len(inshape)

        c = self.channels
        self.encoder_float = Encoder(in_channel=in_channel, first_out_channel=c)
        self.encoder_fixed = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # bottleNeck
        self.conv_bottleNeck = nn.Sequential(
            ConvInsBlock(16 * c, 8 * c, 3, 1),
            ConvInsBlock(8 * c, 8 * c, 3, 1)
        )

        # warp scale 2
        self.reg_conv3 = nn.Conv3d(2*4*c, 3, 3, 1, 1)
        self.upconv2 = UpConvBlock(8*c, 4*c, 4, 2)
        self.dfi_2 = DFIBlock([s//4 for s in inshape], 4)
        self.transformer_2 = SpatialTransformer([s//4 for s in inshape])
        self.nff_2 = NFFBlock(3*4*c)

        # warp scale 1
        self.reg_conv2 = nn.Conv3d(3*4*c, 3, 3, 1, 1)
        self.upconv1 = UpConvBlock(3*4*c, 2*c, 4, 2)
        self.dfi_1 = DFIBlock([s//2 for s in inshape], 3)
        self.transformer_1 = SpatialTransformer([s//2 for s in inshape])
        self.nff_1 = NFFBlock(3*2*c)

        # warp scale 0
        self.reg_conv1 = nn.Conv3d(3*2*c, 3, 3, 1, 1)
        self.upconv0 = UpConvBlock(3*2*c, c, 4, 2)
        self.dfi_0 = DFIBlock(inshape, 2)
        self.transformer_0 = SpatialTransformer(inshape)
        self.nff_0 = NFFBlock(3*c)

        # decoder layers
        self.reg_conv0 = nn.Sequential(
            nn.Conv3d(3*c, c, 3, 1, 1),
            nn.Conv3d(c, 3, 3, 1, 1)
        )

        self.integrate = VecInt(inshape)
        self.transformer_out = SpatialTransformer(inshape)
        # self.transformers = nn.ModuleList()
        # for i in range(self.N):
        #     fmap_size = [s // self.fmap_scale_factors[i] for s in inshape]
        #     self.transformers.append(SpatialTransformer(fmap_size))

    def forward(self, moving, fixed):

        # encode stage
        conv0_float, conv1_float, conv2_float, conv3_float = self.encoder_float(moving)
        conv0_fixed, conv1_fixed, conv2_fixed, conv3_fixed = self.encoder_fixed(fixed)
        # c=16, 2c, 4c, 8c  # 160, 80, 40, 20
        fields = []
        # first dec layer
        concat_bottleNeck = torch.cat([conv3_fixed, conv3_float], dim=1)
        concat_bottleNeck = self.conv_bottleNeck(concat_bottleNeck)  # (1,128,20,24,20)


        predict_cache = []

        # warping scale 2
        pred3 = self.reg_conv3(concat_bottleNeck)  # (1,3,20,24,20)
        predict_cache.append(pred3)  # 输出第一个速度场

        deconv2 = self.upconv2(concat_bottleNeck)   # (1, 64, 40, 48, 40)
        warping_field_2 = self.dfi_2(predict_cache)  # (1,3,40,48,40)
        conv2_float = self.transformer_2(conv2_float, warping_field_2)  # (1, 64, 40, 48, 40)
        concat2 = self.nff_2(conv2_fixed, conv2_float, deconv2)  #  (1, 3 * 64, 40, 48, 40)

        # warping scale 1
        pred2 = self.reg_conv2(concat2)  # 20
        predict_cache.append(pred2)

        deconv1 = self.upconv1(concat2)
        warping_field_1 = self.dfi_1(predict_cache)  # 第一个形变场 20
        conv1_float = self.transformer_1(conv1_float, warping_field_1)
        concat1 = self.nff_1(conv1_fixed, conv1_float, deconv1)

        # warping scale 0
        pred1 = self.reg_conv1(concat1)  # (1,3,80,96,80)  # 20
        predict_cache.append(pred1)

        deconv0 = self.upconv0(concat1)  # (1,16,160,196,160)
        warping_field_0 = self.dfi_0(predict_cache)  # （1,3,160,196,160)
        conv0_float = self.transformer_0(conv0_float, warping_field_0)  # （1,16,160,196,160)
        concat0 = self.nff_0(conv0_fixed, conv0_float, deconv0)  # （1,48,160,196,160)

        pred0 = self.reg_conv0(concat0)
        pred0 = self.integrate(pred0)
        flow = self.transformer_out(warping_field_0, pred0) + pred0

        y_moved = self.transformer_out(moving, flow)

        return y_moved, flow

if __name__ == '__main__':
#     # model = VoxResNet().cuda()
#     # A = torch.ones((1,1,160,196,160))
#     # B = torch.ones((1,1,160,196,160))
#     # output1 = model(A.cuda())
#     # output2 = model(B.cuda())
#     # for i in range(len(output2)):
#     #     print(torch.sum(output1[i]==output2[i]).item())
#     #     print(output1[i].shape[0]*output1[i].shape[1]*output1[i].shape[2]*output1[i].shape[3]*output1[i].shape[4])
    model = PCNet().cuda(2)
    # print(str(model))
    A = torch.ones((1, 1, 160, 192, 160))
    B = torch.ones((1, 1, 160, 192, 160))
    out, flow = model(A.cuda(2), B.cuda(2))
    print(out.shape, flow.shape)