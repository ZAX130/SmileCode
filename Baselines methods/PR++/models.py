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

class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.factor = scale_factor
        self.mode = 'trilinear'

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


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out




class Encoder(nn.Module):
    def __init__(self, in_channel=1, first_channel=8):
        super(Encoder, self).__init__()
        c = first_channel
        self.block1 = ConvBlock(in_channel, c)
        self.block2 = ConvBlock(c, c * 2, stride=2)
        self.block3 = ConvBlock(c *2, c * 2, stride=2)
        self.block4 = ConvBlock(c *2, c * 4, stride=2)
        self.block5 = ConvBlock(c * 4, c * 4, stride=2)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        return out1, out2, out3, out4, out5

class DecoderBlock(nn.Module):
    def __init__(self, deconv_channel, skip_channel, out_channel):
        super(DecoderBlock, self).__init__()
        self.Upsample = nn.Upsample(scale_factor=2)
        self.Conv = ConvBlock(deconv_channel+skip_channel, out_channel)

    def forward(self, x, skip):
        out = self.Upsample(x)
        concat = torch.cat([out, skip], dim=1)
        out = self.Conv(concat)
        return out

class BackBone(nn.Module):
    def __init__(self, in_channel=1, first_channel=8):
        super(BackBone, self).__init__()
        c = first_channel
        self.encoder = Encoder(in_channel, c)
        self.decoder1 = DecoderBlock(deconv_channel=c * 4, skip_channel=c * 4, out_channel=c * 4)
        self.decoder2 = DecoderBlock(deconv_channel=c * 4, skip_channel=c * 2, out_channel=c * 4)
        self.decoder3 = DecoderBlock(deconv_channel=c * 4, skip_channel=c * 2, out_channel=c * 2)
        self.decoder4 = DecoderBlock(deconv_channel=c * 2, skip_channel=c, out_channel=c * 2)
        self.decoder5 = ConvBlock(in_channels=2*c, out_channels=c)

    def forward(self, x, y):
        feat_x5, feat_x4, feat_x3, feat_x2, feat_x1 = self.encoder(x)
        feat_y5, feat_y4, feat_y3, feat_y2, feat_y1 = self.encoder(y)
        # 8, 16, 16, 32, 32 # 1, 1/2, 1/4, 1/6, 1/8
        #Decoder
        out_x1 = self.decoder1(feat_x1, feat_x2)  # (32, 1/8)
        out_x2 = self.decoder2(out_x1, feat_x3)  # (32, 1/4)
        out_x3 = self.decoder3(out_x2, feat_x4)  # (16, 1/2)
        out_x4 = self.decoder4(out_x3, feat_x5)  # (16, 1)
        out_x5 = self.decoder5(out_x4)  # (8, 1)
        output_x = [out_x1, out_x2, out_x3, out_x4, out_x5]

        out_y1 = self.decoder1(feat_y1, feat_y2)
        out_y2 = self.decoder2(out_y1, feat_y3)
        out_y3 = self.decoder3(out_y2, feat_y4)
        out_y4 = self.decoder4(out_y3, feat_y5)
        out_y5 = self.decoder5(out_y4)
        output_y = [out_y1, out_y2, out_y3, out_y4, out_y5]
        return output_x, output_y

class PRBlock(nn.Module):
    def __init__(self, size, in_channel, in_flow=True, scale=True):
        super(PRBlock, self).__init__()
        self.scale = scale
        self.in_flow = in_flow
        if in_flow:
            self.stn = SpatialTransformer(size)
            if scale:
                self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.flow = nn.Conv3d(in_channel*2, 3, 3, 1, 1)
        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, x, y, flow=None):
        if self.in_flow:
            if self.scale:
                flow = self.upsample(flow*2)
            x = self.stn(x, flow)
        stack = torch.cat([x, y], dim=1)
        flow = self.flow(stack)
        return flow
class Correlation3D(nn.Module):
    """
    Main model
    """
    def __init__(self,in_channel, kernel_size=3, d=3, sw=1, sf=2):
        super(Correlation3D, self).__init__()
        self.kernel_size = kernel_size
        self.block = nn.Sequential()
        self.d = d
        self.sw = sw
        self.sf = sf
        self.w = torch.ones((in_channel, 1, self.kernel_size, self.kernel_size, self.kernel_size)).cuda()

    def forward(self, mov, fix):
        B, C, H, W, T = mov.shape

        pm = nnf.conv3d(mov, self.w, stride=self.sw, padding=1, groups=C) # H
        pf = nnf.conv3d(fix, self.w, stride=self.sw, padding=self.sf+1, groups=C)  # H+4

        concat = []
        for i in range(self.d):
            for j in range(self.d):
                for k in range(self.d):
                    pf_crop = pf[:, :, i*self.sf:(i*self.sf+H), j*self.sf:(j*self.sf+W), k*self.sf:(k*self.sf+T)]
                    concat.append(torch.sum(pm*pf_crop, dim=1, keepdim=True))

        corr = torch.cat(concat, dim=1)/self.kernel_size**3
        return corr

class PRplusplusBlock(nn.Module):
    def __init__(self, size, in_channel, in_flow=True, scale=True, kernel_size=3, d=3, sw=1, sf=2):
        super(PRplusplusBlock, self).__init__()
        self.scale = scale
        self.in_flow = in_flow
        if in_flow:
            self.stn = SpatialTransformer(size)
            if scale:
                self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.corr = Correlation3D(in_channel, kernel_size, d, sw, sf)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel*2+kernel_size**3, in_channel*2+kernel_size**3, 3, 1, 1),
            nn.Conv3d(in_channel * 2 + kernel_size ** 3, in_channel, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU()
        )
        self.flow = nn.Conv3d(in_channel, 3, 3, 1, 1)
        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, x, y, flow=None):
        if self.in_flow:
            if self.scale:
                flow = self.upsample(flow*2)
            x = self.stn(x, flow)
        corr = self.corr(x, y)

        stack = torch.cat([x, corr, y], dim=1)
        x = self.conv1(stack)
        res = self.conv2(x)
        flow = self.flow(x+res)
        return flow

class PRNet(nn.Module):
    def __init__(self, size=(80, 96, 80), in_channel=1, first_channel=8):
        super(PRNet, self).__init__()
        c = first_channel
        self.net = BackBone(in_channel, c)
        self.prblock1 = PRBlock(size=None, in_channel=4*c, in_flow=False, scale=False)
        self.prblock2 = PRBlock(size=[s // 4 for s in size], in_channel=4*c, in_flow=True, scale=True)
        self.prblock3 = PRBlock(size=[s // 2 for s in size], in_channel=2*c, in_flow=True, scale=True)
        self.prblock4 = PRBlock(size=size, in_channel=2*c, in_flow=True, scale=True)
        self.prblock5 = PRBlock(size=size, in_channel=c, in_flow=True, scale=False)


        self.transformers = nn.ModuleList()
        for i in range(3):
            self.transformers.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, y):
        output_x, output_y = self.net(x, y)
        flow1 = self.prblock1(output_x[0], output_y[0])
        flow2 = self.prblock2(output_x[1], output_y[1], flow=flow1)
        flow3 = self.prblock3(output_x[2], output_y[2], flow=flow2)
        flow4 = self.prblock4(output_x[3], output_y[3], flow=flow3)
        flow5 = self.prblock5(output_x[4], output_y[4], flow=flow4)

        flow = self.transformers[2](self.up(flow1* 2), flow2)
        flow = self.transformers[1](self.up(flow * 2), flow3)
        flow = self.transformers[0](self.up(flow * 2), flow4)
        flow = self.transformers[0](flow, flow5)

        y_moved = self.transformers[0](x, flow)

        return y_moved, flow

class PRNetplusplus(nn.Module):
    def __init__(self, size=(80, 96, 80), in_channel=1, first_channel=8):
        super(PRNetplusplus, self).__init__()
        c = first_channel
        self.net = BackBone(in_channel, c)
        self.prblock1 = PRplusplusBlock(size=None, in_channel=4*c, in_flow=False, scale=False)
        self.prblock2 = PRplusplusBlock(size=[s // 4 for s in size], in_channel=4*c, in_flow=True, scale=True)
        self.prblock3 = PRplusplusBlock(size=[s // 2 for s in size], in_channel=2*c, in_flow=True, scale=True)
        self.prblock4 = PRplusplusBlock(size=size, in_channel=2*c, in_flow=True, scale=True)
        self.prblock5 = PRplusplusBlock(size=size, in_channel=c, in_flow=True, scale=False)

        self.transformers = nn.ModuleList()
        for i in range(3):
            self.transformers.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = ResizeTransform(scale_factor=2)

    def forward(self, x, y):
        output_x, output_y = self.net(x, y)
        flow = self.prblock1(output_x[0], output_y[0])

        w = self.prblock2(output_x[1], output_y[1], flow=flow)
        flow = self.transformers[2](flow, w) + w

        w = self.prblock3(output_x[2], output_y[2], flow=flow)
        flow = self.transformers[1](flow, w) + w

        w = self.prblock4(output_x[3], output_y[3], flow=flow)
        flow = self.transformers[0](flow, w) + w

        w = self.prblock5(output_x[4], output_y[4], flow=flow)
        flow = self.transformers[0](flow, w) + w

        y_moved = self.transformers[0](x, flow)

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
    gpu_idx = 0
    torch.cuda.set_device(gpu_idx)
    size = (80, 96, 80)
    model = PRNetplusplus(size).cuda()
    # print(str(model))
    A = torch.ones((1, 1, *size))
    B = torch.ones((1, 1, *size))
    out, flow = model(A.cuda(), B.cuda())
    print(out.shape, flow.shape)