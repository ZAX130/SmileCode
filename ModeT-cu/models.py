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
from functional import modetqkrpb_cu

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

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
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
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        x = x[:,:,1:-1,1:-1,1:-1]
        return self.actout(x)

class DeconvBlock(nn.Module):
    def __init__(self, dec_channels, skip_channels):
        super(DeconvBlock, self).__init__()
        self.upconv = UpConvBlock(dec_channels, skip_channels)
        self.conv = nn.Sequential(
            ConvInsBlock(2*skip_channels, skip_channels),
            ConvInsBlock(skip_channels, skip_channels)
        )
    def forward(self, dec, skip):
        dec = self.upconv(dec)
        out = self.conv(torch.cat([dec, skip], dim=1))
        return out

class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvBlock(in_channel, c),
            ConvInsBlock(c, 2*c),
            ConvInsBlock(2*c, 2*c)
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8 * c, 16* c),
            ConvInsBlock(16 * c, 16 * c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(16 * c, 32 * c),
            ConvInsBlock(32 * c, 32 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8

        return out0, out1, out2, out3, out4

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat

class CWM(nn.Module):
    def __init__(self, in_channels, channels):
        super(CWM, self).__init__()

        c = channels
        self.num_fields = in_channels // 3

        self.conv = nn.Sequential(
            ConvInsBlock(in_channels, channels, 3, 1),
            ConvInsBlock(channels, channels, 3, 1),
            nn.Conv3d(channels, self.num_fields, 3, 1, 1),
            nn.Softmax(dim=1)
        )

        self.upsample = nn.Upsample(
                scale_factor=2,
                mode='trilinear',
                align_corners=True
            )

    def forward(self, x):

        x = self.upsample(x)
        weight = self.conv(x)

        weighted_field = 0

        for i in range(self.num_fields):
            w = x[:, 3*i: 3*(i+1)]
            weight_map = weight[:, i:(i+1)]
            weighted_field = weighted_field + w*weight_map

        return 2*weighted_field


class ModeTransformer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, qk_scale=None, use_rpb=True):
        super().__init__()


        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = kernel_size
        self.use_rpb = use_rpb
        if use_rpb:
            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.rpb_size, self.rpb_size, self.rpb_size))
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [kernel_size] * 3]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        v = grid.reshape(self.kernel_size**3, 3)
        self.register_buffer('v', v)

    def forward(self, q, k):

        B, H, W, T, C = q.shape

        q = q.reshape(B, H, W, T, self.num_heads, C // self.num_heads).permute(0,4,1,2,3,5) * self.scale  #1,heads,H,W,T,dims
        pd = self.kernel_size - 1  # 2
        pdr = pd // 2  # 1

        k = k.permute(0, 4, 1, 2, 3)  # 1, C, H, W, T
        k = nnf.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))  # 1, C, H+2, W+2, T+2
        k = k.reshape(B, self.num_heads, C // self.num_heads, H+pd,W+pd,T+pd).permute(0, 1, 3, 4, 5, 2) # 1,heads,H+2,W+2,T+2,dims
        attn = modetqkrpb_cu(q,k,self.rpb)
        attn = attn.softmax(dim=-1)  # B h H W T num_tokens
        x = (attn @ self.v)  # B x N x heads x 1 x 3
        x = x.permute(0, 1, 5, 2, 3, 4).reshape(B, -1, H, W, T)

        return x

class ModeT_cu(nn.Module):
    def __init__(self,
                 inshape=(160,192,160),
                 in_channel=1,
                 channels=4,
                 head_dim=6,
                 num_heads=[8, 4, 2, 1, 1],
                 scale=1):
        super(ModeT_cu, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.projblock1 = ProjectionLayer(2*c, dim=head_dim*num_heads[4])
        self.mdt1 = ModeTransformer(head_dim*num_heads[4], num_heads[4], qk_scale=scale)

        self.projblock2 = ProjectionLayer(4*c, dim=head_dim*num_heads[3])
        self.mdt2 = ModeTransformer(head_dim*num_heads[3], num_heads[3], qk_scale=scale)

        self.projblock3 = ProjectionLayer(8*c, dim=head_dim*num_heads[2])
        self.mdt3 = ModeTransformer(head_dim*num_heads[2], num_heads[2], qk_scale=scale)
        self.cwm3 = CWM(3 * num_heads[2], 3 * num_heads[2] * 2)

        self.projblock4 = ProjectionLayer(16*c, dim=head_dim*num_heads[1])
        self.mdt4 = ModeTransformer(head_dim*num_heads[1], num_heads[1], qk_scale=scale)
        self.cwm4 = CWM(3 * num_heads[1], 3 * num_heads[1] * 2)

        self.projblock5 = ProjectionLayer(32*c, dim=head_dim*num_heads[0])
        self.mdt5 = ModeTransformer(head_dim*num_heads[0], num_heads[0], qk_scale=scale)
        self.cwm5 = CWM(3*num_heads[0], 3*num_heads[0]*2)

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))

    def forward(self, moving, fixed):

        # encode stage
        M1, M2, M3, M4, M5 = self.encoder(moving)
        F1, F2, F3, F4, F5 = self.encoder(fixed)

        q5, k5 = self.projblock5(F5), self.projblock5(M5)
        w = self.mdt5(q5, k5)
        w = self.cwm5(w)
        flow = w

        M4 = self.transformer[3](M4, flow)
        q4,k4 = self.projblock4(F4), self.projblock4(M4)
        w=self.mdt4(q4, k4)
        w = self.cwm4(w)
        flow = self.transformer[2](self.upsample_trilin(2*flow), w)+w

        M3 = self.transformer[2](M3, flow)
        q3, k3 = self.projblock3(F3), self.projblock3(M3)
        w = self.mdt3(q3, k3)
        w = self.cwm3(w)
        flow = self.transformer[1](self.upsample_trilin(2 * flow), w) + w

        M2 = self.transformer[1](M2, flow)
        q2,k2 = self.projblock2(F2), self.projblock2(M2)
        w=self.mdt2(q2, k2)
        flow = self.upsample_trilin(2 *(self.transformer[1](flow, w)+w))

        M1 = self.transformer[0](M1, flow)
        q1, k1 = self.projblock1(F1), self.projblock1(M1)
        w=self.mdt1(q1, k1)
        flow = self.transformer[0](flow, w)+w

        y_moved = self.transformer[0](moving, flow)

        return y_moved, flow

if __name__ == '__main__':
    inshape = (1, 1, 80, 96, 80)
    model = ModeT(inshape[2:]).cuda(2)
    A = torch.ones(inshape)
    B = torch.ones(inshape)
    out, flow = model(A.cuda(2), B.cuda(2))
    print(out.shape, flow.shape)
