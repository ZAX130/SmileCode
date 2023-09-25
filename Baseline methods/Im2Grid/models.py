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

class PositionalEncodingLayer(nn.Module):
    def __init__(self, in_channels, dim=6):
        super().__init__()
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(torch.zeros(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))
        channels = int(np.ceil(dim / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.proj(feat)

        batch_size, x, y, z, orig_ch = feat.shape
        pos_x = torch.arange(x, device=feat.device).type(torch.FloatTensor)
        pos_y = torch.arange(y, device=feat.device).type(torch.FloatTensor)
        pos_z = torch.arange(z, device=feat.device).type(torch.FloatTensor)
        inv_freq_x = torch.pi / (x - 1)
        inv_freq_y = torch.pi / (y - 1)
        inv_freq_z = torch.pi / (z - 1)
        sin_inp_x = (pos_x*inv_freq_x).unsqueeze(-1)
        sin_inp_y = (pos_y*inv_freq_y).unsqueeze(-1)
        sin_inp_z = (pos_z*inv_freq_z).unsqueeze(-1)
        emb_x = torch.cat((sin_inp_x.cos(), sin_inp_x.sin()), dim=-1).unsqueeze(1).unsqueeze(1)  # 160, 1, 1, 2
        emb_y = torch.cat((sin_inp_y.cos(), sin_inp_y.sin()), dim=-1).unsqueeze(1)  # 192, 1, 2
        emb_z = torch.cat((sin_inp_z.cos(), sin_inp_z.sin()), dim=-1)  # 224， 2
        emb = torch.zeros((x,y,z,self.channels*3),device=feat.device).type(feat.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        feat = feat + self.alpha * emb

        return feat

class CoTr(nn.Module):
    def __init__(self, kernel_size=3, norm=nn.LayerNorm):
        super().__init__()



        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        # self.rpb_size = 2 * kernel_size - 1

        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [kernel_size] * 3]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def makeV(self, N):
        # v.shape: (1, N, self.kernel_size**3, 3)
        v = self.grid.reshape(self.kernel_size**3, 3).unsqueeze(0).repeat(N, 1, 1).unsqueeze(0)
        return v


    def forward(self, q, k):

        B, H, W, T, C = q.shape
        N = H * W * T
        num_tokens = int(self.kernel_size ** 3)

        q = q.reshape(B, N, C, 1).transpose(2, 3)   # 1, N, 1, dim
        pd = self.kernel_size - 1  # 2
        pdr = pd // 2  # 1

        k = k.permute(0, 4, 1, 2, 3)  # C, H, W, T
        k = nnf.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))  # 1, C, H+2, W+2, T+2
        k = k.flatten(0, 1)  # C, H+2, W+2, T+2
        k = k.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).permute(0, 4, 5, 6, 1, 2, 3)  # C, 3, 3, 3, H, W, T
        k = k.reshape(B, C, num_tokens, N)  # 这步突然炸显存
        k = k.permute(0, 3, 2, 1)  # (B, N, num_tokens, dim)

        attn = (q @ k.transpose(-2, -1))  # B x N x H x 1 x num_tokens
        attn = attn.softmax(dim=-1)

        v = self.makeV(N)
        x = (attn @ v)  # B x N x H x 1 x C
        x = x.reshape(B, H, W, T, 3).permute(0, 4, 1, 2, 3)

        return x


class Im2grid(nn.Module):
    def __init__(self, inshape=(160,192,160), flow_multiplier=1.,in_channel=1, channels=4):
        super(Im2grid, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.inshape = inshape

        dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.peblock = nn.ModuleList()

        self.peblock1 = PositionalEncodingLayer(2*c)
        self.peblock2 = PositionalEncodingLayer(4*c)
        self.peblock3 = PositionalEncodingLayer(8*c)
        self.peblock4 = PositionalEncodingLayer(16*c)
        self.peblock5 = PositionalEncodingLayer(32*c)
        self.cotr = CoTr()

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))

    def forward(self, moving, fixed):

        # encode stage
        M1, M2, M3, M4, M5 = self.encoder(moving)
        F1, F2, F3, F4, F5 = self.encoder(fixed)

        fields = []

        q5, k5 = self.peblock5(F5), self.peblock5(M5)
        w = self.cotr(q5, k5)
        flow = self.upsample_trilin(2*w)

        M4 = self.transformer[3](M4, flow)
        q4,k4 = self.peblock4(F4), self.peblock4(M4)
        w=self.cotr(q4, k4)
        flow = self.upsample_trilin(2 *(self.transformer[3](flow, w)+w))

        M3 = self.transformer[2](M3, flow)
        q3, k3 = self.peblock3(F3), self.peblock3(M3)
        w = self.cotr(q3, k3)
        flow = self.upsample_trilin(2 * (self.transformer[2](flow, w) + w))

        M2 = self.transformer[1](M2, flow)
        q2,k2 = self.peblock2(F2), self.peblock2(M2)
        w=self.cotr(q2, k2)
        flow = self.upsample_trilin(2 *(self.transformer[1](flow, w)+w))

        M1 = self.transformer[0](M1, flow)
        q1, k1 = self.peblock1(F1), self.peblock1(M1)
        w=self.cotr(q1, k1)
        flow = self.transformer[0](flow, w)+w

        y_moved = self.transformer[0](moving, flow)

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
    inshape = (1, 1, 80, 96, 80)
    model = Im2grid(inshape[2:]).cuda(2)
    # print(str(model))
    A = torch.ones(inshape)
    B = torch.ones(inshape)
    out, flow = model(A.cuda(2), B.cuda(2))
    print(out.shape, flow.shape)