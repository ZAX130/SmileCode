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


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, final=False):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        if final:
            self.upconv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.upconv.weight.shape))
    def forward(self, x):
        x = self.upconv(x)
        x = x[:, :, 1:-1, 1:-1, 1:-1]
        return x

class UpConvLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvLeakyReLU, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.actout = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.upconv(x)
        x = x[:,:,1:-1,1:-1,1:-1]
        return self.actout(x)



class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=2, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv1 = ConvBlock(in_channel, c, kernal_size=3, stride=2) #80 96 80

        self.conv2 = ConvBlock(c, 2*c, kernal_size=3, stride=2) # 40

        self.conv3 = nn.Sequential(
            ConvBlock(2*c, 4*c, kernal_size=3, stride=2),
            ConvBlock(4 * c, 4 * c, kernal_size=3, stride=1)
        ) # 20

        self.conv4 = nn.Sequential(
            ConvBlock(4*c, 8*c, kernal_size=3, stride=2),
            ConvBlock(8 * c, 8 * c, kernal_size=3, stride=1)
        ) #10 12 10

        self.conv5 = nn.Sequential(
            ConvBlock(8*c, 16*c, kernal_size=3, stride=2),
            ConvBlock(16 * c, 16 * c, kernal_size=3, stride=1)
        ) # 5 4

        self.conv6 = nn.Sequential(
            ConvBlock(16*c, 32*c, kernal_size=3, stride=2),
            ConvBlock(32 * c, 32 * c, kernal_size=3, stride=1)
        ) # 2 3 2

    def forward(self, x):
        out1 = self.conv1(x)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/16
        out5 = self.conv5(out4)  # 1/32
        out6 = self.conv6(out5)  # 1/64
        return [out1, out2, out3, out4, out5, out6]




class VTN(nn.Module):
    def __init__(self, inshape=(160,192,160), flow_multiplier=1.,in_channel=2, channels=16, warp=True):
        super(VTN, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.warp = warp

        dims = 3
        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)
        #c, 2c, 4c, 8c, 16c, 32c

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.Pred6 = nn.Conv3d(32*c, dims, 3, 1, 1)
        self.Upsamp6to5 = UpConvBlock(dims, dims, kernel_size=4, stride=2)
        self.Deconv5 = UpConvLeakyReLU(32*c, 16*c, 4, 2)

        self.Pred5 = nn.Conv3d(16*c + 16*c + dims, dims, 3, 1, 1)
        self.Upsamp5to4 = UpConvBlock(dims, dims, 4, 2)
        self.Deconv4 =  UpConvLeakyReLU(16*c + 16*c + dims, 8*c, 4, 2)

        self.Pred4 = nn.Conv3d(8*c + 8*c + dims, dims, 3, 1, 1)
        self.Upsamp4to3 = UpConvBlock(dims, dims, 4, 2)
        self.Deconv3 =  UpConvLeakyReLU(8*c + 8*c + dims, 4*c, 4, 2)

        self.Pred3 = nn.Conv3d(4*c + 4*c + dims, dims, 3, 1, 1)
        self.Upsamp3to2 = UpConvBlock(dims, dims, 4, 2)
        self.Deconv2 =  UpConvLeakyReLU(4*c + 4*c + dims, 2*c, 4, 2)

        self.Pred2 = nn.Conv3d(2*c + 2*c + dims, dims, 3, 1, 1)
        self.Upsamp2to1 = UpConvBlock(dims, dims, 4, 2)
        self.Deconv1 =  UpConvLeakyReLU(2*c + 2*c + dims, c, 4, 2)

        self.Pred0 = UpConvBlock(c + c + dims, dims, 4, 2, final=True)

        if warp:
            self.transformer = SpatialTransformer(inshape)
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(m.weight)
        self.apply(_init_weights)

    def forward(self, moving, fixed):

        concatImgs = torch.cat([moving, fixed], dim=1)
        conv1, conv2, conv3, conv4, conv5, conv6 = self.encoder(concatImgs)

        w = self.Pred6(conv6) #32*c=>3
        w = self.Upsamp6to5(w)
        deconv5 = self.Deconv5(conv6)
        concat5 = torch.cat([conv5, deconv5, w], dim=1)

        w = self.Pred5(concat5)
        w = self.Upsamp5to4(w)
        deconv4 = self.Deconv4(concat5)
        concat4 = torch.cat([conv4, deconv4, w], dim=1)

        w = self.Pred4(concat4)
        w = self.Upsamp4to3(w)
        deconv3 = self.Deconv3(concat4)
        concat3 = torch.cat([conv3, deconv3, w], dim=1)

        w = self.Pred3(concat3)
        w = self.Upsamp3to2(w)
        deconv2 = self.Deconv2(concat3)
        concat2 = torch.cat([conv2, deconv2, w], dim=1)

        w = self.Pred2(concat2)
        w = self.Upsamp2to1(w)
        deconv1 = self.Deconv1(concat2)
        concat1 = torch.cat([conv1, deconv1, w], dim=1)

        flow = self.Pred0(concat1)

        flow = flow*self.flow_multiplier

        if self.warp:
            y_moved = self.transformer(moving, flow)
            return y_moved, flow
        else:
            return flow

class RCN(nn.Module):
    """
    Main model
    """

    def __init__(self, inshape=(160,192,160), flow_multiplier=1.,in_channel=2, channels=16, n_cascade=10):
        super(RCN, self).__init__()
        self.vtn = nn.ModuleList()
        for i in range(n_cascade):
            self.vtn.append(VTN(inshape=inshape, flow_multiplier=flow_multiplier,
                       in_channel=in_channel, channels=channels, warp=False))
        self.n = n_cascade
        self.transformer = SpatialTransformer(inshape)
    def forward(self, moving, fixed):
        flow = 0
        subflows = []
        moved = moving
        for i in range(self.n):
            w = self.vtn[i](moved, fixed)
            subflows.append(w)
            if i == 0:
                flow = w
            else:
                flow = w + self.transformer(flow, w)
            moved = self.transformer(moving, flow)

        return moved, flow, *subflows

class RCN_test(nn.Module):
    """
    Main model
    """

    def __init__(self, inshape=(160,192,160), flow_multiplier=1.,in_channel=2, channels=16, n_cascade=10):
        super(RCN_test, self).__init__()
        self.vtn = nn.ModuleList()
        for i in range(n_cascade):
            self.vtn.append(VTN(inshape=inshape, flow_multiplier=flow_multiplier,
                       in_channel=in_channel, channels=channels, warp=False))
        self.n = n_cascade
        self.transformer = SpatialTransformer(inshape)
    def forward(self, moving, fixed):
        flow = 0
        moved = moving
        for i in range(self.n):
            w = self.vtn[i](moved, fixed)
            if i == 0:
                flow = w
            else:
                flow = w + self.transformer(flow, w)
            moved = self.transformer(moving, flow)

        return moved, flow

if __name__ == '__main__':
#     # model = VoxResNet().cuda()
#     # A = torch.ones((1,1,160,196,160))
#     # B = torch.ones((1,1,160,196,160))
#     # output1 = model(A.cuda())
#     # output2 = model(B.cuda())
#     # for i in range(len(output2)):
#     #     print(torch.sum(output1[i]==output2[i]).item())
#     #     print(output1[i].shape[0]*output1[i].shape[1]*output1[i].shape[2]*output1[i].shape[3]*output1[i].shape[4])
    model = VTN(inshape=(192, 192, 192)).cuda(2)
    # print(str(model))
    A = torch.ones((1, 1, 192, 192, 192))
    B = torch.ones((1, 1, 192, 192, 192))
    out, flow = model(A.cuda(2), B.cuda(2))
    print(out.shape, flow.shape)