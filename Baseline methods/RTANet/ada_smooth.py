import torch
from torch import nn


class GaussianSmoother(nn.Module):
    def __init__(self):
        super(GaussianSmoother, self).__init__()

    def forward(self, field, log):

        # field = field.view([-1, *field.shape[2:]])
        # log = log.view([-1, *log.shape[2:]])
        return voxel_wise_gaussian_smooth(field,size=3,log=log)


def gaussian_kernel(size, mean, std):
    d = torch.distributions.normal.Normal(mean, std)

    vals = d.log_prob(torch.arange(-(size//2), size//2 + 1).cuda()).exp()
    gauss_kernel = torch.einsum('i,j,k->ijk', vals, vals, vals)
    gauss_kernel = gauss_kernel / gauss_kernel.sum()
    gauss_kernel = gauss_kernel[None,None,...]

    return torch.repeat_interleave(gauss_kernel,3,0)


def voxel_wise_gaussian_smooth(image, size=7, mean=0., std=3., log=None):


    if log is not None:
        # image = torch.nn.ConstantPad3d((1, 1, 1, 1, 1, 1),0)(image)
        image = torch.nn.functional.pad(image, (1, 1, 1, 1, 1, 1))
        masks, stds = generate_mask(log)
        kernels = [gaussian_kernel(size, mean, s) for s in stds]
        smooth_images = [torch.nn.functional.conv3d(image, k, groups=3) for k in kernels]
        smoothed_image = 0
        for m, s in zip(masks, smooth_images):
            smoothed_image += m*s


    else:
        gauss_kernel = gaussian_kernel(size, mean, std)
        smoothed_image = torch.nn.functional.conv3d(image, gauss_kernel, groups=3)

    return smoothed_image


def generate_mask(log):
    log = torch.exp(log / 2.0)
    # choose the maximum of uncertainty vector
    log_norm = torch.max(log, dim=1, keepdim=True)[0]

    # For simplification, thress thresholds are generated.
    std_1 = (2. * torch.min(log_norm) + torch.max(log_norm)) / 3.
    std_2 = (torch.min(log_norm) + 2. * torch.max(log_norm)) / 3.
    std_3 = torch.max(log_norm)

    # generate the masks
    mask_1 = (log_norm <= std_1).float()
    mask_2 = (log_norm > std_1).float()
    mask_i = (log_norm <= std_2).float()
    mask_2 = mask_2 * mask_i
    mask_3 = (log_norm > std_2).float()

    return [mask_1, mask_2, mask_3], [std_1, std_2, std_3]

if __name__ == '__main__':
    flow = torch.randn((1,3,80,80,80))
    log = torch.randn((1, 3, 80, 80, 80))
    model = GaussianSmoother()
    out = model(flow,log)
    for f in out:
        print(f.shape)
