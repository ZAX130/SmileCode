import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
from models import VoxelMorph2018, Bilinear
import random
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

same_seeds(24)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def main():

    val_dir = '/LPBA_path/Val/'
    weights = [1, 1]  # loss weights
    lr = 0.0001
    model_idx = -1
    model_folder = 'RTANet_ncc_1_KL_1_lr_{}_onestage/'.format(lr)
    model_dir = 'experiments/' + model_folder

    img_size = (160, 192, 160)
    model = VoxelMorph2018(img_size,simloss='ncc',n=5)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_def, flows = model((x, y))

            for i, flow in enumerate(flows):

                def_out = reg_model(x_seg.cuda().float(), flow.cuda())
                # jac
                jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, ...])
                fold_ratio = np.sum(jac_det <= 0) / np.prod(y.shape[2:])
                eval_det[i].update(fold_ratio, x.size(0))

                # DSC
                dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc_def[i].update(dsc_trans.item(), x.size(0))


            for i in range(5):
                print('Deformed DSC: {:.6f}, det: {}'.format(dsc_trans,jac_det))

            stdy_idx += 1
        for i in range(5):
            print('Deformed DSC: {:.6f} +- {:.6f}'.format(eval_dsc_def[i].avg, eval_dsc_def[i].std))
            print('deformed det: {}, std: {}'.format(eval_det[i].avg, eval_det[i].std))
            # print('deformed time: {}, std: {}'.format(eval_time.avg, eval_time.std))


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()