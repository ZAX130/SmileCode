# SmileCode
The publicly available code for ModeT and the other medical image registration codes released by the Smile Lab.

## ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer (MICCAI2023)

By Haiqiao Wang, Dong Ni, Yi Wang.

Paper link: [[MICCAI]](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_70)

![image](https://github.com/ZAX130/SmileCode/assets/43944700/22f1ff8b-c7ca-4d37-a682-147207ee006e)


### Environment
Code has been tested with Python 3.9 and PyTorch 1.11.
### Dataset
LPBA [[link]](https://resource.loni.usc.edu/resources/atlases-downloads/)
Mindboggle [[link]](https://osf.io/yhkde/)

### Instruction
For convenience, we are sharing the preprocessed [LPBA](https://drive.usercontent.google.com/download?id=1mFzZDn2qPAiP1ByGZ7EbsvEmm6vrS5WO&export=download&authuser=0) dataset used in our experiments. Once uncompressed, simply modify the "LPBA_path" in `train.py` to the path name of the extracted data. Next, you can execute `train.py` to train the network, and after training, you can run `infer.py` to test the network performance.

(Update) We encourage you to try the ModeTv2 code, as it enhances registration accuracy while significantly reducing both runtime and memory usage.
### Citation
If you use the code in your research, please cite:
```
@InProceedings{10.1007/978-3-031-43999-5_70,
author="Wang, Haiqiao and Ni, Dongand Wang, Yi",
title="ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
pages="740--749",
}
```
The overall framework and some network components of the code are heavily based on [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph). We are very grateful for their contributions. The file makePklDataset.py shows how to make a pkl dataset from the original LPBA dataset. If you have any other questions about the .pkl format, please refer to the github page of [[TransMorph_on_IXI]](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md). 
## Other Works
### ModeTv2: GPU-accelerated Motion Decomposition Transformer for Pairwise Optimization in Medical Image Registration

By Haiqiao Wang, Zhuoyuan Wang, Dong Ni, Yi Wang.

Paper link: [[arxiv]](https://arxiv.org/abs/2403.16526), Code link: [[code]](https://github.com/ZAX130/ModeTv2)
### Recursive Deformable Pyramid Network for Unsupervised Medical Image Registration (TMI2024)

By Haiqiao Wang, Dong Ni, Yi Wang.

Paper link: [[TMI]](https://ieeexplore.ieee.org/document/10423043),   Code link: [[code]](https://github.com/ZAX130/RDP)

## Unofficial Pytorch implementations (Baseline Methods)

- [x]  Recursive Cascaded Networks for Unsupervised Medical Image Registration (RCN)

    links: [[original code]](https://github.com/microsoft/Recursive-Cascaded-Networks)  [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhao_Recursive_Cascaded_Networks_for_Unsupervised_Medical_Image_Registration_ICCV_2019_paper.html)  [[code]](https://github.com/ZAX130/SmileCode/tree/main/Baselines%20methods/RCN)
- [x]  Recursive Decomposition Network for Deformable Image Registration (RDN)

    links: [[paper]](https://ieeexplore.ieee.org/abstract/document/9826364)  [[code]](https://github.com/ZAX130/SmileCode/tree/main/Baselines%20methods/RDN)
- [x]  Joint Progressive and Coarse-to-Fine Registration of Brain MRI via Deformation Field Integration and Non-Rigid Feature Fusion (PCnet)

    links: [[original code]](https://github.com/JinxLv/Progressvie-and-Coarse-to-fine-Registration-Network)  [[paper]](https://ieeexplore.ieee.org/abstract/document/9765391)  [[code]](https://github.com/ZAX130/SmileCode/tree/main/Baselines%20methods/PCnet)
- [x]  Dual-stream pyramid registration network (PR++)

    links: [[paper]](https://www.sciencedirect.com/science/article/pii/S1361841522000317)  [[code]](https://github.com/ZAX130/SmileCode/tree/main/Baselines%20methods/PR%2B%2B)
- [x]  Coordinate Translator for Learning Deformable Medical Image Registration (Im2Grid)

    links: [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-18814-5_10)  [[code]](https://github.com/ZAX130/SmileCode/tree/main/Baselines%20methods/Im2Grid)
## How can other datasets be used in this code?
This is a common question, and please refer to the github page of [ChangeDataset.md](https://github.com/ZAX130/ModeTv2/blob/main/ChangeDataset.md) for more information.
