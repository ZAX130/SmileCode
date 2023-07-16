# SmileCode
The publicly available code for medical image registration released by the Smile Lab.

## ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer (MICCAI2023)

By Haiqiao Wang, Dong Ni, Yi Wang.

Paper link: [[arXiv]](https://arxiv.org/abs/2306.05688)
### Environment
Code has been tested with Python 3.9 and PyTorch 1.11.
### Citation
If you use the code in your research, please cite:
```
@misc{wang2023modet,
      title={ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer}, 
      author={Haiqiao Wang and Dong Ni and Yi Wang},
      year={2023},
      eprint={2306.05688},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
The overall framework and some network components of the code are heavily based on [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph). We are very grateful for their contributions.
## Unofficial Pytorch Implements

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
