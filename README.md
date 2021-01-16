# master_thesis_project
This is a study to compare how much the accuracy of image registration changes with Domain Adaptation.
Three methods are compared: [DANN](https://arxiv.org/abs/1409.7495), [WDGR](https://arxiv.org/abs/1707.01217), and [MCDDA](https://github.com/mil-tokyo/MCD_DA).

# requirement
- [pytorch](https://pytorch.org/) == 1.7.1
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) == 0.1.3
- [albumentations](https://github.com/albumentations-team/albumentations) == 0.5.2
- [warmup-scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr) == 0.3.2


# TODO
- Write a multitask learning code for DA, seg, and geo
- DA training code has many common parts, so it should be unified

