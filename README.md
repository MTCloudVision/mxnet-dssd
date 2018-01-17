#   DSSD MXNET
## Features 

- An MXNet implementation of [DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659).
- MXNET implementation for TDM architecture,[Beyond Skip Connections Top Down Modulation for Object Detection](https://arxiv.org/abs/1612.06851).
- Code implementation in this repository is based on the MXNET implementation of SSD available [here](https://github.com/zhreshold/mxnet-ssd).


## Specific VOC mAP Results 

| backbone      | SSD   | SSD_min_loss | DSSD*     | DSSD_stage1 | DSSD_stage2 | SSD+TDM   |
| ------------- | ----- | ------------ | --------- | ----------- | ----------- | --------- |
| vgg16-512     | 75.56 | 75.35        | **75.85** | 65.16       | 75.77       | **76.80** |
| vgg16-300     | 74.65 | 74.59        | 75.74     | **——**      | **——**      | **——**    |
| resnet101-512 | 78.43 | 78.02        | **79.25** | 71.98       | 78.19       | **79.18** |
| resnet101-321 | 75.18 | 74.80        | 75.54     | **——**      | **——**      | **——**    |

- DSSD*: results by use our traning strategy,for more details,please see [here]#TODO 


## Requirements

We tested our code on:

Ubuntu 16.04, Python 2.7 with

numpy(1.11.0), cv2(3.3.0-dev)

mxnet 0.11.0

## Preparation for Training

1.Download the converted pretrained vgg16_reduced model here.

2.Prepare VOC datasets and generate .rec files by using tools/prepare_pascal.sh

3.Set TDM or DSSD mode in function get_config from symbol/symbol_factory.py.By default, DSSD mode is used,please set all configs by your needs. 

4.start training

```
python train.py
```
or choice a bash file which provide in ./script/ to run some default parameters setting,such like try the two stage training strategy.
```
bash scripts/stage1_dssd_train_res_voc.sh
```


## Demo
1. Download model, available at [TO UPDATE], and place it in the model folder. 
2. run demo.py


## References

1.SSD: Liu W, Anguelov D, Erhan D, Szegedy C, Reed S, Fu CY, Berg AC. SSD: Single shot multibox detector. InEuropean Conference on Computer Vision 2016 Oct 8 (pp. 21-37). Springer International Publishing.[Link](https://arxiv.org/abs/1512.02325)

2.DSSD: Fu CY, Liu W, Ranga A, Tyagi A, Berg AC. DSSD: Deconvolutional Single Shot Detector. arXiv preprint arXiv:1701.06659. 2017 Jan 23. [Link](https://arxiv.org/abs/1701.06659)

3.TDM: Shrivastava A, Sukthankar R, Malik J, Gupta A. Beyond Skip Connections: Top-Down Modulation for ObjectDetection. arXiv preprint arXiv:1612.06851. 2016 Dec 20.[Link](https://arxiv.org/abs/1612.06851)

