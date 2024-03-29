# Video-aided Unsupervised Grammar Induction

We investigate video-aided grammar induction, which learns a constituency parser from both unlabeled text and its corresponding video.
We explore rich features (e.g. action, object, scene, audio, face, OCR and speech) from videos, taking the recent Compound PCFG as the baseline. 
We further propose a Multi-Modal Compound PCFG model (MMC-PCFG) to effectively aggregate these rich features from different modalities.

[PDF](https://arxiv.org/pdf/2104.04369.pdf)

<img align="right" width="400" height="400" src="figures/illustration.png">

## News
- :beers: Our follow-up work is accepted by EMNLP 2022. [[PDF]](https://arxiv.org/pdf/2210.12309.pdf) [[Code]](https://github.com/Sy-Zhang/PTC-PCFG)
- :trophy: Our paper wins the **best long paper** award at NAACL 2021.
- :sunny: Our paper is accepted by NAACL 2021.

## Prerequisites
- pytorch 1.5.0
- python 3.7
- easydict
- terminaltables
- tqdm
- numpy
- (A forked version) [Torch-Struct](https://github.com/zhaoyanpeng/pytorch-struct)
- (Optional) [benepar](https://github.com/nikitakit/self-attentive-parser)



## Quick Start

Please download the data from [dropbox](https://www.dropbox.com/sh/9gc6aqzmutg4ief/AADynBuWBPj8sdeacj1bSVxDa?dl=0) or [box](https://rochester.box.com/s/cj52lgoayvotunmqy1awm214jbbelonv), and save it to the `data` folder.
Preprocessing details are described [here](#preprocessing-details).

### Training
Run the following commands for training:
```
sh scripts/didemo/train.sh 
sh scripts/youcook2/train.sh 
sh scripts/msrvtt/train.sh 
```
For VC-PCFG and MMC-PCFG, each run will take approximate 2 hours on DiDeMo, 1 hour on YouCook2 and 10 hours on MSRVTT with a single GTX 1080Ti or GTX 2080Ti.


### Testing
Our trained model are provided in [dropbox](https://www.dropbox.com/sh/ijvjq49i396qnkw/AADUOYpoXbuEHHTwUhA2B5CCa?dl=0) or [box](https://rochester.box.com/s/off57xg1wt47gjzd8ku9kye5pgy6a99w). Please download them to the `final_checkpoints` folder.

Then, run the following commands for evaluation:
```
sh scripts/didemo/test.sh 
sh scripts/youcook2/test.sh 
sh scripts/msrvtt/test.sh 
```

## Preprocessing Details
Download the data from [CVPR 2020: Video Pentathlon challenge](https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/challenge.html) and save it to the `data` folder.
After that, preprocess sentences with the following scripts:
```
python tools/preprocess_captions.py
python tools/compute_gold_trees.py
python tools/generate_vocabularies.py
```  


## Acknowledgements
This repo is developed based on [vpcfg](https://github.com/zhaoyanpeng/vpcfg) and [detr](https://github.com/facebookresearch/detr).

## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@InProceedings{zhang2021video,
author = {Zhang, Songyang and Song, Linfeng and Jin, Lifeng and Xu, Kun and Yu, Dong and Luo, Jiebo},
title = {Video-aided Unsupervised Grammar Induction},
booktitle = {NAACL},
year = {2021}
} 
```
