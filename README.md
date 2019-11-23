# Learning to Assemble Neural Module Tree Networks for Visual Grounding

This repository contains the code for the following paper:
- Daqing Liu, Hanwang Zhang, Feng Wu, Zheng-Jun Zha, *Learning to Assemble Neural Module Tree Networks for Visual Grounding*. in ICCV, 2019. ([PDF](https://arxiv.org/pdf/1812.03299))

## Installation
1. Install Python 3 ([Anaconda](https://www.anaconda.com/distribution/) recommended)
2. Install [Pytorch](https://pytorch.org/) v1.0 or higher:
``` sh
pip3 install torch torchvision
```
3. Clone with Git, and then enter the root directory:
``` sh
git clone --recursive https://github.com/daqingliu/NMTree.git && cd NMTree
```
4. Prepare data
    - Follow `data/README.md` to prepare images and refcoco/refcoco+/refcocog annotations. Or simply run:
    ``` sh
    # it will cost some time accordding to your network
    bash data/prepare_data.sh
    ```
    - Our visual features are extracted by [MAttNet](https://github.com/daqingliu/MAttNet), please follow the instruction. Or just download and uncompress [Refcocog visual features](https://drive.google.com/file/d/14JM7XNJKvdDPGzRQ1w9Ru-8-1dW-54Gt/view?usp=sharing) into `data/feats/refcocog_umd` for testing this repo.
    - Preprocess vocabulary:
    ``` sh
    python misc/parser.py --dataset refcocog --split_by umd
    ```

## Training
``` sh
python tools/train.py \
    --id det_nmtree_01 \
    --dataset refcocog \
    --split_by umd \
    --grounding_model NMTree \
    --data_file data_dep \
    --batch_size 128 \
    --glove glove.840B.300d_dep \
    --visual_feat_file matt_res_gt_feats.pth
```

## Evaluation
``` sh
python tools/eval_gt.py \
    --log_path log/refcocog_umd_nmtree_01 \
    --dataset refcocog \
    --split_by umd \

python tools/eval_det.py \
    --log_path log/refcocog_umd_nmtree_01 \
    --dataset refcocog \
    --split_by umd
```

## Citation
```
@inproceedings{liu2019learning,
title={Learning to Assemble Neural Module Tree Networks for Visual Grounding},
author={Liu, Daqing and Zhang, Hanwang and Zha, Zheng-Jun and Feng, Wu},
booktitle={The IEEE International Conference on Computer Vision (ICCV)},
year={2019}
}
```

## Acknowledgments
Some codes come from [Refer](https://github.com/lichengunc/refer), [MattNet](https://github.com/lichengunc/MAttNet), and [gumbel-softmax](https://github.com/ericjang/gumbel-softmax).

This project is maintained by [Liu Daqing](https://github.com/daqingliu). Welcome issues and PRs.
