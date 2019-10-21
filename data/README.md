## Data Download

- **MS-COCO**: Please download *train2014.zip (2014 Train images \[83k/13GB\])* in [this website](http://cocodataset.org/#download) and extract into "data/images" folder. It contains all RefCOCO, RefCOCO+, RefCOCOg images.
- **Refer Datasets**: Please download the cleaned data and extract them into "data/datasets" folder: [refcoco](http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip), [refcoco+](http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), and [refcocog](http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip).
- **Glove Word Vectors**: Please downlaod *[glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip)* and extract it into "data/" folder.

## File Tree
Please restructure above files as below:
```
data
├── images
│   └── train2014
├── datasets
│   ├── refcoco
│   │   ├── instances.json
│   │   ├── refs(google).p
│   │   └── refs(unc).p
│   ├── refcoco+
│   │   ├── instances.json
│   │   └── refs(unc).p
│   └── refcocog
│       ├── instances.json
│       └── refs(google).p
└── feats
    ├── refcoco_unc
    │   ├── matt_res_gt_feats.pth
    │   ├── matt_res_det_feats.pth
    │   ├── matt_vgg_gt_feats.pth
    │   └── matt_vgg_det_feats.pth
    ├── refcoco+_unc
    │   ├── matt_res_gt_feats.pth
    │   ├── matt_res_det_feats.pth
    │   ├── matt_vgg_gt_feats.pth
    │   └── matt_vgg_det_feats.pth
    ├── refcocog_umd
    │   ├── matt_res_gt_feats.pth
    │   ├── matt_res_det_feats.pth
    │   ├── matt_vgg_gt_feats.pth
    │   └── matt_vgg_det_feats.pth
    └── glove.840B.300d.txt
```
