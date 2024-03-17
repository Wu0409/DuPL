# DuPL
This repository contains the source code of CVPR 2024 paper: DuPL: Dual Student with Trustworthy Progressive Learning for Robust Weakly Supervised Semantic Segmentation

<img src="paper/overview.jpg"  width="100%"/>

### TODO
- [ ] Release the pre-trained checkpoints and segmentation results.
- [ ] Add the visualization scripts for CAM and segmentation results.
- [ ] Update the citations.

### Update Log
- **Mar. 17, 2024**: Basic training code released.

## Get Started

### Training Environment
The implementation is based on PyTorch 1.13.1 with single-node multi-gpu training. Please install the required packages by running:
```bash
pip install -r requirements.txt
```

### Datasets
<details>
<summary>
VOC dataset
</summary>

#### 1. Download from official website

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar –xvf VOCtrainval_11-May-2012.tar
```
#### 2. Download the augmented annotations
The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). Here is a download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `VOCdevkit/VOC2012`. The directory should be: 

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```
</details>

<details>

<summary>
COCO dataset
</summary>

#### 1. Download
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```
#### 2. Generating VOC style segmentation labels for COCO
The training pipeline use the VOC-style segmentation labels for COCO dataset, please download the converted masks from [One Drive](https://1drv.ms/u/s!As-yzQ0hGhUXiMZUJiXOEn2pkWP80g?e=1HaHAd).
The directory should be:
``` bash
MSCOCO/
├── coco2014
│    ├── train2014
│    └── val2014
└── SegmentationClass (extract the downloaded "coco_mask.tar.gz")
     ├── train2014
     └── val2014
```
NOTE: You can also use the scripts provided at this [repo](https://github.com/alicranck/coco2voc) to covert the COCO segmentation masks.

</details>

## Experiments
### Training DuPL
To train the segmentation model for the VOC dataset, please run
```bash
# For Pascal VOC 2012
python -m torch.distributed.run --nproc_per_node=2 train_final_voc.py --data_folder [../VOC2012]

# For MSCOCO
python -m torch.distributed.run --nproc_per_node=4 train_final_coco.py --data_folder [../MSCOCO/coco2014]
```
### Evaluation
To evaluate the trained model, please run
```bash
# For Pascal VOC 2012
python eval_seg_voc.py --data_folder [../VOC2012] --model_path [path_to_model]

# For MSCOCO
python -m torch.distributed.launch --nproc_per_node=4 eval_seg_coco_ddp.py --data_folder [../MSCOCO/coco2014] --label_folder [../MSCOCO/SegmentationClass] --model_path [path_to_model]
```

Convert rgb segmentation labels for the official VOC evaluation:
```bash
# modify the "dir" and "target_dir" before running
python convert_voc_rgb.py
```

NOTE: 
* The segmentation results will be saved at the checkpoint directory. You can visualize the results if needed.
* The evaluation of MSCOCO use DDP to accelerate the evaluation stage. Please make sure the `torch.distributed.launch` is available in your environment.
* We highly recommend use high-performance CPU for CRF post-processing, which is time-consuming. On `MS COCO`, it may cost several hours for CRF post-processing.

## Results
We have provided DuPL's pre-trained checkpoints on VOC and COCO datasets. With these checkpoints, it should be expected to reproduce the exact performance listed below.

|Dataset|Backbone| *val* |              Log              |Weights| *val* (with MS+CRF) | *test* (with MS+CRF) |
|:---:|:---:|:-----:|:-----------------------------:|:---:|:-------------------:|:--------------------:|
|VOC|DeiT-B| 69.9  | [log](./logs/dupl_train_voc)  |[weights]()|        72.2         |         71.6         |
|VOC|ViT-B|  --   |            [log]()            |[weights]()|        73.3         |         72.8         |
|COCO|DeiT-B|  --   | [log](./logs/dupl_train_coco) |[weights]()|        43.5         |          --          |
|COCO|ViT-B|  --   |            [log]()            |[weights]()|        44.6         |          --          |

**The checkpoints and segmentation results will be released soon.** The VOC test results are evaluated on the official server, and the result links are provided in the paper.

## Citation
Please kindly cite our paper if you find it's helpful in your work:
```bibtex
% To be updated
```


## Acknowledgement
We would like to thank all the researchers who open source their works to make this project possible, especially thanks to the authors of [Toco](https://github.com/rulixiang/ToCo/tree/main) for their brilliant work.