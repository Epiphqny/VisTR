## VisTR: End-to-End Video Instance Segmentation with Transformers

This is the official implementation of the [VisTR paper](https://arxiv.org/abs/2011.14503):

<p align="center">
<img src="https://user-images.githubusercontent.com/16319629/110786946-b99aa080-82a7-11eb-98e4-85478ca4eeac.png" width="600">
</p>


### Installation
We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/Epiphqny/vistr.git
```
Then, install PyTorch 1.6 and torchvision 0.7:
```
conda install pytorch==1.6.0 torchvision==0.7.0
```
Install pycocotools
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```
Compile DCN module(requires GCC>=5.3, cuda>=10.0)
```
cd models/dcn
python setup.py build_ext --inplace
```

### Preparation

Download and extract 2019 version of YoutubeVIS  train and val images with annotations from
[CodeLab](https://competitions.codalab.org/competitions/20128#participate-get_data) or [YoutubeVIS](https://youtube-vos.org/dataset/vis/).
We expect the directory structure to be the following:
```
VisTR
├── data
│   ├── train
│   ├── val
│   ├── annotations
│   │   ├── instances_train_sub.json
│   │   ├── instances_val_sub.json
├── models
...
```

Download the pretrained [DETR models](https://drive.google.com/drive/folders/1DlN8uWHT2WaKruarGW2_XChhpZeI9MFG?usp=sharing) on COCO and save it to the pretrained path.


### Training

Training of the model requires at least 32g memory GPU, we performed the experiment on 32g V100 card.

To train baseline VisTR on a single node with 8 gpus for 16 epochs, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --backbone resnet101/50 --ytvos_path /path/to/ytvos --masks --pretrained_weights /path/to/pretrained_path
```

### Inference

```
python inference.py --masks --model_path /path/to/model_weights --save_path /path/to/results.json
```

### Models

We provide baseline VisTR models, and plan to include more in future. AP is computed on YouTubeVIS dataset by submitting the result json file to the [CodeLab](https://competitions.codalab.org/competitions/20128#results) system, and inference time is calculated by pure model inference time (without data-loading and post-processing).

   <table>
     <thead>
       <tr style="text-align: right;">
         <th></th>
         <th>name</th>
         <th>backbone</th>
         <th>FPS</th>
         <th>mask AP</th>
         <th>model</th>
         <th>md5</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <th>0</th>
         <td>VisTR</td>
         <td>R50</td>
         <td>69.9</td>
         <td>34.4</td>
         <td><a href="">vistr_r50(Please wait)</a></td>
         <td></td>
       </tr>
       <tr>
         <th>1</th>
         <td>VisTR</td>
         <td>R101</td>
         <td>57.7</td>
         <td>36.5</td>
         <td><a href="https://drive.google.com/file/d/1PrigF8oJW1TSNDMV9cNDfHc_rP1Oi5Si/view?usp=sharing">vistr_r101</a></td>
         <td>2b8d412225121fb1694427ab69a40656</td>
       </tr>
   </table>


### License

VisTR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

### Acknowledgement
We would like to thank the [DETR](https://github.com/facebookresearch/detr) open-source project for its awesome work, part of the code are modified from its project.

### Citation

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@article{wang2020end,
  title={End-to-End Video Instance Segmentation with Transformers},
  author={Wang, Yuqing and Xu, Zhaoliang and Wang, Xinlong and Shen, Chunhua and Cheng, Baoshan and Shen, Hao and Xia, Huaxia},
  journal={arXiv preprint arXiv:2011.14503},
  year={2020}
}
```

