'''
Inference code for VisTR
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import torch.nn.functional as F
import json
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as mask_util



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the model weights.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument('--num_ins', default=10, type=int,
                        help="Number of instances")
    parser.add_argument('--num_queries', default=360, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--img_path', default='data/ytvos/valid/JPEGImages/')
    parser.add_argument('--ann_path', default='data/ytvos/annotations/instances_val_sub.json')
    parser.add_argument('--save_path', default='results.json')
    parser.add_argument('--dataset_file', default='ytvos')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    #parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

CLASSES=['person','giant_panda','lizard','parrot','skateboard','sedan','ape',
         'dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
         'train','horse','turtle','bear','motorbike','giraffe','leopard',
         'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
         'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
         'tennis_racket']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
          [0.700, 0.300, 0.600]]
transform = T.Compose([
    T.Resize(300),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b



def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    num_frames = args.num_frames
    num_ins = args.num_ins
    with torch.no_grad():
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        state_dict = torch.load(args.model_path)['model']
        model.load_state_dict(state_dict)
        folder = args.img_path
        videos = json.load(open(args.ann_path,'rb'))['videos']
        vis_num = len(videos)
        result = [] 
        for i in range(vis_num):
            print("Process video: ",i)
            id_ = videos[i]['id']
            length = videos[i]['length']
            file_names = videos[i]['file_names']
            clip_num = math.ceil(length/num_frames)
            
            img_set=[]
            if length<num_frames:
                clip_names = file_names*(math.ceil(num_frames/length))
                clip_names = clip_names[:num_frames]
            else:
                clip_names = file_names[:num_frames]
            if len(clip_names)==0:
                continue
            if len(clip_names)<num_frames:
                clip_names.extend(file_names[:num_frames-len(clip_names)])
            for k in range(num_frames):
                im = Image.open(os.path.join(folder,clip_names[k]))
                img_set.append(transform(im).unsqueeze(0).cuda())
            img=torch.cat(img_set,0)
            # inference time is calculated for this operation
            outputs = model(img)
            # end of model inference
            logits, boxes, masks = outputs['pred_logits'].softmax(-1)[0,:,:-1], outputs['pred_boxes'][0], outputs['pred_masks'][0]
            pred_masks =F.interpolate(masks.reshape(num_frames,num_ins,masks.shape[-2],masks.shape[-1]),(im.size[1],im.size[0]),mode="bilinear").sigmoid().cpu().detach().numpy()>0.5
            pred_logits = logits.reshape(num_frames,num_ins,logits.shape[-1]).cpu().detach().numpy()
            pred_masks = pred_masks[:length] 
            pred_logits = pred_logits[:length]
            pred_scores = np.max(pred_logits,axis=-1)
            pred_logits = np.argmax(pred_logits,axis=-1)
            for m in range(num_ins):
                if pred_masks[:,m].max()==0:
                    continue
                score = pred_scores[:,m].mean()
                #category_id = pred_logits[:,m][pred_scores[:,m].argmax()]
                category_id = np.argmax(np.bincount(pred_logits[:,m]))
                instance = {'video_id':id_, 'score':float(score), 'category_id':int(category_id)}
                segmentation = []
                for n in range(length):
                    if pred_scores[n,m]<0.001:
                        segmentation.append(None)
                    else:
                        mask = (pred_masks[n,m]).astype(np.uint8) 
                        rle = mask_util.encode(np.array(mask[:,:,np.newaxis], order='F'))[0]
                        rle["counts"] = rle["counts"].decode("utf-8")
                        segmentation.append(rle)
                instance['segmentations'] = segmentation
                result.append(instance)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(result,f)
                    
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
