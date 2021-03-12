"""
Instance Sequence Matching
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, multi_iou
INF = 100000000

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_frames : int = 36, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the sequence level matching
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        for i in range(bs):
            out_prob = outputs["pred_logits"][i].softmax(-1)
            out_bbox = outputs["pred_boxes"][i]
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]
            tgt_valid = targets[i]["valid"]
            num_out = 10
            num_tgt = len(tgt_ids)//self.num_frames
            out_prob_split = out_prob.reshape(self.num_frames,num_out,out_prob.shape[-1]).permute(1,0,2)
            out_bbox_split = out_bbox.reshape(self.num_frames,num_out,out_bbox.shape[-1]).permute(1,0,2).unsqueeze(1)
            tgt_bbox_split = tgt_bbox.reshape(num_tgt,self.num_frames,4).unsqueeze(0)
            tgt_valid_split = tgt_valid.reshape(num_tgt,self.num_frames)
            frame_index = torch.arange(start=0,end=self.num_frames).repeat(num_tgt).long()
            class_cost = -1 * out_prob_split[:,frame_index,tgt_ids].view(num_out,num_tgt,self.num_frames).mean(dim=-1)
            bbox_cost = (out_bbox_split-tgt_bbox_split).abs().mean((-1,-2))
            iou_cost = -1 * multi_iou(box_cxcywh_to_xyxy(out_bbox_split),box_cxcywh_to_xyxy(tgt_bbox_split)).mean(-1)
            #TODO: only deal with box and mask with empty target
            cost = self.cost_class*class_cost + self.cost_bbox*bbox_cost + self.cost_giou*iou_cost
            out_i, tgt_i = linear_sum_assignment(cost.cpu())
            index_i,index_j = [],[]
            for j in range(len(out_i)):
                tgt_valid_ind_j = tgt_valid_split[j].nonzero().flatten()
                index_i.append(tgt_valid_ind_j*num_out + out_i[j])
                index_j.append(tgt_valid_ind_j + tgt_i[j]* self.num_frames)
            if index_i==[] or index_j==[]:
                indices.append((torch.tensor([]).long().to(out_prob.device),torch.tensor([]).long().to(out_prob.device)))
            else:
                index_i = torch.cat(index_i).long()
                index_j = torch.cat(index_j).long()
                indices.append((index_i,index_j))
        return indices

def build_matcher(args):
    return HungarianMatcher(num_frames = args.num_frames, cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
