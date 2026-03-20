import os
import sys
import numpy as np
import random
import cv2
import torch
from torch.nn import functional as F
def generate_hw_gt( target, class_num = 20 ):
    h,w = target.shape   
    target = torch.from_numpy(target)
    target_c = target.clone()
    target_c[target_c==255]=0    
    target_c = target_c.long()
    target_c = target_c.view(h*w)
    target_c = target_c.unsqueeze(1)
    target_onehot = torch.zeros(h*w,class_num)
    target_onehot.scatter_( 1, target_c, 1 )      #h*w,class_num
    target_onehot = target_onehot.transpose(0,1)
    target_onehot = target_onehot.view(class_num,h,w)
    # h distribution ground truth
    hgt = torch.zeros((class_num,h))
    hgt=( torch.sum( target_onehot, dim=2 ) ).float()
    hgt[0,:] = 0  
    max = torch.max(hgt,dim=1)[0]         #c,1
    min = torch.min(hgt,dim=1)[0]
    max = max.unsqueeze(1)  
    min = min.unsqueeze(1) 
    hgt = hgt / ( max + 1e-5 )   
    # w distribution gound truth
    wgt = torch.zeros((class_num,w))
    wgt=( torch.sum(target_onehot, dim=1 ) ).float()
    wgt[0,:]=0
    max = torch.max(wgt,dim=1)[0]         #c,1
    min = torch.min(wgt,dim=1)[0]
    max = max.unsqueeze(1)    
    min = min.unsqueeze(1)
    wgt = wgt / ( max + 1e-5 )
    #===========================================================
    hwgt = torch.matmul( hgt.transpose(0,1), wgt )
    max = torch.max( hwgt.view(-1), dim=0 )[0]
    # print(max)
    hwgt = hwgt / ( max + 1.0e-5 )
    #====================================================================
    return hgt, wgt, hwgt #,cch, ccw  gt_hw

def generate_edge(label, edge_width=3):
    label = label.type(torch.cuda.FloatTensor)
    if len(label.shape) == 2:
        label = label.unsqueeze(0)
    n, h, w = label.shape
    edge = torch.zeros(label.shape, dtype=torch.float).cuda()
    # right
    edge_right = edge[:, 1:h, :]
    edge_right[(label[:, 1:h, :] != label[:, :h - 1, :]) & (label[:, 1:h, :] != 255)
               & (label[:, :h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :, :w - 1]
    edge_up[(label[:, :, :w - 1] != label[:, :, 1:w])
            & (label[:, :, :w - 1] != 255)
            & (label[:, :, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:, :h - 1, :w - 1]
    edge_upright[(label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])
                 & (label[:, :h - 1, :w - 1] != 255)
                 & (label[:, 1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:, :h - 1, 1:w]
    edge_bottomright[(label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])
                     & (label[:, :h - 1, 1:w] != 255)
                     & (label[:, 1:h, :w - 1] != 255)] = 1

    kernel = torch.ones((1, 1, edge_width, edge_width), dtype=torch.float).cuda()
    with torch.no_grad():
        edge = edge.unsqueeze(1)
        edge = F.conv2d(edge, kernel, stride=1, padding=1)
    edge[edge!=0] = 1
    edge = edge.squeeze()
    return edge