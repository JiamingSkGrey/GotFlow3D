import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from model.flot.gconv import GeoSetConv
from model.flot.graph import ParGraph

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      


class GeoDGCNN_flow2(nn.Module):
    def __init__(self, k, emb_dims, dropout):
        super(GeoDGCNN_flow2, self).__init__()
        # self.args = args
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(96)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        
        self.conv1 = nn.Sequential(nn.Conv2d(32*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 96, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(352, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(1376, 512, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv8 = nn.Conv1d(256, 128, kernel_size=1, bias=False)

        self.feat_conv1 = GeoSetConv(3, 32)
        self.feat_conv2 = GeoSetConv(32, 64)
        self.feat_conv3 = GeoSetConv(64, 96)
        

    def forward(self, x):

        geo_graph = ParGraph.construct_graph(x, self.k)
        g1 = self.feat_conv1(x, geo_graph)     # B x nb_feat_out x N
        g2 = self.feat_conv2(g1, geo_graph)
        g3 = self.feat_conv3(g2, geo_graph)
        g1 = g1.transpose(1, 2).contiguous() 
        g2 = g2.transpose(1, 2).contiguous() 
        g3 = g3.transpose(1, 2).contiguous() 

        x = get_graph_feature(g1, k=self.k)     
        x = self.conv1(x)                       
        x = self.conv2(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]    

        x = get_graph_feature(x2, k=self.k)     
        x = self.conv3(x)                       
        x = self.conv4(x)                       
        x3 = x.max(dim=-1, keepdim=False)[0]    

        mid = torch.cat((g1, x2, x3, g2, g3), dim=1)      

        x = self.conv5(mid)                       

        x = torch.cat((x, mid), dim=1)   

        x = self.conv6(x)                       
        x = self.conv7(x)                       
        x = self.dp1(x)
        x = self.conv8(x)                       
        
        return x