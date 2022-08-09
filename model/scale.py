import torch
import torch.nn as nn
import torch.nn.functional as F

def KnnDistance(pcloud, nb_neighbors):
        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        ) # [b,n,n]

        distance_matrix[distance_matrix<0] = 0
        dis = torch.sqrt(torch.topk(distance_matrix, k=nb_neighbors, dim=2, largest=False, sorted=True).values) # # [b,n,k]
        # max_dis = dis[:,:,-1]

        nn_dis = torch.mean(dis) 

        return nn_dis 
