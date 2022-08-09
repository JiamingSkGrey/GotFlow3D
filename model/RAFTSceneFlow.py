import torch
import torch.nn as nn

from model.extractor import FlotEncoder, FlotGraph
from model.corr2 import CorrBlock2
from model.update import UpdateBlock
from model.scale import KnnDistance
import model.ot as ot
from model.model_dgcnn import GeoDGCNN_flow2


class RSF_DGCNN(nn.Module):
    def __init__(self, args):
        super(RSF_DGCNN, self).__init__()
        self.hidden_dim = 64
        self.context_dim = 64
        self.feature_extractor = GeoDGCNN_flow2(k=32, emb_dims=1024, dropout=0.5)
        self.context_extractor = FlotEncoder()
        # self.graph_extractor = FlotGraph()
        self.corr_block = CorrBlock2(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k)
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)

        self.scale_offset = nn.Parameter(torch.ones(1)/2.0) # torch.ones(1)/10.0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, p, num_iters=12):
        # feature extraction
        [xyz1, xyz2] = p # B x N x 3
 
        fmap1 = self.feature_extractor(p[0])
        fmap2 = self.feature_extractor(p[1])
        ## modified scale ##
        nn_distance = KnnDistance(p[0], 3)
        voxel_scale = self.scale_offset * nn_distance

        # correlation matrix
        transport = ot.sinkhorn(fmap1.transpose(1,-1), fmap2.transpose(1,-1), xyz1, xyz2, 
            epsilon=torch.exp(self.epsilon) + 0.03, 
            gamma=self.gamma, #torch.exp(self.gamma), 
            max_iter=1)
        self.corr_block.init_module(fmap1, fmap2, xyz2, transport)

        fct1, graph_context = self.context_extractor(p[0])

        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords1, coords2 = xyz1, xyz1
        flow_predictions = []
        all_delta_flow = []  

        for itr in range(num_iters):
            coords2 = coords2.detach()
            corr = self.corr_block(coords=coords2, all_delta_flow=all_delta_flow, num_iters=num_iters, scale=voxel_scale)  
            flow = coords2 - coords1
            net, delta_flow = self.update_block(net, inp, corr, flow, graph_context)
            all_delta_flow.append(delta_flow)  
            coords2 = coords2 + delta_flow
            flow_predictions.append(coords2 - coords1)

        return flow_predictions