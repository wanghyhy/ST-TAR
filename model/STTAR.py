import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class GCN_Layer(nn.Module):
    def __init__(self, num_of_features, num_of_filter):

        super(GCN_Layer, self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features=num_of_features,
                      out_features=num_of_filter),
            nn.ReLU()
        )

    def forward(self, input, adj):

        batch_size, _, _ = input.shape  # 224,197,3
        adj = torch.from_numpy(adj).to(input.device)
        adj = adj.repeat(batch_size, 1, 1)  # 224,197,197
        input = torch.bmm(adj, input)  # 224,197,3
        output = self.gcn_layer(input)  # 224,197,64
        return output


class CrossAttention(nn.Module):
    def __init__(self, d_model, transformer_hidden_size, num_of_heads, num_of_target_time_feature,
                 dropout=0):
        super(CrossAttention, self).__init__()
        self.fc0 = nn.Linear(in_features=d_model, out_features=transformer_hidden_size)
        self.fc1 = nn.Linear(in_features=d_model, out_features=transformer_hidden_size//2)
        self.fct = nn.Linear(in_features=32, out_features=transformer_hidden_size//2)
        self.attention_model = nn.MultiheadAttention(embed_dim=transformer_hidden_size, num_heads=4,batch_first=True)

    def forward(self, input_features, other_features, target_time_feature, N):

        batch_size = target_time_feature.size(0)
        transformer_output = self.fc0(input_features)

        other_features = self.fc1(other_features)
        target_time_feature= torch.unsqueeze(target_time_feature, 1).repeat(1, N, 1).view(batch_size * N, -1)
        target_output=self.fct(target_time_feature)


        # Attention Mechanism
        attn_output, _ = self.attention_model(query=torch.cat((torch.unsqueeze(other_features, 1),torch.unsqueeze(target_output,1)), dim=2),key=transformer_output, value=transformer_output)

        return torch.squeeze(attn_output)


class GraphConv(nn.Module):
    def __init__(self, num_of_graph_feature, nums_of_graph_filters, north_south_map, west_east_map):

        super(GraphConv, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.road_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.road_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.risk_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.risk_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.poi_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.poi_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

    def forward(self, graph_feature, road_adj, risk_adj, poi_adj):

        batch_size, T, D1, N = graph_feature.shape

        road_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.road_gcn:
            road_graph_output = gcn_layer(road_graph_output, road_adj)

        risk_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.risk_gcn:
            risk_graph_output = gcn_layer(risk_graph_output, risk_adj)

        graph_output = road_graph_output + risk_graph_output

        if poi_adj is not None:
            poi_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
            for gcn_layer in self.poi_gcn:
                poi_graph_output = gcn_layer(poi_graph_output, poi_adj)
            graph_output += poi_graph_output

        graph_output = graph_output.view(batch_size, T, N, -1) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .view(batch_size * N, T, -1) \
            .view(batch_size, T, -1, N)

        return graph_output


class GraphTsEncoder(nn.Module):
    def __init__(self, transformer_hidden_size, num_of_target_time_feature,
                 north_south_map, west_east_map, num_of_heads):

        super(GraphTsEncoder, self).__init__()

        self.ata = CrossAttention(d_model=transformer_hidden_size,
                                             transformer_hidden_size=transformer_hidden_size,
                                             num_of_heads=num_of_heads,
                                             num_of_target_time_feature=num_of_target_time_feature)

        self.north_south_map = north_south_map
        self.west_east_map = west_east_map

    def forward(self, g_output, o_output, target_time_feature, grid_node_map):
        batch_size, T, _, N = g_output.shape

        graph_output = g_output.view(batch_size, T, N, -1) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .view(batch_size * N, T, -1)
        
        other_output = torch.mean(o_output,dim=1).view(batch_size, N, -1) \
            .contiguous() \
            .view(batch_size * N, -1)

        graph_output = graph_output.view(batch_size * N, T, -1)  # （32*197, 7, 64）

        ata_output = self.ata(graph_output, other_output, target_time_feature, N)

        ata_output = ata_output.view(batch_size, N, -1).contiguous()  # (32,197,64)

        grid_node_map_tmp = torch.from_numpy(grid_node_map) \
            .to(ata_output.device) \
            .repeat(batch_size, 1, 1)
        ata_output = torch.bmm(grid_node_map_tmp, ata_output) \
            .permute(0, 2, 1) \
            .view(batch_size, -1, self.north_south_map, self.west_east_map)
        return ata_output


class ST_Model(nn.Module):
    def __init__(self,  transformer_hidden_size,pre_len,
                 num_of_target_time_feature, num_of_graph_feature, nums_of_graph_filters, north_south_map,
                 west_east_map, num_of_heads):
        
        super(ST_Model, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.fusion_channel = 16

        self.GConv = nn.ModuleList([GraphConv(num_of_graph_feature, nums_of_graph_filters,
                                              north_south_map[0], west_east_map[0]).cuda(),
                                    GraphConv(num_of_graph_feature, nums_of_graph_filters,
                                              north_south_map[1], west_east_map[1]).cuda(),
                                    GraphConv(num_of_graph_feature, nums_of_graph_filters,
                                              north_south_map[2], west_east_map[2]).cuda(),
                                    GraphConv(num_of_graph_feature, nums_of_graph_filters,
                                              north_south_map[3], west_east_map[3]).cuda()])

        self.GTsEncoder = nn.ModuleList([GraphTsEncoder(transformer_hidden_size,
                                                        num_of_target_time_feature,
                                                        north_south_map[0], west_east_map[0],
                                                        num_of_heads).cuda(),
                                         GraphTsEncoder(transformer_hidden_size,
                                                        num_of_target_time_feature,
                                                        north_south_map[1], west_east_map[1],
                                                        num_of_heads).cuda(),
                                         GraphTsEncoder(transformer_hidden_size,
                                                        num_of_target_time_feature,
                                                        north_south_map[2], west_east_map[2],
                                                        num_of_heads).cuda(),
                                         GraphTsEncoder(transformer_hidden_size,
                                                        num_of_target_time_feature,
                                                        north_south_map[3], west_east_map[3],
                                                        num_of_heads).cuda()])

        self.graph_weight = nn.ModuleList([nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda(),
                                           nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda(),
                                           nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda(),
                                           nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda()
                                           ])

    def forward(self, input, target_time_feature,
                road_adj, risk_adj, poi_adj, grid_node_map, trans):
        
        batch_size, _, _, _ = input[0].shape

        graph_output = []

        for i in range(4):
            graph_output.append(self.GConv[i](input[i], road_adj[i], risk_adj[i], poi_adj[i]))

        # from fine to coarse
        for i in range(4 - 1):
            f_graph_output = graph_output[i]
            c_graph_output = graph_output[i + 1]

            batch_size, T, _, f_N = f_graph_output.shape
            batch_size1, T, _, c_N = c_graph_output.shape

            # coarse to fine
            c_graph_output = c_graph_output.reshape(batch_size1 * T, -1, c_N)
            cf_out = torch.matmul(c_graph_output, trans[i] / 3)
            f1_graph_output = f_graph_output + 0.2 * cf_out.reshape(batch_size1, T, -1, f_N)

            # fine to coarse
            f_graph_output = f_graph_output.reshape(batch_size * T, -1, f_N)
            fc_out = torch.matmul(f_graph_output, trans[i].permute(0, 2, 1) / 3)

            c_graph_output = c_graph_output.reshape(batch_size1, T, -1, c_N)
            c1_graph_output = c_graph_output + 0.8 * fc_out.reshape((batch_size, T, -1, c_N))

            graph_output[i] = f1_graph_output
            graph_output[i + 1] = c1_graph_output
        return graph_output
    
    def time_forward(self, graph_output, other_output, target_time_feature, grid_node_map):
        time_output=[]
        for i in range(4):
            time_output.append(self.GTsEncoder[i](graph_output[i], other_output[i], target_time_feature[i], grid_node_map[i]))
            time_output[i] = self.graph_weight[i](time_output[i])
        
        return time_output

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_h*2, 1))

    def forward(self, real_sample, false_sample):

        sc_rl = torch.squeeze(self.net(real_sample), dim=2)
        sc_fk = torch.squeeze(self.net(false_sample), dim=2)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits

class InterlossModel(nn.Module):

    def __init__(self, c_in, num_nodes, device):
        super(InterlossModel, self).__init__()

        self.num_nodes = num_nodes
        self.device = device
        self.disc = nn.Bilinear(c_in, c_in, 1)
        self.b_xent = nn.BCEWithLogitsLoss()

    def forward(self, z1, z2):
        
        z1=z1.reshape(z1.shape[0],z1.shape[1],-1).permute(0,2,1)
        z2=z2.reshape(z2.shape[0],z2.shape[1],-1).permute(0,2,1)
        temporal_n = z1.shape[0]
        temporal_idx = torch.randperm(temporal_n)
        spatial_n = z1.shape[1]
        spatial_idx = torch.randperm(spatial_n)
        temporal_z2 = z2[temporal_idx]
        spatial_z2 = z2[:, spatial_idx, :]
        real_logits = self.disc(z1, z2)
        spatial_logits = self.disc(z1, spatial_z2)
        temporal_logits = self.disc(z1, temporal_z2)
        logits = torch.cat((real_logits, spatial_logits, temporal_logits))

        lbl_rl = torch.ones(temporal_n, self.num_nodes, 1)  # label_pos
        lbl_fk = torch.zeros(temporal_n*2, self.num_nodes, 1)  # label_neg
        lbl = torch.cat((lbl_rl, lbl_fk))
        if self.device != 'cpu':
            lbl = lbl.to(logits.device)

        #logits = self.disc(real_sample, false_sample)
        loss = self.b_xent(logits, lbl)
        return loss

class STTAR(nn.Module):
    def __init__(self,  transformer_hidden_size, pre_len,
                 num_of_target_time_feature, num_of_graph_feature, nums_of_graph_filters, north_south_map,
                 west_east_map, num_of_heads):

        super(STTAR, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.fusion_channel = 16

        self.ST_Model= ST_Model(transformer_hidden_size, pre_len,
                 num_of_target_time_feature, num_of_graph_feature, nums_of_graph_filters, north_south_map,
                 west_east_map, num_of_heads)
        
        self.interlossModel = InterlossModel(self.fusion_channel, north_south_map[0]*west_east_map[0], device='cuda')
        
        self.output_layer = nn.ModuleList([nn.Linear(2*self.fusion_channel * north_south_map[0] * west_east_map[0],
                                                     pre_len * north_south_map[0] * west_east_map[0]).cuda(),
                                           nn.Linear(2*self.fusion_channel * north_south_map[1] * west_east_map[1],
                                                     pre_len * north_south_map[1] * west_east_map[1]).cuda(),
                                           nn.Linear(2*self.fusion_channel * north_south_map[2] * west_east_map[2],
                                                     pre_len * north_south_map[2] * west_east_map[2]).cuda(),
                                           nn.Linear(2*self.fusion_channel * north_south_map[3] * west_east_map[3],
                                                     pre_len * north_south_map[3] * west_east_map[3]).cuda()])

    def forward(self, r_input, w_input, target_time_feature,
                road_adj, risk_adj, poi_adj, grid_node_map, trans):
        
        batch_size=r_input[0].shape[0]
        
        rr_output=self.ST_Model(r_input, target_time_feature,
                road_adj, risk_adj, poi_adj, grid_node_map, trans)
        
        ww_output=self.ST_Model(w_input, target_time_feature,
                road_adj, risk_adj, poi_adj, grid_node_map, trans)
        
        r_output=self.ST_Model.time_forward(rr_output, ww_output, target_time_feature,grid_node_map)
        
        w_output=self.ST_Model.time_forward(ww_output, rr_output, target_time_feature,grid_node_map)
        
        
        fusion_output = []
        final_output = []
        classification_output = []

        inter_loss=self.interlossModel(r_output[0], w_output[0])
        
        for i in range(4):
            fusion_output.append(torch.cat((r_output[i], w_output[i]), dim=1))  

        for i in range(4):
            fusion_output[i] = fusion_output[i].contiguous().view(batch_size, -1)
            final_output.append(self.output_layer[i](fusion_output[i])
                                .view(batch_size, -1, self.north_south_map[i], self.west_east_map[i]))

            classification_output.append(torch.relu(final_output[i].view(final_output[i].shape[0], -1)))

        return final_output, classification_output, inter_loss