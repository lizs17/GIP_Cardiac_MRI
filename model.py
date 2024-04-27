import os
import time
import argparse
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import OrderedDict
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- Independent CNNs
# ----------------------------------------------------------------------------------------------------------------------
class BasicModule_1(nn.Module):
    def __init__(self, inCH, outCH, outSize=None):
        super(BasicModule_1, self).__init__()
        alpha = 1.0
        modules = [
            ('conv', nn.Conv2d(in_channels=inCH, out_channels=outCH, kernel_size=3, stride=1, padding=1, bias=True)),
            ('act', nn.ELU(alpha=alpha, inplace=True)),
            ('norm', nn.InstanceNorm2d(num_features=outCH)),
        ]
        if outSize is not None:
            modules = [('upsample', nn.Upsample(size=outSize, mode='bilinear'))] + modules
        self.net = nn.Sequential(OrderedDict(modules))
    def forward(self, x):
        out = self.net(x)
        return out

class generator_G(nn.Module):
    def __init__(self, device, imSize, Nt, ch, ch_z, ch_out):
        super(generator_G, self).__init__()
        self.device = device
        self.imSize = imSize
        self.Nt = Nt
        # network (separate part)
        self.conv_backbone = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=ch_z, out_channels=16 * ch, kernel_size=3, stride=1, padding=1, bias=True),
                BasicModule_1(inCH=16 * ch, outCH=8 * ch, outSize=(16, 16)),
                BasicModule_1(inCH=8 * ch, outCH=4 * ch, outSize=(32, 32)),
                BasicModule_1(inCH=4 * ch, outCH=2 * ch, outSize=(64, 64)),
                BasicModule_1(inCH=2 * ch, outCH=1 * ch, outSize=(128, 128)),
                BasicModule_1(inCH=1 * ch, outCH=1 * ch, outSize=imSize),
                BasicModule_1(inCH=1 * ch, outCH=1 * ch),
            ) for tt in range(self.Nt)
        ])
        self.conv_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1 * ch, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            ) for tt in range(self.Nt)
        ])
    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, 0.02)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                 nn.init.constant_(module.weight, 1)
                 nn.init.constant_(module.bias, 0)
    def forward(self, z):
        '''
        Input:
            z: (1, ch_z) + size_z
        '''
        # to dynamic image
        x_dynamic = []
        for tt in range(self.Nt):
            x_dynamic.append(self.conv_backbone[tt](z))
        x_feature = torch.cat(x_dynamic, dim=0)
        # pretrain
        x_out = []
        for tt in range(self.Nt):
            x_out.append(self.conv_head[tt](x_feature[tt:(tt + 1), :, : ,:]))
        x = torch.cat(x_out, dim=0)  # (Nt, ch) + imSize
        return x_feature, x
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- FEN
# ----------------------------------------------------------------------------------------------------------------------
class FEN(nn.Module):
    def __init__(self, device, CH):
        super(FEN, self).__init__()
        self.device = device
        self.CH = CH
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=CH, out_channels=CH // 2, kernel_size=5, stride=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=CH // 2, out_channels=CH // 2, kernel_size=5, stride=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=CH // 2, out_channels=CH // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=CH // 2, out_channels=CH // 2, kernel_size=4, stride=2, padding=1, bias=False),
        )
    def forward(self, x):
        # x: (N, CH, Ny, Nx)
        # ------ use residual for feature extraction
        xres = x - torch.mean(x, dim=0, keepdim=True)
        # ------ extract feature (N, CHf, fy, fx)
        xfeat = self.net(xres)
        return xfeat

def distance_cos(x):
    # x: (N, Nf)
    # larger element in 'SI' means higher similarity index
    SI = torch.matmul(x, x.T)
    return SI

def get_knn_idx(x, K, using_posEmbed=False, posEmbed_matrix=None, alpha=0.1):
    # x: (N, Nf)
    # K: number of neighbor
    # posEmbed_matrix: (N, N)
    N = x.size(0)
    # get SI matrix
    SI = distance_cos(x)
    if using_posEmbed:
        assert (posEmbed_matrix >= 0.0).all()  # posEmbed should be greater than 0.0
        SI -= alpha * posEmbed_matrix
    # find k-NN: nn_idx: (N, K)
    _, nn_idx = torch.topk(SI, k=K, dim=1, largest=True)  # get the highest SI
    # check self-node should be the first neighbor
    assert (nn_idx[:, 0] == torch.arange(0, N, dtype=nn_idx.dtype, device=nn_idx.device)).all()
    return nn_idx

class KNN_layer(nn.Module):
    def __init__(self, K=3):
        super().__init__()
        self.K = K
    def forward(self, x):
        # x: (N, CH, fy, fx)
        N = x.size(0)
        # vectorize and normalize
        xvec = x.reshape(N, -1)
        xvec = F.normalize(xvec, p=2, dim=1)  # (N, Nf)
        # get k-NN neighbor index
        nn_idx = get_knn_idx(x=xvec, K=self.K)
        return nn_idx
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- GAL
# ----------------------------------------------------------------------------------------------------------------------
class GraphAggregateLayer(nn.Module):
    def __init__(self, device, N, K, CH):
        super(GraphAggregateLayer, self).__init__()
        self.device = device
        self.N = N
        self.K = K
        self.CH = CH
        # aggregation weights
        self.Wagg = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, stride=1, padding=1, bias=True)
                for k in range(self.K)
            ])
            for i in range(self.N)
        ])
        # activation
        self.act = nn.ModuleList([nn.ELU(alpha=1.0, inplace=True) for i in range(self.N)])
        # normalization
        self.norm = nn.BatchNorm2d(num_features=CH)
    def forward(self, X, nn_idx):
        """
            X: vertex feature  (N, CH, Ny, Nx)
            nn_idx: adjacent index (N, K)  # row_i: the neighbor of vertex_i
        """
        X_hidden = []
        for i in range(self.N):
            idx = nn_idx[i, :]
            assert (idx[0] == i)  # make sure self is the fist neighbor
            Xi_hidden = []
            for k in range(self.K):
                h_ik = self.Wagg[i][k](X[idx[k]:(idx[k] + 1), :, :, :])
                Xi_hidden.append(h_ik)
            Xi_hidden = torch.cat(Xi_hidden, dim=0)  # (K, CH, Ny, Nx)
            # the hidden feature of the i-th vertex
            Xi_hidden = torch.mean(Xi_hidden, dim=0, keepdim=True)  # (1, CH, Ny, Nx)
            Xi_hidden = self.act[i](Xi_hidden)
            # append
            X_hidden.append(Xi_hidden)
        X_hidden = torch.cat(X_hidden, dim=0)  # (N, CH, Ny, Nx)
        X_hidden = self.norm(X_hidden)
        return X_hidden
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- GUL
# ----------------------------------------------------------------------------------------------------------------------
class GraphUpdateLayer(nn.Module):
    def __init__(self, device, N, K, CH):
        super(GraphUpdateLayer, self).__init__()
        self.device = device
        self.N = N
        self.K = K
        self.CH = CH
        # weights for input
        self.Wupd0 = nn.ModuleList([
            nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, stride=1, padding=1, bias=True)
            for i in range(self.N)
        ])
        # activation
        self.act = nn.ModuleList([nn.ELU(alpha=1.0, inplace=True) for i in range(self.N)])
        # normalization
        self.norm = nn.ModuleList([nn.InstanceNorm2d(num_features=CH) for i in range(self.N)])
        # weight update
        self.Wupd = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(in_channels=CH + CH, out_channels=CH, kernel_size=3, stride=1, padding=1, bias=True)
                for k in range(self.K)
            ])
            for i in range(self.N)
        ])
        # activation final
        self.actFinal = nn.ModuleList([nn.ELU(alpha=1.0, inplace=True) for i in range(self.N)])
    def forward(self, X, Xhid, nn_idx):
        """
            X:    vertex feature  (N, CH, Ny, Nx)
            Xhid: vertex hidden feature  (N, CH, Ny, Nx)
            nn_idx:    adjacent matrix (N, K)  # row_i: the neighbor of vertex_i
        """
        Xtmp = []
        for i in range(self.N):
            # the input vertex
            Xi = self.Wupd0[i](X[i:(i + 1), :, :, :])
            Xi = self.norm[i](self.act[i](Xi))
            # cat the features
            Xtmp_i = torch.cat([Xi, Xhid[i:(i + 1), :, :, :]], dim=1)  # (1, 2 * CH, Ny, Nx)
            # append
            Xtmp.append(Xtmp_i)
        Xtmp = torch.cat(Xtmp, dim=0)  # (N, 2 * CH, Ny, Nx)
        # graph update
        Xout = []
        for i in range(self.N):
            idx = nn_idx[i, :]  # get the neighbor indexes
            assert (idx[0] == i)  # make sure self is the first neighbor
            Xout_i = []
            for k in range(self.K):
                X_ik = self.Wupd[i][k](Xtmp[idx[k]:(idx[k] + 1), :, :, :])
                Xout_i.append(X_ik)
            Xout_i = torch.cat(Xout_i, dim=0)  # (K, CH, Ny, Nx)
            # the hidden feature of the i-th vertex
            Xout_i = torch.mean(Xout_i, dim=0, keepdim=True)  # (1, CH, Ny, Nx)
            Xout_i = self.actFinal[i](Xout_i)
            # append
            Xout.append(Xout_i)
        Xout = torch.cat(Xout, dim=0)  # (N, CH, Ny, Nx)
        return Xout
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------- GCN
# ----------------------------------------------------------------------------------------------------------------------
class BasicModule_res(nn.Module):
    def __init__(self, CH):
        super(BasicModule_res, self).__init__()
        alpha = 1.0
        modules = [
            ('conv', nn.Conv2d(in_channels=CH, out_channels=CH, kernel_size=3, stride=1, padding=1, bias=True)),
            ('act', nn.ELU(alpha=alpha, inplace=True)),
            ('norm', nn.InstanceNorm2d(num_features=CH)),
        ]
        self.net = nn.Sequential(OrderedDict(modules))
    def forward(self, x):
        out = self.net(x) + x
        return out

class generator_GCN(nn.Module):
    def __init__(self, device, imSize, Nt, K, CH, ch_out):
        super(generator_GCN, self).__init__()
        self.device = device
        self.imSize = imSize
        self.Nt = Nt
        self.CH = CH
        # ----- Feature Extraction Layer
        self.FEN = FEN(device=device, CH=CH)
        # ----- K-Nearest Neighbor
        self.KNN = KNN_layer(K=K)
        # ----- Graph Aggregate Layer
        self.Agg = GraphAggregateLayer(device=device, N=Nt, K=K, CH=CH)
        # ----- Graph Update Layer
        self.Upd = GraphUpdateLayer(device=device, N=Nt, K=K, CH=CH)
        # ----- conv head
        self.conv = nn.ModuleList([
            nn.Sequential(
                BasicModule_res(CH=CH),
                BasicModule_res(CH=CH),
            ) for tt in range(Nt)
        ])
        self.convFinal = nn.Conv2d(in_channels=CH, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=True)
    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                 nn.init.constant_(module.weight, 1)
                 if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    def forward(self, X, nn_idx=None):
        '''
        Input:
            X: (Nt, CH, Ny, Nx)
            A: (Nt, Nt)    adjacent matrix (N, N)  # row_i: the neighbor of vertex_i
            nn_idx: (Nt, K)
        '''
        if nn_idx is None:
            # ------ node feature extraction
            X_feat = self.FEN(x=X)
            # ------ get adjacent matrix
            nn_idx = self.KNN(x=X_feat)
        # ------ graph convolution
        X_hidden = self.Agg(X=X, nn_idx=nn_idx)  # (Nt, CH, Ny, Nx)
        X_out = self.Upd(X=X, Xhid=X_hidden, nn_idx=nn_idx)  # (Nt, CH, Ny, Nx)
        # ------ additional branched CNNs
        x_dynamic = []
        for tt in range(self.Nt):
            x_dynamic.append(self.conv[tt](X_out[tt:(tt+1), :, :, :]))
        x_rec = torch.cat(x_dynamic, dim=0)  # (Nt, CH) + imSize
        x_rec = self.convFinal(x_rec)
        return x_rec


