# coding: utf-8
import os

import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
from compressai.models.ours import InvCompress
from model.layers import MVCompressor, ResidualCompressor, Mask, Compressor
from model.gmflow.gmflow import GMFlow

device = torch.device("cuda")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.FlowNet = GMFlow()
        self.invcompress = InvCompress()
        self.compressor = Compressor()
        self.masknet = Mask()

        self.upsample_flow = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x_before, x_current, x_after, train):
        N, C, H, W = x_current.size()
        num_pixels = N * H * W

        # Flow calculation and processing
        flow_ba = F.avg_pool2d(self.FlowNet(x_before, x_after) / 2., 4)
        flow_ab = F.avg_pool2d(self.FlowNet(x_after, x_before) / 2., 4)

        nn, cc, hh, ww = flow_ab.size()
        flow_ba = self.pad(flow_ba)
        flow_ab = self.pad(flow_ab)

        flow_cb = F.avg_pool2d(self.FlowNet(x_current, x_before), 4)
        flow_ca = F.avg_pool2d(self.FlowNet(x_current, x_after), 4)

        flow_cb = self.pad(flow_cb)
        flow_ca = self.pad(flow_ca)

        diff_flow = torch.cat([flow_cb - flow_ab, flow_ca - flow_ba], dim=1)
        flow_result, bits_offsetf, bits_offsetz = self.compressor.mv_compressor(diff_flow)

        flow_cb_hat, flow_ca_hat = torch.chunk(flow_result, 2, dim=1)
        flow_cb_hat = flow_cb_hat + flow_ab
        flow_cb_hat = F.interpolate(flow_cb_hat[:, :, :hh, :ww], size=(H, W), mode='bilinear', align_corners=False)
        flow_ca_hat = flow_ca_hat + flow_ba
        flow_ca_hat = F.interpolate(flow_ca_hat[:, :, :hh, :ww], size=(H, W), mode='bilinear', align_corners=False)

        fw, bw = self.backwarp(x_before, flow_cb_hat), self.backwarp(x_after, flow_ca_hat)

        mask = self.masknet(torch.cat([fw, bw], dim=1))
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False).repeat([1, 3, 1, 1])

        x_current_hat = mask * fw + (1.0 - mask) * bw

        residual = x_current - x_current_hat

        residual_result, bits_residualf, bits_residualz = self.compressor.res_compressor(residual)
        residual_hat = residual_result

        x_current_hat = residual_hat + x_current_hat

        out = self.compressor.GetLoss(x_current, x_current_hat, bits_offsetf, bits_offsetz, bits_residualf,
                                      bits_residualz, num_pixels)

        if train:
            return x_current_hat, out
        else:
            return x_current_hat, out["bpp"]

    def pad(self, im):
        (m, c, w, h) = im.size()

        p1 = (64 - (w % 64)) % 64
        p2 = (64 - (h % 64)) % 64

        pad = nn.ReflectionPad2d(padding=(0, p2, 0, p1))
        return pad(im)

    def backwarp(self, tenInput, tenFlow):

        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]),
                                tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]),
                                tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid = torch.cat([tenHor, tenVer], 1).to(device)
        # end

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                             tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

        return torch.nn.functional.grid_sample(input=tenInput,
                                               grid=(backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1),
                                               mode='bilinear', padding_mode='border', align_corners=False)


