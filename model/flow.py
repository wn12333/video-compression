import torch

import getopt
import math
import numpy
import os
import sys
# from .motion import optical_flow
import sys
sys.path.append('/root/autodl-tmp/code/LHBDC/model')
from gmflow.gmflow import GMFlow

device = torch.device("cuda")
##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).to(device)
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
    # end

    ##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        class Preprocess(torch.nn.Module):
            def __init__(self):
                super(Preprocess, self).__init__()
            # end

            def forward(self, tenInput):
                tenBlue = (tenInput[:, 0:1, :, :] - 0.406) / 0.225
                tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
                tenRed = (tenInput[:, 2:3, :, :] - 0.485) / 0.229

                return torch.cat([ tenRed, tenGreen, tenBlue ], 1)
            # end
        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            # end

            def forward(self, tenInput):
                return self.netBasic(tenInput)
            # end
        # end

        self.netPreprocess = Preprocess()
        self.netflow = GMFlow()
        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

        #self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load('/kuacc/users/mustafaakinyilmaz/Video_Compression/network-sintel-final.pytorch').items() })
    # end

    def forward(self, tenFirst, tenSecond):

        tenFlow = self.netflow.forward(tenFirst, tenSecond,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=False,
                # **kwargs,
                )

        return tenFlow
