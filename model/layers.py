import torch
import torch.nn as nn
import torch.nn.functional as F


from compressai.models import MeanScaleHyperprior

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import (
    GDN,
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from .subnet import *
from .subnet.bitEstimator import ICLR17EntropyCoder
gpu_num = torch.cuda.device_count()



def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class Compressor(nn.Module):
    def __init__(self):
        super(Compressor, self).__init__()
        self.addres = False
        self.Encoder = FeatureEncoder()
        self.Decoder = FeatureDecoder()
        self.aligner = PCD_Align()

        self.OffsetPriorEncoder = OffsetPriorEncodeNet()
        self.OffsetPriorDecoder = OffsetPriorDecodeNet()
        self.EntropyCoding_offsetf = NIPS18nocEntropyCoder_mv()
        self.EntropyCoding_offsetz = ICLR17EntropyCoder(out_channel_mv)

        self.resEncoder = ResEncodeNet()
        self.resDecoder = ResDecodeNet()
        self.resPriorEncoder = ResPriorEncodeNet()
        self.resPriorDecoder = ResPriorDecodeNet()
        self.EntropyCoding_residualf = NIPS18nocEntropyCoder_res()
        self.EntropyCoding_residualz = ICLR17EntropyCoder(out_channel_resN)

        
        self.motionbpp = 1
        self.residualbpp = 1
        self.true_lambda = 128
        self.finalmse = 1


    def Trainall(self):
        for p in self.parameters():
            p.requires_grad = True

    def TrainwoMotion(self):
        for p in self.parameters():
            p.requires_grad = True
        for p in self.OffsetPriorEncoder.parameters():
            p.requires_grad = False
        for p in self.OffsetPriorDecoder.parameters():
            p.requires_grad = False
        for p in self.EntropyCoding_offsetf.parameters():
            p.requires_grad = False
        for p in self.EntropyCoding_offsetz.parameters():
            p.requires_grad = False
        for p in self.aligner.parameters():
            p.requires_grad = False


    def Trainstage(self, global_step):
        if global_step < 200000:
            self.motionmse = 1
            self.motionbpp = 1
            self.residualbpp = 0
            self.finalmse = 0
        elif global_step < 400000:
            self.TrainwoMotion()
            self.motionmse = 0
            self.motionbpp = 0
            self.residualbpp = 0
            self.finalmse = 1
        elif global_step < 500000:
            self.TrainwoMotion()
            self.motionmse = 0
            self.motionbpp = 0
            self.residualbpp = 1
            self.finalmse = 1
        else:
            self.Trainall()
            self.motionmse = 0
            self.motionbpp = 1
            self.residualbpp = 1
            self.finalmse = 1


    def Q(self, x):
        if self.training:
            return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        else:
            return torch.round(x)

    def mv_compressor(self, mv_input):
        
        # print("mv_input的形状", mv_input.shape)
        # mv_feature = self.Encoder(mv_input)
        # print("mv_input编码后的的形状", mv_input.shape)

        encoded_offset_prior = self.OffsetPriorEncoder(mv_input)
        # print("offset_prior的形状", encoded_offset_prior.shape)
        q_offset_prior, bits_offsetz = self.EntropyCoding_offsetz(encoded_offset_prior)
        # print("q_offset_prior的形状", q_offset_prior.shape)
        decoded_offset_prior = self.OffsetPriorDecoder(q_offset_prior)
        # print("mvinput,decoded_offset_prior的形状", mv_input.shape, decoded_offset_prior.shape)
        q_offset, bits_offsetf = self.EntropyCoding_offsetf(mv_input, decoded_offset_prior)

        # mv_output = self.Decoder(q_offset)

        return q_offset, bits_offsetf, bits_offsetz

    def res_compressor(self, res_input):
        res_feature = self.Encoder(res_input)
        # print("res_input的形状", res_input.shape)
        encoded_residual = self.resEncoder(res_feature)
        # print("res_input编码后的的形状", encoded_residual.shape)

        # hyperprior
        encoded_residual_prior = self.resPriorEncoder(encoded_residual)
        q_encoded_residual_prior, bits_residualz = self.EntropyCoding_residualz(encoded_residual_prior)
        decoded_residual_prior = self.resPriorDecoder(q_encoded_residual_prior)
        # print("decoded_residual_prior的形状", decoded_residual_prior.shape)
        q_encoded_residual, bits_residualf = self.EntropyCoding_residualf(encoded_residual, decoded_residual_prior)

        res_output_feature = self.resDecoder(q_encoded_residual)
        res_output = self.Decoder(res_output_feature)  

        return res_output, bits_residualf, bits_residualz

    def GetLoss(self, image, image_re, bits_offsetf, bits_offsetz, bits_residualf, bits_residualz, allarea):
        out = dict()

        out["mse_loss"] = torch.mean((image_re - image).pow(2))

        out["bpp_offsetf"] = bits_offsetf / allarea
        out["bpp_offsetz"] = bits_offsetz / allarea
        out["bpp_residualf"] = bits_residualf / allarea
        out["bpp_residualz"] = bits_residualz / allarea
        out["mv_bpp"] = out["bpp_offsetf"] + out["bpp_offsetz"]
        out["res_bpp"] = out["bpp_residualf"] + out["bpp_residualz"]
        out["bpp"] = out["mv_bpp"] + out["res_bpp"]

        # out["b_loss"] = self.true_lambda * (self.finalmse *out["mse_loss"]) + self.motionbpp * (out["bpp_offsetf"] + out["bpp_offsetz"]) + self.residualbpp * (out["bpp_residualf"] + out["bpp_residualz"])

        return out


class MVCompressor(MeanScaleHyperprior):

    def __init__(self, N=128, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(4, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 4, 2),
        )
        
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        print("经过compress模块")
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat)
        
        return {"x_hat": x_hat}
        
class ResidualCompressor(MeanScaleHyperprior):

    def __init__(self, N=128, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
        
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}    
        

class Mask(nn.Module):

    def __init__(self, ch=32):

        super(Mask, self).__init__()
                
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = conv(6, ch, 5, 1)
        self.conv2 = conv(ch, ch*2, 5, 1)
        self.conv3 = conv(ch*2, ch*4, 3, 1)
        self.bottleneck = conv(ch*4, ch*4, 3, 1)
        self.deconv1 = conv(ch*8, ch*4 ,3, 1)
        self.deconv2 = conv(ch*4+ch*2, ch*2, 5, 1)
        self.deconv3 = conv(ch*2+ch, ch, 5, 1)
        self.conv4 = conv(ch, 1, 5, 1)
    
        
    def forward(self, x):

        x = self.conv1(x)
        conv1 = F.relu(x)
        x = self.pool(conv1)

        x = self.conv2(x)
        conv2 = F.relu(x)
        x = self.pool(conv2)

        x = self.conv3(x)
        conv3 = F.relu(x)
        x = self.pool(conv3)

        x = self.bottleneck(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv3], dim=1)
        x = self.deconv1(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv2], dim=1)
        x = self.deconv2(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv1], dim=1)
        x = self.deconv3(x)
        x = F.relu(x)

        mask = torch.sigmoid(self.conv4(x))

        return mask
