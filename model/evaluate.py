import numpy as np
import torch
from .utils.utils import InputPadder
from .utils.flow_viz import save_vis_flow_tofile



@torch.no_grad()
def inference_on_dir(model,
                     image1,     # image还是numpy数组，（1.240.416.3）
                     image2,
                     padding_factor=8,
                     inference_size=None,
                     paired_data=False,  # dir of paired testdata instead of a sequence
                     save_flo_flow=False,  # save as .flo for quantative evaluation
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     pred_bidir_flow=False,
                     fwd_bwd_consistency_check=False,
                     ):
    """ Inference on a directory """
    model.eval()

    # print("降维前维数", image1.shape)

    image1 = np.squeeze(image1, axis=0)     # 降维操作
    image2 = np.squeeze(image2, axis=0)

    # 在这里的image需要numpy数组，形状是（240.416.3）
    if len(image1.shape) == 2:  # gray image, for example, HD1K
        image1 = np.tile(image1[..., None], (1, 1, 3))
        image2 = np.tile(image2[..., None], (1, 1, 3))
    else:
        image1 = image1[..., :3]
        image2 = image2[..., :3]

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    if inference_size is None:
        padder = InputPadder(image1.shape, padding_factor=padding_factor)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
    else:
        image1, image2 = image1[None].cuda(), image2[None].cuda()

    # resize before inference
    if inference_size is not None:
        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                               align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                               align_corners=True)

    # Define a function to test the model
    results_dict = model(image1, image2,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         pred_bidir_flow=pred_bidir_flow,
                         )

    flow_pr = results_dict['flow_preds'][-1]  # torch的tensor 形状是[1, 2, 240, 416]

    # flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
    #
    # save_vis_flow_tofile(flow, 'output\\flow3.png')
    #
    # flow2 = np.expand_dims(flow, axis=0)
    #
    # flow_pr_numpy = flow_pr.cpu().numpy()
    # # 使用NumPy数组创建TensorFlow Tensor并改变形状         #torch到tensorflow


    return flow_pr            # 返回值



