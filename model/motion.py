import torch
import numpy as np

# import evaluate
from .gmflow.gmflow import GMFlow
from .evaluate import inference_on_dir

def optical_flow(image1, image2):
    seed = 326
    torch.manual_seed(seed)
    np.random.seed(seed)
    # print("传来图片的维数", image1.shape)

    # 将输入张量转换为NumPy数组
    image1 = image1.cpu().detach().numpy()
    image2 = image2.cpu().detach().numpy()

    # 重新排列维度 (B, C, H, W) 到 (B, H, W, C)
    image1 = image1.transpose(0, 2, 3, 1)
    image2 = image2.transpose(0, 2, 3, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True
    # model
    model = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   ).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    resume = '/root/autodl-tmp/code/LHBDC/model/pretrained/gmflow_things-e9887eda.pth'
    if resume:
        # print('Load checkpoint: %s' % resume)

        loc = 'cuda:{}'.format(0)
        checkpoint = torch.load(resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model.load_state_dict(weights, strict=True)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                True:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']


    flow = inference_on_dir(model,
                            image1,
                            image2,
                            padding_factor=8,
                            inference_size=None,
                            paired_data=False,  # dir of paired testdata instead of a sequence
                            save_flo_flow=False,  # save as .flo for quantative evaluation
                            attn_splits_list=[2],
                            corr_radius_list=[-1],
                            prop_radius_list=[-1],
                            pred_bidir_flow=False,
                            fwd_bwd_consistency_check=False,
                            )
    return flow     



