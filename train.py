# coding: utf-8

# In[1]:

import time
import torch
import torch.nn as nn
from torch.autograd import Variable, variable
import numpy as np
import glob
import os
import itertools
import warnings

warnings.filterwarnings('ignore')
import imageio
from natsort import natsorted
import math
import logging
import time
import random
import torch.optim as optim
import matplotlib.pyplot as plt
from model import m
import tqdm
from datetime import datetime


device = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 日志配置
logging.basicConfig(filename="results/train.log", level=logging.INFO)


def normalize(tensor):
    norm = (tensor) / 255.
    return norm


def float_to_uint8(image):
    clip = np.clip(image, 0, 1) * 255.
    im_uint8 = np.round(clip).astype(np.uint8).transpose(1, 2, 0)
    return im_uint8


def MSE(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    return mse


def PSNR(mse, data_range):
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return psnr


def calculate_distortion_loss(out, real):
    distortion_loss = torch.mean((out - real) ** 2)
    return distortion_loss


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-4, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.metrics = metrics

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # Adjust the bpp_loss calculation
        out["bpp_loss"] = sum(
            (torch.log(likelihoods + 1e-9).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.metrics == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["ms_ssim_loss"] = None
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        elif self.metrics == 'ms-ssim':
            out["mse_loss"] = None
            out["ms_ssim_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * out["ms_ssim_loss"] + out["bpp_loss"]

        return out


def train_sequences_loader(path=""):
    sequences = []
    folders = natsorted(glob.glob(path + "*"))
    for f in folders:
        v = natsorted(glob.glob(f + "/*"))
        for vid in v:
            sequences.append(vid)
    return sequences


def train_video_loader(sequences):
    video_paths = []
    for sequence in sequences:
        videos = natsorted(glob.glob(sequence + "/*"))
        video_paths.extend(videos)
    return video_paths


def load_train_list(train_list_path):
    with open(train_list_path, 'r') as file:
        train_images = [line.strip() for line in file.readlines()]
    return train_images


def train_image_loader(video_paths):
    gop_video_batch = []

    for video in video_paths:
        gop_im_list = natsorted(glob.glob(video + "/*.png"))
        gop_video_batch.append(gop_im_list)

    return gop_video_batch

def prepare_train_data(gop_video_batch):
    X_train = []

    patch_size = 256
    size = 5
    length = 7

    for gop_ims in gop_video_batch:

        s = random.randint(0, length - size)

        gop_split = gop_ims[s:s + size]

        sample_im = imageio.imread(gop_split[0])

        x = random.randint(0, sample_im.shape[1] - patch_size)
        y = random.randint(0, sample_im.shape[0] - patch_size)

        img1 = imageio.imread(gop_split[0])
        img1 = img1[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1)
        img2 = imageio.imread(gop_split[size // 2])
        img2 = img2[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1)
        img3 = imageio.imread(gop_split[-1])
        img3 = img3[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1)

        img_concat = np.concatenate((img1, img2), axis=0)
        img_concat = np.concatenate((img_concat, img3), axis=0)

        X_train.append(img_concat)

    X_train = np.array(X_train)

    return X_train


def i_compress(im, model):
    out1 = model.invcompress(im)
    return out1


def b_compress(im_before, im_current, im_after, model, train=True):
    dec_current, out = model.forward(im_before, im_current, im_after, True)
    return dec_current, out



def train_one_step(im_batch, model, optimizer, aux_optimizer, criterion):
    alpha = 3141.
    beta = 1.
    true_lambda = 2048
    model = model.train()


    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    X_train = prepare_train_data(im_batch)
    X_train = normalize(torch.from_numpy(X_train).to(device).float())

    x1 = X_train[:, 0:3]
    x2 = X_train[:, 3:6]
    x3 = X_train[:, 6:9]

    out_x1 = i_compress(x1, model)
    out_x3 = i_compress(x3, model)


    dec_x2, loss_out = b_compress(out_x1["x_hat"], x2, out_x3["x_hat"], model, True)

    criterion_out_x1 = criterion(out_x1, x1)
    i_dist_loss_x1 = criterion_out_x1["loss"]
    i_rate_loss_x1 = criterion_out_x1["bpp_loss"]

    b_dist_loss = loss_out["mse_loss"]
    b_mv_loss = loss_out["mv_bpp"]
    b_res_loss = loss_out["res_bpp"]
    b_bpp_loss = (loss_out["mv_bpp"] + loss_out["res_bpp"]) / 2


    loss = i_dist_loss_x1 + i_rate_loss_x1 + alpha * b_dist_loss + beta * b_bpp_loss

    loss.backward()

    optimizer.step()
    aux_optimizer.step()

    return loss.item(), i_dist_loss_x1.item(), i_rate_loss_x1.item(), b_dist_loss.item(), b_bpp_loss.item()


def save_model(model, child, name):
    state = {
        "state_dict": getattr(model, child).state_dict()
    }
    torch.save(state, name + "model.pth")


def save_all_model(model, save_path):
    state = {
        "state_dict": model.state_dict()
    }
    torch.save(state, save_path)


def save_optimizer(optimizer, name):
    state = {
        "state_dict": optimizer.state_dict()
    }
    torch.save(state, name + "optimizer.pth")


def cosine_annealing_lr(current_step, total_steps, initial_lr):
    return initial_lr * 0.5 * (1 + math.cos(math.pi * current_step / total_steps))


def main():
    b_loss_list = []
    i_loss_list = []
    train_loss_list = []
    iterations_list = []
    i_bpp_loss_list = []
    b_bpp_loss_list = []


    total_train_step = 2000
    train_step = 1000

    learning_rate_1 = 1e-4  # 1.e-5
    learning_rate_2 = 1e-3  # 1e-2

    model = m.Model()
    model = model.to(device).float()

    checkpoint_path = ""
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
    aux_parameters = set(p for n, p in model.named_parameters() if n.endswith(".quantiles"))
    optimizer = optim.Adam(parameters, lr=learning_rate_1)
    aux_optimizer = optim.Adam(aux_parameters, lr=learning_rate_2)


    criterion = RateDistortionLoss(lmbda=1e-2, metrics='mse')

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    start_time = time.time()

    sequences = train_sequences_loader()
    videos = train_video_loader(sequences)

    batch_size = 4


    step_train_loss = 0
    step_i_loss = 0
    step_b_loss = 0
    step_i_bpp_loss = 0
    step_b_bpp_loss = 0

    print('Start training at:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

    # Training loop
    for minibatch_processed in tqdm.tqdm(range(1, total_train_step + 1)):

        gop_video_batch = random.sample(videos, batch_size)
        gop_im_batch = train_image_loader(gop_video_batch)

        loss, i_loss, i_bpp_loss, b_loss, b_bpp_loss = train_one_step(gop_im_batch, model, optimizer,
                                                                      aux_optimizer, criterion)
        step_train_loss += loss
        step_i_loss += i_loss
        step_i_bpp_loss += i_bpp_loss
        step_b_loss += b_loss
        step_b_bpp_loss += b_bpp_loss

        if minibatch_processed % train_step == 0:
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            logging.info("Training started at: " + time.ctime(start_time))
            logging.info("Current time: " + current_time)
            logging.info("iterations: " + str(minibatch_processed))
            logging.info("train loss: " + str(step_train_loss / train_step))
            logging.info("i loss: " + str(step_i_loss / train_step))
            logging.info("i bpp loss: " + str(step_i_bpp_loss / train_step))
            logging.info("b loss: " + str(step_b_loss / train_step))
            logging.info("b bpp loss: " + str(step_b_bpp_loss / train_step))
            logging.info("**************")

            train_loss_list.append(step_train_loss / train_step)
            i_loss_list.append(step_i_loss / train_step)
            i_bpp_loss_list.append(step_i_bpp_loss / train_step)
            b_loss_list.append(step_b_loss / train_step)
            b_bpp_loss_list.append(step_b_bpp_loss / train_step)
            iterations_list.append(minibatch_processed)

            step_train_loss = 0
            step_i_loss = 0
            step_i_bpp_loss = 0
            step_b_loss = 0
            step_b_bpp_loss = 0

    fig, axs = plt.subplots(2, 5, figsize=(15, 5))

    axs[0, 0].plot(iterations_list, b_loss_list, label='B Loss')
    axs[0, 0].set_title('B Loss')
    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('Loss')

    axs[0, 1].plot(iterations_list, train_loss_list, label='Train Loss')
    axs[0, 1].set_title('Train Loss')
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('Loss')

    axs[0, 2].plot(iterations_list, i_loss_list, label='I Loss')
    axs[0, 2].set_title('I Loss')
    axs[0, 2].set_xlabel('Iterations')
    axs[0, 2].set_ylabel('Loss')

    axs[0, 3].plot(iterations_list, i_bpp_loss_list, label='I BPP Loss')
    axs[0, 3].set_title('I BPP Loss')
    axs[0, 3].set_xlabel('Iterations')
    axs[0, 3].set_ylabel('Loss')

    axs[0, 4].plot(iterations_list, b_bpp_loss_list, label='B bpp Loss')
    axs[0, 4].set_title('B bpp Loss')
    axs[0, 4].set_xlabel('Iterations')
    axs[0, 4].set_ylabel('Loss')

    plt.savefig("results/loss_plots.png")

    final_model_path = f"results/final_model_9_6_1433_1e-2_{total_train_step}.pth"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved at {final_model_path}")

    Donetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging.info(f"Training finished at {Donetime}")

    print('Training done at:', Donetime)


if __name__ == '__main__':
    main()
