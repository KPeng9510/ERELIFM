# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import os
import shutil
from functools import partial
from time import time
from dataset.dataloader import get_dataloader, get_transform
from dataset.dataset_with_label_noise_mnist import SingleDomainData, SingleClassData, MultiDomainData
from lightly import loss
from lightly import transforms
from lightly.models.modules import heads
import torch
from omegaconf import OmegaConf
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets_prep import get_dataset
from EMA import EMA
from models import create_network
from torchdiffeq import odeint_adjoint as odeint

model = 'convnet'
target = ['mnist']#['']
source =  ['syn', 'mnist_m', 'svhn' ,'mnist']#['sketch', 'cartoon', 'art_painting', 'photo']
asym = False
nsr = 0.5
gpu = 0

source.remove(target[0])

if nsr == 0.5:
    ns = '0_5'
if nsr == 0.2:
    ns = '0_2'
if nsr == 0.8:
    ns = '0_8'    
# faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb


def sample_from_model(model, x_0):
    t = torch.tensor([1.0, 0.0], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    return fake_image


# %%
def train(args):
    from diffusers.models import AutoencoderKL

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float32
    set_seed(args.seed + accelerator.process_index)
    source_domain = sorted(args.source_domain)
    target_domain = sorted(args.target_domain)
    known_classes = sorted(args.known_classes)
    unknown_classes = sorted(args.unknown_classes)
    batch_size = args.batch_size

    train_dir = 'path/to/DDG/train'
    val_dir = 'path/to/DDG/val'
    test_dir = ['path/to/DDG/train', 'path/to/DDG/val']
    sub_batch_size = batch_size // 2
    small_img = True      


    num_domain = len(source_domain)
    num_classes = len(known_classes)
    class_index = [i for i in range(num_classes)]
    #domain_specific_loader = []
    transform = transforms.SimCLRTransform(input_size=256, cj_prob=0.8)
    loader = MultiDomainData(root_dir=train_dir, domain=source_domain, classes=known_classes, nsr=nsr, asymetric=asym,transform=get_transform("train", small_img=small_img), simclr_transform=transform, train=True)
    #weak_labels = []
    N = len(loader)
    import pickle as pkl
    f = open('path/to/Digits_index_'+target_domain[0]+'_0_5_re.pkl', 'rb+')
    indexes = pkl.load(f)
    f.close()
    clean_indexes = indexes[0]
    noisy_indexes = indexes[1]
    
    trainset_1 = torch.utils.data.Subset(loader, clean_indexes)
    all_domains = []
    all_index = []
    all_labels = []

    data_loader = torch.utils.data.DataLoader(
        trainset_1,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = create_network(args).to(device, dtype=dtype)
    if args.use_grad_checkpointing and "DiT" in args.model_type:
        model.set_gradient_checkpointing()

    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    accelerator.print("AutoKL size: {:.3f}MB".format(get_weight(first_stage_model)))
    accelerator.print("FM size: {:.3f}MB".format(get_weight(model)))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

    data_loader, model, optimizer, scheduler = accelerator.prepare(data_loader, model, optimizer, scheduler)

    exp = args.exp
    parent_dir = "./scala_digits_l_saved_info_digits_residual_"+target_domain[0]+"_0_5/latent_flow/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if accelerator.is_main_process:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            config_dict = vars(args)
            OmegaConf.save(config_dict, os.path.join(exp_path, "config.yaml"))
    accelerator.print("Exp path:", exp_path)
    init_epoch = 0
    epoch = init_epoch
    global_step = 0
    if args.resume or os.path.exists(os.path.join(exp_path, "content.pth")):
        pass


    elif args.model_ckpt and os.path.exists(os.path.join(exp_path, args.model_ckpt)):
        checkpoint_file = os.path.join(exp_path, args.model_ckpt)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        epoch = int(args.model_ckpt.split("_")[-1][:-4])
        init_epoch = 0 #0epoch
        model.load_state_dict(checkpoint, strict=False)
        global_step = 0

        accelerator.print("=> loaded checkpoint (epoch {})".format(epoch))
        del checkpoint
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    use_label = True if "imagenet" in args.dataset else False
    is_latent_data = True if "latent" in args.dataset else False
    log_steps = 0
    start_time = time()


    for epoch in range(init_epoch, args.num_epoch + 1):
        for iteration, (x, x_, y, y_t, d_s, d_t) in enumerate(data_loader):
            x_0 = x_.to(device, dtype=dtype, non_blocking=True) - x.to(device, dtype=dtype, non_blocking=True)
            y = None if not use_label else y.to(device, non_blocking=True)
            model.zero_grad()
            if is_latent_data:
                z_0 = x_0 * args.scale_factor
            else:
                z_0 = first_stage_model.encode(x_0).latent_dist.sample().mul_(args.scale_factor)
            # sample t
            t = torch.rand((z_0.size(0),), dtype=dtype, device=device)
            t = t.view(-1, 1, 1, 1)
            z_1 = torch.randn_like(z_0)
            z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
            u = (1 - 1e-5) * z_1 - z_0
            # estimate velocity

            v = model(t.squeeze(), z_t, y, y_t, d_s.long(), d_t.long())
            loss = F.mse_loss(v, u)
            accelerator.backward(loss)
            optimizer.step()
            global_step += 1
            log_steps += 1
            if iteration % 100 == 0:
                if accelerator.is_main_process:
                    # Measure training speed:
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    accelerator.print(
                        "epoch {} iteration{}, Loss: {}, Train Steps/Sec: {:.2f}".format(
                            epoch, iteration, loss.item(), steps_per_sec
                        )
                    )
                    # Reset monitoring variables:
                    log_steps = 0
                    start_time = time()

        if not args.no_lr_decay:
            scheduler.step()

        #if accelerator.is_main_process:
        if True:
            if epoch % args.plot_every == 0:
                with torch.no_grad():
                    rand = torch.randn_like(z_0)[:4]
                    if y is not None:
                        y = y[:4]
                    sample_model = partial(model, y=y, y_t=y_t[:4], d_s=d_s[:4].long(), d_t=d_t[:4].long())
                    # sample_func = lambda t, x: model(t, x, y=y)
                    fake_sample = sample_from_model(sample_model, rand)[-1]
                    fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
                torchvision.utils.save_image(
                    fake_image,
                    os.path.join(exp_path, "image_epoch_{}.png".format(epoch)),
                    normalize=True,
                    value_range=(-1, 1),
                )
                accelerator.print("Finish sampling")

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    accelerator.print("Saving content.")
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        "model_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }

                    torch.save(content, os.path.join(exp_path, "content.pth"))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(
                    model.state_dict(),
                    os.path.join(exp_path, "model_{}.pth".format(epoch)),
                )
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser("ddgan parameters")
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default="", help="Model ckpt to init from")    #/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/LFM/checkpoint/model_875.pth
    parser.add_argument('--source-domain', nargs='+', default=source)
    parser.add_argument('--target-domain', nargs='+', default=target)
    parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4', '5'])
    parser.add_argument('--unknown-classes', nargs='+', default=['6', '7', '8', '9'])

    parser.add_argument(
        "--model_type",
        type=str,
        default="DiT-L/2",
        help="model_type",
        choices=[
            "adm",
            "ncsn++",
            "ddpm++",
            "DiT-B/2",
            "DiT-S/2",
            "DiT-L/2",
            "DiT-L/4",
            "DiT-XL/2",
        ],
    )
    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsample rate of input image by the autoencoder",
    )
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of model")
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=(16,),
        help="resolution of applying attention",
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        type=int,
        default=(1, 1, 2, 2, 4, 4),
        help="channel mult",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--label_dim", type=int, default=0, help="label dimension, 0 if unconditional")
    parser.add_argument(
        "--augment_dim",
        type=int,
        default=0,
        help="dimension of augmented label, 0 if not used",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="num classes")
    parser.add_argument(
        "--label_dropout",
        type=float,
        default=0.0,
        help="Dropout probability of class labels for classifier-free guidance",
    )

    # Original ADM
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--use_origin_adm", action="store_true")
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument("--num_heads", type=int, default=4, help="number of head")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")

    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")

    # training
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--dataset", default="imagenet_256", help="name of dataset")
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument(
        "--use_grad_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing for mem saving",
    )

    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=1200)

    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate g")

    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument("--use_ema", action="store_true", default=False, help="use EMA or not")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate for EMA")

    parser.add_argument("--save_content", action="store_true", default=False)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=10,
        help="save content for resuming every x epochs",
    )
    parser.add_argument("--save_ckpt_every", type=int, default=25, help="save ckpt every x epochs")
    parser.add_argument("--plot_every", type=int, default=5, help="plot every x epochs")



    args = parser.parse_args()

    train(args)
