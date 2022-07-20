import os
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import shutil
from PIL import Image

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from options.test_options import TestOptions
from models.psp import pSp
from models.e4e import e4e
from utils.model_utils import ENCODER_TYPES
from utils.common import tensor2im, blend
from utils.inference_utils import run_on_batch

os.makedirs('images_random', exist_ok=True)
os.makedirs('latents_random', exist_ok=True)

def run():

    test_opts = TestOptions().parse()
    os.makedirs(test_opts.temporary_path, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    if opts.encoder_type in ENCODER_TYPES['pSp']:
        net = pSp(opts)
    else:
        net = e4e(opts)

    net.eval()
    net.cuda()
    for i in range(100000):
        latent=torch.randn((1,512), device='cuda')

        output, latent = net.decoder([latent], input_is_latent=False, randomize_noise=None, return_latents=True)
        output = tensor2im(output[0])
        output.save('./images_random/{}.png'.format(i))
        np.save('./latents_random/{}.npy'.format(i), latent.detach().cpu().numpy()[0][0])
    
    shutil.rmtree(opts.temporary_path)

if __name__ == '__main__':
    run()
