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
import cv2

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

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def main():
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
    cam=GradCAM(model=net.encoder, target_layer=net.encoder.input_layer[0])

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root='./gradcam_inputs',
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if not os.path.isfile(os.path.join(opts.exp_dir, 'avg_image.jpg')):
        avg_image = net(net.latent_avg.unsqueeze(0),
                        input_code=True,
                        randomize_noise=False,
                        return_latents=False,
                        average_code=True)[0]
        avg_image = avg_image.to('cuda').float().detach()
    else:
        avg_image=Image.open(os.path.join(opts.exp_dir, 'avg_image.jpg')).convert('RGB')
        avg_image=transforms_dict['transform_inference'](avg_image)

    for input_batch in dataloader:
        input_cuda = input_batch.cuda().float()
        for image_idx, input_image in enumerate(input_batch):
            input_image=input_image.unsqueeze(0).to("cuda").float()

            avg_image_for_batch = avg_image.unsqueeze(0).to("cuda")
            x_input = torch.cat([input_image, avg_image_for_batch], dim=1)

            # for iter in range(opts.n_iters_per_batch):
            #         if iter == 0:
            #             avg_image_for_batch = avg_image.unsqueeze(0).to("cuda")
            #             x_input = torch.cat([input_image, avg_image_for_batch], dim=1)
            #         else:
            #             x_input = torch.cat([input_image, y_hat], dim=1)

            #         y_hat, latent = net(x_input,
            #                             latent=None if iter==0 else latent,
            #                             randomize_noise=False,
            #                             return_latents=True,
            #                             resize=True)
            grayscale_cam = cam(input_tensor=x_input, target_category=12)

            grayscale_cam = grayscale_cam[0, :]


            original_img = cv2.resize(cv2.imread('./gradcam_inputs/sjmoon.png'), (256,256))/256
            print(original_img)
            print(grayscale_cam)
            visualization = show_cam_on_image(original_img, grayscale_cam)
            cv2.imwrite('grad.png',visualization)
            print('Saved!!!')

if __name__ == '__main__':
    main()
