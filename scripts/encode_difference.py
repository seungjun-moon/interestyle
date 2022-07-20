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
from datasets.images_dataset import ImagesDataset
from options.test_options import TestOptions
from models.psp import pSp
from models.e4e import e4e
from utils.model_utils import ENCODER_TYPES
from utils.common import tensor2im, blend, get_concat_h
from utils.inference_utils import run_on_batch

import random


mask_path='/home/ubuntu/seungjun/psp/CelebA_val_segmentation_background'
mask_temp_path='./new_masks'

def run():
    test_opts = TestOptions().parse()
    os.makedirs(test_opts.temporary_path, exist_ok=True)
    os.makedirs(mask_temp_path, exist_ok=True)

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
    image_list = opts.image_list.split(',')
    for name in image_list:
        shutil.copy(os.path.join(opts.data_path, name+'.png'), os.path.join(opts.temporary_path,name+'.png'))
        shutil.copy(os.path.join(mask_path, name+'.png'), os.path.join(mask_temp_path,name+'.png'))

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = ImagesDataset(source_root=opts.temporary_path,
                            target_root=opts.temporary_path,
                            source_transform=transforms_dict['transform_inference'],
                            target_transform=transforms_dict['transform_inference'],
                            mask_root=mask_temp_path,
                            opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    # get the image corresponding to the latent average
    if not os.path.isfile(os.path.join(opts.exp_dir, 'avg_image.jpg')):
        avg_image = net(net.latent_avg.unsqueeze(0),
                        input_code=True,
                        randomize_noise=False,
                        return_latents=False,
                        average_code=True)[0]
        avg_image = avg_image.to('cuda').float().detach()
        tensor2im(avg_image).save(os.path.join(opts.exp_dir, 'avg_image.jpg'))
    else:
        print('Use an existing average image file')
        avg_image=Image.open(os.path.join(opts.exp_dir, 'avg_image.jpg')).convert('RGB')
        avg_image=transforms_dict['transform_inference'](avg_image)

    ct=0
    lb_dict=[]
    for idx, input_batch in enumerate(dataloader):
        input_images,_,masks = input_batch
        with torch.no_grad():
            for input_image,mask in zip(input_images, masks):
                out_image_list=[]
                input_image=input_image.unsqueeze(0).to("cuda").float()
                out_image_list.append(tensor2im(input_image[0]))
                mask = (mask+1)/2
                mask = mask.to("cuda")
                input_image_with_mask = input_image * mask
                out_image_list.append(tensor2im(input_image_with_mask[0]))

                for iter in range(opts.n_iters_per_batch):
                    if iter == 0:
                        avg_image_for_batch = avg_image.unsqueeze(0).to("cuda")
                        x_input = torch.cat([input_image_with_mask, avg_image_for_batch], dim=1)
                    else:
                        x_input = torch.cat([input_image_with_mask, y_hat], dim=1)

                    y_hat, latent = net(x_input,
                                        latent=None if iter==0 else latent,
                                        randomize_noise=False,
                                        return_latents=True,
                                        resize=True) #print with 256, for the concatenation.
                out_image_list.append(tensor2im(y_hat[0]))



                ##### Get latents corresponding to l_b #####
                

                for iter in range(5):
                    if iter == 0:
                        x_input = torch.cat([input_image, input_image_with_mask], dim=1)
                        y_hat, latent_bg = net(x_input,
                                               latent=None,
                                               randomize_noise=False,
                                               return_latents=True,
                                               resize=True) #print with 256, for the concatenation.
                    else:
                        x_input = torch.cat([input_image, y_hat], dim=1)
                        y_hat, latent_bg = net.forward(x_input, latent=latent_bg, return_latents=True)
                
                    ### Random selection
                    lb_dict.append(latent_bg)
                    latent_bg = random.sample(lb_dict,1)[0]
                    print(len(lb_dict))
                    #####

                    output, _=net.decoder([latent+latent_bg], input_is_latent=True, randomize_noise=None, return_latents=False)
                    output = tensor2im(output[0])
                    output = output.resize((256,256))
                    out_image_list.append(output)

                    # output_bg, _=net.decoder([latent_bg], input_is_latent=True, randomize_noise=None, return_latents=False)
                    # output_bg = tensor2im(output_bg[0])
                    # output_bg = output_bg.resize((256,256))
                    # out_image_list.append(output_bg)

                    img = get_concat_h(out_image_list, len(out_image_list), 0)
                    img.save('comp_{}_{}.png'.format(ct,iter))
                    ct+=1


    shutil.rmtree(opts.temporary_path)
    shutil.rmtree(mask_temp_path)

if __name__ == '__main__':
    run()
