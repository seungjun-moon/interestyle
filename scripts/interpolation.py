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


def run(count):
    new_count=count
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

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts,
                               return_name=True)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=True,
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
    step=0
    for input_batch, name_batch in tqdm(dataloader):
        with torch.no_grad():          
            input_image=input_batch.to("cuda").float()

            for iter in range(opts.n_iters_per_batch):
                if iter == 0:
                    avg_image_for_batch = avg_image.unsqueeze(0).repeat(input_image.shape[0], 1, 1, 1).to("cuda")
                    x_input = torch.cat([input_image, avg_image_for_batch], dim=1)
                else:
                    x_input = torch.cat([input_image, y_hat], dim=1)

                y_hat, latent = net(x_input,
                                    latent=None if iter==0 else latent,
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=True) #print with 256, for the concatenation.
            #latent : B*18*512

            new_latent = torch.zeros((latent.shape[0]//2, latent.shape[1],latent.shape[2])).to('cuda')
            for num in range(0,len(latent),2):
                for x in range(18):
                    new_latent[num//2][x]=latent[num][x] if x > 4 else latent[num+1][x]
                    new_latent[num//2][x]=latent[num][x] if x < 7 else latent[num+1][x]

        print('Finish Calculating Latents')
        # latent = blend(image_list).to('cuda')
        output, _ = net.decoder([new_latent], input_is_latent=True, randomize_noise=None, return_latents=False)
        for i in range(len(output)):
            im = tensor2im(output[i])
            im.save('./mix_middle_interestyle/mix_{}.png'.format(new_count))
            new_count+=1
            if new_count > 10001:
                return False, new_count
    return True, new_count
    
    shutil.rmtree(opts.temporary_path)

if __name__ == '__main__':
    run(1)
