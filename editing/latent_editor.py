import torch
from utils.common import tensor2im


class LatentEditor(object):

    def __init__(self, stylegan_generator):
        self.generator = stylegan_generator
        self.interfacegan_directions = {
            'age': torch.load('editing/interfacegan_directions/age.pt').cuda(),
            'smile': torch.load('editing/interfacegan_directions/smile.pt').cuda(),
            'pose': torch.load('editing/interfacegan_directions/pose.pt').cuda(),
            'eyeglass': torch.load('editing/interfacegan_directions/eyeglass.pt').cuda(),
            'age1.5': torch.load('editing/interfacegan_directions/age-1.5eyeglass.pt').cuda(),
            'deage1.5': torch.load('editing/interfacegan_directions/deage1.5.pt').cuda(),
            'deage': torch.load('editing/interfacegan_directions/deage.pt').cuda(),
            'age2': torch.load('editing/interfacegan_directions/age-eyeglass.pt').cuda(),
            'age3': torch.load('editing/interfacegan_directions/age-2eyeglass.pt').cuda()
        }

    def layerwise_adding(self,latents,direction,f,factor=0.2,w=[1 for i in range(18)]):
        edit_latent=torch.empty(latents.shape).cuda()
        # latents : B*18*512
        for i,latent in enumerate(latents):
            for idx in range(latent.shape[0]):
                edit_latent[i][idx] = latents[i][idx]+f*factor*w[idx]*direction
        return edit_latent

    def apply_interfacegan(self, latents, direction, factor=0.5, factor_range=None):
        edit_latents = []
        direction = self.interfacegan_directions[direction]
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                # edit_latent = latents + f * factor * direction
                # v2 : edit_latent=self.layerwise_adding(latents,direction,f,factor,w=[0 for i in range(4)]+[1 for i in range(14)])
                # v3 : edit_latent=self.layerwise_adding(latents,direction,f,factor,w=[0 for i in range(5)]+[1 for i in range(13)])
                # v4 : edit_latent=self.layerwise_adding(latents,direction,f,factor,w=[0 for i in range(6)]+[1 for i in range(12)])
                # edit_latent=self.layerwise_adding(latents,direction,f,factor,w=[0 for i in range(4)]+[1 for i in range(14)])
                # edit_latent=self.layerwise_adding(latents,direction,f,factor,w=[1 for i in range(4)]+[0 for i in range(14)])
                edit_latent=self.layerwise_adding(latents,direction,f,factor,w=[0 for i in range(4)]+[1 for i in range(14)])
                edit_latents.append(edit_latent)
            edit_latents = torch.stack(edit_latents).transpose(0, 1)
        else:
            edit_latents = latents + factor * direction
        return self._latents_to_image(edit_latents)

    def _latents_to_image(self, all_latents):
        sample_results = {}
        with torch.no_grad():
            for idx, sample_latents in enumerate(all_latents):
                images, _ = self.generator([sample_latents], randomize_noise=False, input_is_latent=True)
                sample_results[idx] = [tensor2im(image) for image in images]
        return sample_results
