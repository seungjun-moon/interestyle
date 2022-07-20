from torch.utils.data import Dataset
from PIL import Image, ImageOps
from utils import data_utils
import torch
import random

class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, mask_root=None, target_transform=None, source_transform=None, gaussian_augmentation=False):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		if mask_root != None:
			self.mask_paths = sorted(data_utils.make_dataset(mask_root))
		else:
			self.mask_paths = None
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.gaussian_augmentation = gaussian_augmentation
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		p=random.random()
		from_path = self.source_paths[index]
		to_path = self.target_paths[index]

		from_im = Image.open(from_path).convert('RGB')
		to_im = Image.open(to_path).convert('RGB')
		if p > 0.5:
			from_im=from_im.transpose(Image.FLIP_LEFT_RIGHT)
			to_im=to_im.transpose(Image.FLIP_LEFT_RIGHT)
		if self.gaussian_augmentation:
			from PIL import ImageFilter
			from_im_1 = from_im.filter(ImageFilter.GaussianBlur(radius = 1))
			from_im_2 = from_im.filter(ImageFilter.GaussianBlur(radius = 4))
			from_im_3 = from_im.filter(ImageFilter.GaussianBlur(radius = 9))
			from_im_4 = from_im.filter(ImageFilter.GaussianBlur(radius = 16))

		if self.target_transform:
			to_im = self.target_transform(to_im)
			from_im_1 = self.target_transform(from_im_1)
			from_im_2 = self.target_transform(from_im_2)
			from_im_3 = self.target_transform(from_im_3)
			from_im_4 = self.target_transform(from_im_4)
			### occur error when self.source_transform is not None.

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		from_im_list = torch.stack((from_im_4,from_im_3,from_im_2,from_im_1,from_im))
		if self.mask_paths != None:
			mask_path = self.mask_paths[index]
			mask = Image.open(mask_path).convert('RGB')
			if p > 0.5:
				mask=mask.transpose(Image.FLIP_LEFT_RIGHT)
			mask = self.target_transform(mask)
			if not self.gaussian_augmentation:
				return from_im, to_im, mask
			else:
				return from_im_list, to_im, mask

		if not self.gaussian_augmentation:
			return from_im, to_im
		else:
			return from_im_list, to_im
