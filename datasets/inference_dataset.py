from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class InferenceDataset(Dataset):

	def __init__(self, root, opts, return_name=False, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts
		self.return_name = return_name

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		if self.transform:
			from_im = self.transform(from_im)
		if self.return_name==False:
			return from_im
		else:
			name = from_path.split('/')[-1]
			return from_im,name