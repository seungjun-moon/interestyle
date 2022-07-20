import numpy as np
import torch

v1 = torch.load('age.pt').numpy()[0] #feature to manipulate. + 1*512 --> 512 dim
v2 = torch.load('eyeglass.pt').numpy()[0] #feature entangled undesirably

v_norm = np.sqrt(sum(v2**2))

proj_v1 = (np.dot(v1,v2)/v_norm**2)*v2

new_v1= v1-proj_v1

print(np.dot(new_v1, v2))

new_v1 = torch.from_numpy(new_v1).unsqueeze(0)

print(new_v1.shape)

torch.save(new_v1, 'age-eyeglass.pt')