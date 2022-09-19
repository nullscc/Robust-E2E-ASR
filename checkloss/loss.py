import numpy as np
import torch
from warpctc_pytorch import CTCLoss

float_out = torch.tensor(np.load("npy/float_out.npy"))
targets = torch.tensor(np.load("npy/targets.npy"))
output_sizes = torch.tensor(np.load("npy/output_sizes.npy"))
target_sizes = torch.tensor(np.load("npy/target_sizes.npy"))

ctc_loss = CTCLoss()
print(ctc_loss(float_out, targets, output_sizes, target_sizes))
