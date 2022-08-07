import torch
import torchvision
# print(torchvision.__version__)

import torchmetrics.image.ssim

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image
import numpy as np
from torchvision import transforms
psnr = PeakSignalNoiseRatio()
ssim = StructuralSimilarityIndexMeasure()
import torch
preds = Image.open(r"C:\MyDataset\test_dataset\ITS\gt\1400.png")
target = Image.open(r"C:\MyDataset\test_dataset\ITS\haze\1400_1.png")
preds = np.array(preds)
target = np.array(target)
preds = torch.tensor(preds)

target = torch.tensor(target)


# preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
#
# target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])

mypsnr = psnr(preds, target)
# print(preds.shape())

print(preds.size())
# myssim = ssim(preds, target)
print(mypsnr)
# print(myssim)

#
# my_psnr = psnr(r"C:\MyDataset\test_dataset\ITS\gt\1400.png",r"C:\MyDataset\test_dataset\ITS\haze\1400_1.png")