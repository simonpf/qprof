import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from cloud_colocations.data import GpmData
from typhon.retrieval.qrnn import QRNN
from qprof.models.unet.pytorch import UNet

################################################################################
# Parse arguments
################################################################################

import argparse

parser = argparse.ArgumentParser(description='Train unet.')
parser.add_argument("training_data", type=str, nargs=1, help="The training data.")
parser.add_argument("levels", type=int, nargs=1, help="Number of downscaling blocks.")
parser.add_argument("n_features", type=int, nargs=1, help="Number of features in network.")

args = parser.parse_args()
training_data = args.training_data[0]
level = args.levels[0]
n_features = args.n_features[0]

################################################################################
# Train network
################################################################################

data = GpmData(training_data)
n = len(data)
training_data, validation_data = torch.utils.data.random_split(data, [int(0.9 * n), n - int(0.9 * n)])
training_loader = DataLoader(training_data, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=16, shuffle=True)

quantiles = [0.05, 0.15, 0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95]
unet = UNet(13, 11, 64, 5)
qrnn = QRNN(13, quantiles, model = unet)

qrnn.train(training_loader, validation_loader, gpu = True, lr = 1e-2,  n_epochs=20)
qrnn.train(training_loader, validation_loader, gpu = True, lr = 5e-3,  n_epochs=20)
qrnn.train(training_loader, validation_loader, gpu = True, lr = 2e-3,  n_epochs=20)
qrnn.model.save("unet.pt")
