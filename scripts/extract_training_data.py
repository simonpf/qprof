import glob
import os
import numpy as np
import tqdm
from qprof.data import create_output_file, extract_data

dendrite = os.path.join(os.environ["HOME"], "Dendrite")
path = os.path.join(dendrite, "UserAreas", "Simon", "gprof", "training_data_full.nc")
data_path = os.path.join(dendrite, "UserAreas", "Teo", "GPROFfiles")

file = create_output_file(path)
extract_data(data_path, file, subsampling = 0.1)
file.close()
