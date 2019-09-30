import glob
import os
import numpy as np
import tqdm
from qprof.data import create_output_file, extract_data

dendrite = os.path.join(os.environ["HOME"], "Dendrite")
path = os.path.join(dendrite, "UserAreas", "Simon", "gprof", "test_data.nc")

days = np.arange(1, 2)
months = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
years = [14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15]

file = create_output_file(path)
for y, m in zip(years, months):
    print("Processing {}/{}".format(y, m))
    for d in tqdm.tqdm(days):
        extract_data(y, m, d, file)
file.close()
