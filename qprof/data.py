"""
qprof.data
==========

This module provides a pytorch-compatible interface to the qprof training data.
"""
import glob
import numpy as np
import netCDF4
import os
import tqdm
import torch
from torch.utils.data import Dataset

################################################################################
# Torch dataset
################################################################################

class RainRates(Dataset):
    """
    Pytorch dataset for the GProf training data.

    This class is a wrapper around the netCDF4 files that are used to store
    the GProf training data. It provides as input vector the brightness
    temperatures and as output vector the surface precipitation.
    """
    def __init__(self, path):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: The path to the netCDF4 file with the training data
        """

        super(RainRates, self).__init__()

        try:
            qprof_path = os.environ["QPROF_DATA_PATH"]
        except:
            qprof_path = os.path.join(os.environ["HOME"],
                                      "src",
                                      "qprof",
                                      "data")

        if os.path.isabs(path):
            path = path
        else:
            path = os.path.join(qprof_path, path)

        self.file = netCDF4.Dataset(path, mode = "r")
        self.n_samples = self.file.dimensions["samples"].size

        self.mins = self.file["tbs_min"][:]
        self.maxs = self.file["tbs_max"][:]

    def __len__(self):
        """
        The number of entries in the training data. This is part of the
        pytorch interface for datasets.

        Return:
            int: The number of samples in the data set
        """
        return self.n_samples

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        x = self.file.variables['tbs'][i, :]
        x = (x - self.mins) / (self.maxs - self.mins)
        y = self.file.variables['surface_precipitation'][i]
        return torch.tensor(x), torch.tensor(y)

################################################################################
# Read GPROF binary data
################################################################################

types =  [('nx', 'i4'), ('ny', 'i4')]
types += [('year', 'i4'), ('month', 'i4'), ('day', 'i4'), ('hour', 'i4'), ('minute', 'i4'), ('second', 'i4')]
types +=  [('lat', 'f4'), ('lon', 'f4')]
types += [('sfccode', 'i4'), ('tcwv', 'i4'), ('T2m', 'i4')]
types += [('Tb_{}'.format(i), 'f4') for i in range(13)]
types += [('sfcprcp', 'f4'), ('cnvprcp', 'f4')]

def read_file(f):
    """
    Read GPM binary file.

    Arguments:
        f(str): Filename of file to read

    Returns:
        numpy memmap object pointing to the data.
    """
    data = np.memmap(f,
                     dtype = types,
                     mode =  "r",
                     offset = 10 + 26 * 4)
    return data

def check_sample(data):
    """
    Check that brightness temperatures are within a valid range.

    Arguments:
        data: The data array containing the data of one training sample.

    Return:
        Whether the given sample contains valid tbs.
    """
    return all([data[i] > 0 and data[i] < 1000 for i in range(13, 26)])

def write_to_file(file, data):
    """
    Write data to NetCDF file.

    Arguments
        file: File handle to the output file.
        data: numpy memmap object pointing to the binary data
    """
    v_tbs = file.variables["tbs"]
    v_lats = file.variables["lats"]
    v_lons = file.variables["lons"]
    v_sfccode = file.variables["sfccode"]
    v_tcwv = file.variables["tcwv"]
    v_t2m = file.variables["T2m"]
    v_surf_precip = file.variables["surface_precipitation"]
    v_tbs_min = file.variables["tbs_min"]
    v_tbs_max = file.variables["tbs_max"]

    i = file.dimensions["samples"].size
    for d in data:

        # Check if sample is valid.
        if not check_sample(d):
            continue

        v_lats[i] = d[8]
        v_lons[i] = d[9]
        v_sfccode[i] = d[10]
        v_tcwv[i] = d[11]
        v_t2m[i] = d[12]
        for j in range(13):
            v_tbs_min[j] = np.minimum(v_tbs_min[j], d[13 + j])
            v_tbs_max[j] = np.maximum(v_tbs_max[j], d[13 + j])
            v_tbs[i, j] = d[13 + j]
        v_surf_precip[i] = d[26]
        i += 1

def create_output_file(path):
    """
    Creates netCDF4 output file to store training data in.

    Arguments:
        path: Filename of the file to create.

    Returns:
        netCDF4 Dataset object pointing to the created file
    """
    file = netCDF4.Dataset(path, "w")
    file.createDimension("channels", size = 13)
    file.createDimension("samples", size = None) #unlimited dimensions
    file.createVariable("tbs", "f4", dimensions = ("samples", "channels"))
    file.createVariable("tbs_min", "f4", dimensions = ("channels"))
    file.createVariable("tbs_max", "f4", dimensions = ("channels"))
    file.createVariable("lats", "f4", dimensions = ("samples",))
    file.createVariable("lons", "f4", dimensions = ("samples",))
    file.createVariable("sfccode", "f4", dimensions = ("samples",))
    file.createVariable("tcwv", "f4", dimensions = ("samples",))
    file.createVariable("T2m", "f4", dimensions = ("samples",))
    file.createVariable("surface_precipitation", "f4", dimensions = ("samples",))

    file["tbs_min"][:] = 1e30
    file["tbs_max"][:] = 0.0

    return file

def extract_data(year, month, day, file):
    """
    Extract training data from GPROF binary files for given year, month and day.

    Arguments:
        year(int): The year from which to extract the training data.
        month(int): The month from which to extract the training data.
        day(int): The day for which to extract training data.
        file: File handle of the netCDF4 file into which to store the results.
    """
    dendrite = os.path.join(os.environ["HOME"], "Dendrite")
    year = str(year)
    month = str(month)
    if len (month) == 1:
        month = "0" + month
    day = str(day)
    if len (day) == 1:
        day = "0" + day
    path = os.path.join(dendrite, "UserAreas", "Teo", "GPROFfiles", str(year) + str(month))
    files = glob.glob(os.path.join(path, "GMI.CSU.20" + year + month + day + "*.dat"))

    for f in tqdm.tqdm(files):
        with open(f, 'rb') as fn:
                data = read_file(fn)
                write_to_file(file, data)
