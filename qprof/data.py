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
    def __init__(self,
                 path,
                 batch_size = None,
                 normalize_rain_rates = False,
                 mode = "training"):
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
        days = self.file.variables["day"][:]

        self.mode = mode
        self.train_indices = np.where(days > 1)[0]
        np.random.shuffle(self.train_indices)
        self.test_indices = np.where(days <= 1)[0]

        if self.mode == "training":
            self.indices = self.train_indices
        else:
            self.indices = self.test_indices
        self.size = self.indices.size
        self.size = 10000000

        self.n_samples = self.indices.size
        self.mins = self.file["tbs_min"][:]
        self.maxs = self.file["tbs_max"][:]
        self.batch_size = batch_size

        self.normalize_rain_rates = normalize_rain_rates
        if self.normalize_rain_rates:
            self.rr_max = self.file.variables["surface_precipitation"][:].max()
            self.rr_min = self.file.variables["surface_precipitation"][:].min()
            self.rr_mean = self.file.variables["surface_precipitation"][:].mean()
            self.rr_std = self.file.variables["surface_precipitation"][:].std()

    def __len__(self):
        """
        The number of entries in the training data. This is part of the
        pytorch interface for datasets.

        Return:
            int: The number of samples in the data set
        """
        if self.batch_size is None:
            return self.size
        else:
            return self.size // self.batch_size

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if self.batch_size is None:
            bs = 1
        else:
            bs = self.batch_size

        indices = self.indices[i * bs : (i + 1) * bs]
        x = self.file.variables['tbs'][indices, :]
        x = (x - self.mins) / (self.maxs - self.mins)
        y = self.file.variables['surface_precipitation'][indices]
        if self.normalize_rain_rates:
            y = (y - self.rr_mean) / self.rr_std
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

def write_to_file(file, data, subsampling = 0.01):
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

    v_year = file.variables["year"]
    v_month = file.variables["month"]
    v_day = file.variables["day"]
    v_hour = file.variables["hour"]
    v_minute = file.variables["minute"]
    v_second = file.variables["second"]

    i = file.dimensions["samples"].size
    for d in data:

        if np.random.rand() > subsampling:
            continue

        # Check if sample is valid.
        if not check_sample(d):
            continue

        v_year[i] = d[2]
        v_month[i] = d[3]
        v_day[i] = d[4]
        v_hour[i] = d[5]
        v_year[i] = d[6]
        v_second[i] = d[7]

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
    # Also include date.
    file.createVariable("year", "i4", dimensions = ("samples",))
    file.createVariable("month", "i4", dimensions = ("samples",))
    file.createVariable("day", "i4", dimensions = ("samples",))
    file.createVariable("hour", "i4", dimensions = ("samples",))
    file.createVariable("minute", "i4", dimensions = ("samples",))
    file.createVariable("second", "i4", dimensions = ("samples",))

    file["tbs_min"][:] = 1e30
    file["tbs_max"][:] = 0.0

    return file

def extract_data(base_path, file, subsampling = 0.01):
    """
    Extract training data from GPROF binary files for given year, month and day.

    Arguments:
        year(int): The year from which to extract the training data.
        month(int): The month from which to extract the training data.
        day(int): The day for which to extract training data.
        file: File handle of the netCDF4 file into which to store the results.
    """
    files = glob.glob(os.path.join(base_path, "**", "*.dat"))
    for f in tqdm.tqdm(files):
        with open(f, 'rb') as fn:
                data = read_file(fn)
                write_to_file(f, data, subsampling = subsampling)

#def extract_data(year, month, day, file):
#    """
#    Extract training data from GPROF binary files for given year, month and day.
#
#    Arguments:
#        year(int): The year from which to extract the training data.
#        month(int): The month from which to extract the training data.
#        day(int): The day for which to extract training data.
#        file: File handle of the netCDF4 file into which to store the results.
#    """
#    dendrite = os.path.join(os.environ["HOME"], "Dendrite")
#    year = str(year)
#    month = str(month)
#    if len (month) == 1:
#        month = "0" + month
#    day = str(day)
#    if len (day) == 1:
#        day = "0" + day
#    path = os.path.join(dendrite, "UserAreas", "Teo", "GPROFfiles", str(year) + str(month))
#    files = glob.glob(os.path.join(path, "GMI.CSU.20" + year + month + day + "*.dat"))
#
#    for f in tqdm.tqdm(files):
#        with open(f, 'rb') as fn:
#                data = read_file(fn)
#                write_to_file(file, data)
