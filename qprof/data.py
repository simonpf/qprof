import glob
import numpy as np
import netCDF4
import os
import tqdm
from torch.utils.data import Dataset

################################################################################
# Torch dataset
################################################################################

class RainRates(Dataset):
    def __init__(self, path):

        if part == "training":
            folder = "training_data"
        else:
            folder = "test_data"

        try:
            path = os.environ["GPROF_DATA_PATH"]
        except:
            home = os.environ["HOME"]
            path = os.path.join(home, "Dendrite", "UserAreas", "Teo", "qprof")
            print("No environment variable 'GPROF_DATA_PATH' found, using ${HOME}/Dendrite/...")

        self.input_files = glob.glob(os.path.join(path, "input", "*"))
        self.output_files = glob.glob(os.path.join(path, "output", "*"))

types =  [('nx', 'i4'), ('ny', 'i4')]
types += [('year', 'i4'), ('month', 'i4'), ('day', 'i4'), ('hour', 'i4'), ('minute', 'i4'), ('second', 'i4')]
types +=  [('lat', 'f4'), ('lon', 'f4')]
types += [('sfccode', 'i4'), ('tcwv', 'i4'), ('T2m', 'i4')]
types += [('Tb_{}'.format(i), 'f4') for i in range(13)]
types += [('sfcprcp', 'f4'), ('cnvprcp', 'f4')]

################################################################################
# Read GPROF binary data
################################################################################

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

    i = file.dimensions["samples"].size
    for d in data:
        v_lats[i] = d[8]
        v_lons[i] = d[9]
        v_sfccode[i] = d[10]
        v_tcwv[i] = d[11]
        v_t2m[i] = d[12]
        for j in range(13):
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
    file.createVariable("lats", "f4", dimensions = ("samples",))
    file.createVariable("lons", "f4", dimensions = ("samples",))
    file.createVariable("sfccode", "f4", dimensions = ("samples",))
    file.createVariable("tcwv", "f4", dimensions = ("samples",))
    file.createVariable("T2m", "f4", dimensions = ("samples",))
    file.createVariable("surface_precipitation", "f4", dimensions = ("samples",))
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
