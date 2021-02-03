"""
===============================
qprof.file_formats.preprocessor
===============================

This module provides an interface class to read GPROF preprocessor files.
"""
import logging
from datetime import datetime
import re

import numpy as np
import xarray

from qprof.file_formats import retrieval
from pathlib import Path

LOGGER = logging.getLogger(__name__)

###############################################################################
# Struct types
###############################################################################

N_SPECIES = 5
N_TEMPERATURES = 12
N_LAYERS = 28
N_PROFILES = 80
N_CHANNELS = 15

DATE_TYPE = np.dtype(
    [("year", "i2"),
     ("month", "i2"),
     ("day", "i2"),
     ("hour", "i2"),
     ("minute", "i2"),
     ("second", "i2")]
)

ORBIT_HEADER_TYPES = np.dtype(
    [("satellite", "a12"),
     ("sensor", "a12"),
     ("preprocessor", "a12"),
     ("profile_database_file", "a128"),
     ("radiometer_file", "a128"),
     ("calibration_file", "a128"),
     ("granule_number", "i"),
     ("number_of_scans", "i"),
     ("number_of_pixels", "i"),
     ("n_channels", "i"),
     ("frequencies", f"{N_CHANNELS}f4"),
     ("comment", "a40")]
)

SCAN_HEADER_TYPES = np.dtype(
    [("scan_date", DATE_TYPE),
     ("scan_latitude", "f4"),
     ("scan_longitude", "f4"),
     ("scan_altitude", "f4"),
     ])

DATA_RECORD_TYPES = np.dtype(
    [("latitude", "f4"),
     ("longitude", "f4"),

     ("brightness_temperatures", f"{N_CHANNELS}f4"),
     ("earth_incidence_angle", f"{N_CHANNELS}f4"),
     ("wet_bulb_temperature", f"f4"),
     ("lapse_rate", f"f4"),
     ("total_column_water_vapor", f"f4"),
     ("surface_temperature", f"f4"),
     ("two_meter_temperature", f"f4"),

     ("quality_flag", f"i"),
     ("sunglint_angle", f"i1"),
     ("surface_type", f"i1"),
     ("airmass_type", f"i2")]
    )

###############################################################################
# Preprocessor file class
###############################################################################

class GPROFPreprocessorFile:
    """
    Interface to read CSU preprocessor files.

    Attibutes:
        filename: The path of the source file.
        orbit_header: Numpy structured array containing the orbit header.
        n_scans: The number of scans in the file.
        n_pixels: The number of pixels in the file.
    """
    def __init__(self, filename):
        """
        Read preprocessor file.

        Args:
            filename: Path to the file to read.
        """
        self.filename = filename
        with open(self.filename, "rb") as file:
            self.data = file.read()
        self.orbit_header = np.frombuffer(self.data,
                                          ORBIT_HEADER_TYPES,
                                          count=1)
        self.n_scans = self.orbit_header["number_of_scans"][0]
        self.n_pixels = self.orbit_header["number_of_pixels"][0]

    @property
    def satellite(self):
        """
        The satellite from which the data in this file originates.
        """
        return self.orbit_header["satellite"].tobytes().decode().strip()

    @property
    def sensor(self):
        """
        The sensor from which the data in this file originates.
        """
        return self.orbit_header["sensor"].tobytes().decode().strip()

    @property
    def start_date(self):
        """
        The scan date of the first header in the granule.
        """
        header = self.get_scan_header(0)
        year = header["scan_date"]["year"][0]
        month = header["scan_date"]["month"][0]
        day = header["scan_date"]["day"][0]
        hour = header["scan_date"]["hour"][0]
        minute = header["scan_date"]["minute"][0]
        second = header["scan_date"]["second"][0]
        return datetime(year, month, day, hour, minute, second)

    @property
    def end_date(self):
        """
        The scan data of the last header in the granule.
        """
        header = self.get_scan_header(-1)
        year = header["scan_date"]["year"][0]
        month = header["scan_date"]["month"][0]
        day = header["scan_date"]["day"][0]
        hour = header["scan_date"]["hour"][0]
        minute = header["scan_date"]["minute"][0]
        second = header["scan_date"]["second"][0]
        return datetime(year, month, day, hour, minute, second)

    def __repr__(self):
        """
        Represent GPROFPreprocessorFile as string.
        """
        s = f"GPROFPreprocessorFile(satellite={self.satellite}, "
        s += f"sensor={self.sensor}, start_date={self.start_date}, "
        s += f"end_date={self.end_date})"
        return s

    @property
    def scans(self):
        """
        Iterates of the scans in the file. Each scan is returned as Numpy
        structured array of size n_pixels and dtype DATA_RECORD_TYPES.
        """
        for i in range(self.n_scans):
            yield self.get_scan(i)

    def get_scan(self, i):
        """
        Args:
            i: The index of the scan to return.
        Returns:
            The ith scan in the file as numpy structured array of size n_pixels
            and dtype DATA_RECORD_TYPES.
        """
        if i < 0:
            i = self.n_scans + i

        offset = ORBIT_HEADER_TYPES.itemsize
        offset += i * (SCAN_HEADER_TYPES.itemsize
                       + self.n_pixels * DATA_RECORD_TYPES.itemsize)
        offset += SCAN_HEADER_TYPES.itemsize
        return np.frombuffer(self.data,
                             DATA_RECORD_TYPES,
                             count=self.n_pixels,
                             offset=offset)

    def get_scan_header(self, i):
        """
        Args:
            i: The index of the scan to return.

        Returns:
            The header of the ith scan in the file as numpy structured array
            of size n_pixels and dtype DATA_RECORD_TYPES.
        """
        if i < 0:
            i = self.n_scans + i

        offset = ORBIT_HEADER_TYPES.itemsize
        offset += i * (SCAN_HEADER_TYPES.itemsize
                       + self.n_pixels * DATA_RECORD_TYPES.itemsize)
        return np.frombuffer(self.data,
                             SCAN_HEADER_TYPES,
                             count=1,
                             offset=offset)

    def to_xarray_dataset(self):
        """
        Return data in file as xarray dataset.
        """
        data = {k: np.zeros((self.n_scans, self.n_pixels) + d[0].shape)
                for k, d in DATA_RECORD_TYPES.fields.items()}
        for i, s in enumerate(self.scans):
            for k, d in data.items():
                d[i] = s[k]

        dims = ["scans", "pixels", "channels"]
        data = {k: (dims[:len(d.shape)], d) for k, d in data.items()}
        return xarray.Dataset(data)

    def write_retrieval_results(self, path, results):
        """
        Write retrieval result to GPROF binary format.

        Args:
            path: The folder to which to write the result. The filename
                  itself follows the GPORF naming scheme.
            results: Dictionary containing the retrieval results.

        Returns:

            Path object to the created binary file.
        """
        filename = Path(path)
        if filename.is_dir():
             filename = filename / self._get_retrieval_filename()

        with open(filename, "wb") as file:
            self._write_retrieval_orbit_header(file)
            self._write_retrieval_profile_info(file)
            for i in range(self.n_scans):
                self._write_retrieval_scan_header(file, i)

                precip_mean = results["precip_mean"][i]
                precip_1st_tertial = results["precip_1st_tertial"][i]
                precip_3rd_tertial = results["precip_3rd_tertial"][i]
                pop = results["precip_pop"][i]

                self._write_retrieval_scan(file,
                                           precip_mean,
                                           precip_1st_tertial,
                                           precip_3rd_tertial,
                                           pop)
        return filename

    def _get_retrieval_filename(self):
        """
        Produces GPROF compliant filename from retrieval results dict.
        """
        start_date = self.get_scan_header(0)["scan_date"]
        end_date = self.get_scan_header(-1)["scan_date"]

        filename = Path(self.filename).name


        name = "2A.QCORE.GMI.V7."

        year, month, day = [start_date[k][0] for k in ["year", "month", "day"]]
        name += f"{year:02}{month:02}{day:02}-"

        hour, minute, second = [start_date[k][0] for k in ["hour", "minute", "second"]]
        name += f"S{hour:02}{minute:02}{second:02}-"

        hour, minute, second = [end_date[k][0] for k in ["hour", "minute", "second"]]
        name += f"E{hour:02}{minute:02}{second:02}."

        granule_number = self.orbit_header["granule_number"][0]
        name += f"{granule_number:06}.BIN"

        return name

    def _write_retrieval_orbit_header(self, file):
        """
        Writes the retrieval orbit header to an opened binary file..

        Args:
            file: Handle to the binary file to write the data to.
        """
        new_header = np.recarray(1, dtype=retrieval.ORBIT_HEADER_TYPES)
        for k in retrieval.ORBIT_HEADER_TYPES.fields:
            if not k in self.orbit_header.dtype.fields:
                continue
            new_header[k] = self.orbit_header[k]

        new_header["algorithm"] = "QPROF"
        new_header.tofile(file)

    def _write_retrieval_profile_info(self, file):
        """
        Write the retrieval profile info to an opened binary file.

        Args:
            file: Handle to the binary file to write the data to.
        """
        profile_info = np.recarray(1, dtype=retrieval.PROFILE_INFO_TYPES)
        profile_info.tofile(file)

    def _write_retrieval_scan_header(self, file, i):
        """
        Write the scan header corresponding to the ith header in the file
        to a given file stream.

        Args:
            file: Handle to the binary file to write the data to.
            i: The index of the scan for which to write the header.
        """
        header = self.get_scan_header(i)
        scan_header = np.recarray(1, dtype=retrieval.SCAN_HEADER_TYPES)
        scan_header["scan_latitude"] = header["scan_latitude"]
        scan_header["scan_longitude"] = header["scan_longitude"]
        scan_header["scan_altitude"] = header["scan_altitude"]
        scan_header["scan_date"]["year"] = header["scan_date"]["year"]
        scan_header["scan_date"]["month"] = header["scan_date"]["month"]
        scan_header["scan_date"]["day"] = header["scan_date"]["day"]
        scan_header["scan_date"]["hour"] = header["scan_date"]["hour"]
        scan_header["scan_date"]["minute"] = header["scan_date"]["minute"]
        scan_header["scan_date"]["second"] = header["scan_date"]["second"]
        scan_header["scan_date"]["millisecond"] = 0.0
        scan_header.tofile(file)

    def _write_retrieval_scan(self,
                              file,
                              precip_mean,
                              precip_1st_tertial,
                              precip_3rd_tertial,
                              precip_pop):
        """
        Write retrieval data from a full scan to a binary stream.


        Args:
            file: Handle to the binary file to write the data to.
            precip_mean: 1D array containing the mean retrieved precipitation for
                 each pixel.
            precip_1st_tertial: 1D array containing the 1st tertial retrieved from the data.
            precip_3rd_tertial: 1D array containing the 3rd tertial retrieved from the data
            precip_pop: 1D array containing the probability of precipitation in the scan.
        """
        n_pixels = precip_mean.shape[-1]
        pixels = np.recarray(n_pixels, dtype=retrieval.DATA_RECORD_TYPES)

        pixels["surface_precip"] = precip_mean
        pixels["precip_1st_tertial"] = precip_1st_tertial
        pixels["precip_3rd_tertial"] = precip_3rd_tertial
        pixels["pop_index"] = precip_pop.astype(np.dtype("i1"))
        pixels.tofile(file)
