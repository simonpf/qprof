"""
============================
qprof.file_formats.retrieval
============================

This module provides an interface class to read the binary format used for
the results of the GPROF retrievals.
"""
from datetime import datetime
import logging
import gzip

import numpy as np

LOGGER = logging.getLogger(__name__)

N_SPECIES = 5
N_TEMPERATURES = 12
N_LAYERS = 28
N_PROFILES = 80

DATE6_TYPE = np.dtype(
    [("year", "i2"),
     ("month", "i2"),
     ("day", "i2"),
     ("hour", "i2"),
     ("minute", "i2"),
     ("second", "i2")]
)

DATE7_TYPE = np.dtype(
    [("year", "i2"),
     ("month", "i2"),
     ("day", "i2"),
     ("hour", "i2"),
     ("minute", "i2"),
     ("second", "i2"),
     ("millisecond", "i2")]
)

ORBIT_HEADER_TYPES = np.dtype(
    [("satellite", "a12"),
     ("sensor", "a12"),
     ("preprocessor", "a12"),
     ("algorithm", "a12"),
     ("profile_database_file", "a128"),
     ("radiometer_file", "a128"),
     ("creation_date", DATE6_TYPE),
     ("granule_start_date", DATE6_TYPE),
     ("granule_end_date", DATE6_TYPE),
     ("granule_number", "i"),
     ("number_of_scans", "i2"),
     ("number_of_pixels", "i2"),
     ("profile_struct", "i1"),
     ("spares", "a51")]
)

PROFILE_INFO_TYPES = np.dtype(
    [("n_species", "i1"),
     ("n_temps", "i1"),
     ("n_layers", "i1"),
     ("n_profiles", "i1"),
     ("species_description", f"{N_SPECIES}a20"),
     ("height_top_layers", f"{N_LAYERS}f4"),
     ("temperature", f"{N_TEMPERATURES}f4"),
     ("profiles", f"{N_SPECIES * N_TEMPERATURES * N_LAYERS * N_PROFILES}f4")]
)

SCAN_HEADER_TYPES = np.dtype(
    [("scan_latitude", "f4"),
     ("scan_longitude", "f4"),
     ("scan_altitude", "f4"),
     ("scan_date", DATE7_TYPE),
     ("spare", "i2")])

DATA_RECORD_TYPES = np.dtype(
    [("pixel_status", "i1"),
     ("quality_flag", "i1"),
     ("l1c_quality_flag", "i1"),
     ("surface_type_index", "i1"),
     ("tcwv_index", "i1"),
     ("pop_index", "i1"),
     ("t2m_index", "i2"),
     ("airmass_index", "i2"),
     ("sun_glint_angle", "i1"),
     ("precip_flag", "i1"),

     ("latitude", "f4"),
     ("longitude", "f4"),

     ("surface_precip", "f4"),
     ("frozen_precip", "f4"),
     ("convective_precip", "f4"),

     ("rain_water_path", "f4"),
     ("cloud_water_path", "f4"),
     ("ice_water_path", "f4"),

     ("most_likely_precip", "f4"),
     ("precip_1st_tertial", "f4"),
     ("precip_3rd_tertial", "f4"),
     ("profile_t2m_index", "i2"),
     ("profile_number", f"{N_SPECIES}i2"),
     ("profile_scale", f"{N_SPECIES}f4")
     ]
)

class GPROFRetrievalFile:
    """
    This class provides an interface to read and write binary GPROF retrieval
    files.

    One file contains the retrieval results for a single granule of observations.
    The file is organized into a file header followed by a sequence of scans.
    Each scan consists of a file header followed by the retrieval results for
    each pixel in the scan.

    Attributes:
        filename: Path to the binary source file.
        orbit_header: Numpy structured array containing the file header.
        n_scans: The number of scans in the file.
        n_pixels: The number of pixels in each scan.
    """
    def __init__(self, filename):
        """
        Read retrieval file.

        Args:
            filename(``str`` or ``pathlib.Path``): Path to the file to read.
        """
        self.filename = filename
        with gzip.open(filename, "rb") as file:
            self.data = file.read()
        self.orbit_header = np.frombuffer(self.data,
                                          ORBIT_HEADER_TYPES,
                                          count=1)
        self.n_scans = self.orbit_header["number_of_scans"][0]
        self.n_pixels = self.orbit_header["number_of_pixels"][0]

        np.random.seed(self.orbit_header["granule_number"])
        self.scan_indices = np.random.permutation(np.arange(self.n_scans))
        self.pixel_indices = np.random.permutation(np.arange(self.n_pixels))

    @property
    def satellite(self):
        """
        The satellite from which the data originates.
        """
        return self.orbit_header["satellite"].tobytes().decode().strip()

    @property
    def sensor(self):
        """
        The sensor from which the data originates.
        """
        return self.orbit_header["sensor"].tobytes().decode().strip()

    @property
    def start_date(self):
        """
        The start date of the granule.
        """
        year = self.orbit_header["granule_start_date"]["year"][0]
        month = self.orbit_header["granule_start_date"]["month"][0]
        day = self.orbit_header["granule_start_date"]["day"][0]
        hour = self.orbit_header["granule_start_date"]["hour"][0]
        minute = self.orbit_header["granule_start_date"]["minute"][0]
        second = self.orbit_header["granule_start_date"]["second"][0]
        return datetime(year, month, day, hour, minute, second)

    @property
    def end_date(self):
        """
        The end date of the granule.
        """
        year = self.orbit_header["granule_end_date"]["year"][0]
        month = self.orbit_header["granule_end_date"]["month"][0]
        day = self.orbit_header["granule_end_date"]["day"][0]
        hour = self.orbit_header["granule_end_date"]["hour"][0]
        minute = self.orbit_header["granule_end_date"]["minute"][0]
        second = self.orbit_header["granule_end_date"]["second"][0]
        return datetime(year, month, day, hour, minute, second)

    def __repr__(self):
        s = f"GPROFRetrievalFile(satellite={self.satellite}, "
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
        Return pixels in scan as Numpy structured array of size n_pixels and dtype
        DATA_RECORD_TYPES.

        Args:
            i: The index of the scan to return>

        Returns:
            numpy structured array containing the retrieval results in the
            requested scan.
        """
        if i < 0:
            i = self.n_scans + i

        offset = ORBIT_HEADER_TYPES.itemsize + PROFILE_INFO_TYPES.itemsize
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

        offset = ORBIT_HEADER_TYPES.itemsize + PROFILE_INFO_TYPES.itemsize
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
        import xarray
        data = {k: np.zeros((self.n_scans, self.n_pixels) + d[0].shape)
                for k, d in DATA_RECORD_TYPES.fields.items()}
        for i, s in enumerate(self.scans):
            for k, d in data.items():
                d[i] = s[k]

        dims = ["scans", "pixels", "channels"]
        data = {k: (dims[:len(d.shape)], d) for k, d in data.items()}
        return xarray.Dataset(data)
