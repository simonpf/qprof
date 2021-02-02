"""
This module tests the interface for reading bin files.
"""
import numpy as np
import pytest

from qprof.file_formats.bin import (GPROFGMIBinFile,
                                    FileProcessor)

NETCDF4_AVAILABLE = False
try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    pass


def test_bin_reading(test_data):
    """
    Test reading of binary file format on a test file and ensures that the data
    is consistent with what is expected for the bin.
    """
    file = GPROFGMIBinFile(test_data["bin_file"])

    # Check that all t2m values in bin match expected value.
    t2m = file.handle["two_meter_temperature"]
    t2m_bin = file.bin_temperature
    assert np.all((t2m > t2m_bin - 0.5) * (t2m < t2m_bin + 0.5))

    # Check that all tcwv values in bin match expected value.
    tcwv = file.handle["total_column_water_vapor"]
    tcwv_bin = file.tcwv
    assert np.all((tcwv > tcwv_bin - 0.5) * (tcwv < tcwv_bin + 0.5))

    # Check that profile order is random but constant.
    assert file.indices[0] == 21

    assert file.surface_type == 17
    assert file.airmass_type == 2

@pytest.mark.skipif(not NETCDF4_AVAILABLE, reason="netCDF4 package missing.")
def test_processing(test_data):

    input_file = test_data["bin_file"]
    input = GPROFGMIBinFile(input_file)
    folder = input_file.parent

    processor = FileProcessor(folder)

    output_1 = folder / "test_00.nc"
    output_2 = folder / "test_01.nc"

    processor.run_async(output_1, 0.0, 0.5, n_processes=1)
    processor.run_async(output_2, 0.5, 1.0, n_processes=1)

    with netCDF4.Dataset(output_1, "r") as output_1:
        with netCDF4.Dataset(output_2, "r") as output_2:
            for v in input.handle.dtype.fields:
                if v in output_1.variables:
                    data_1 = output_1[v][:]
                    data_2 = output_2[v][:]

                    data = np.concatenate([data_1, data_2], axis=0)
                    data_ref = input.handle[v][input.indices]

                    assert np.all(np.isclose(data, data_ref))



