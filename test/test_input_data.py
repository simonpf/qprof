"""
Tests for the qprof.input_data module.
"""
import numpy as np
import pytest

from qprof.models import get_normalizer, get_model
from qprof.input_data import BinInputData


NETCDF4_AVAILABLE = False
try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    pass


def test_bin_input_data(test_data):
    """
    Ensure that bin data is correctly converted to retrieval input.
    """
    normalizer = get_normalizer()
    input_file = BinInputData(test_data["bin_file"],
                              normalizer)

    batch = input_file.get_batch()
    batch = normalizer.invert(batch)

    bin_data = input_file.bin_file.handle

    assert np.all(np.isclose(batch[:, :3],
                             bin_data["brightness_temperatures"][:, :3]))
    assert np.all(np.isclose(batch[:, 15],
                             bin_data["two_meter_temperature"]))
    assert np.all(np.isclose(batch[:, 16],
                             bin_data["total_column_water_vapor"]))

    st = np.where(batch[:, 17:36])[1]
    assert np.all(np.isclose(st, input_file.bin_file.surface_type))

    at = np.where(batch[:, 36:])[1]
    assert np.all(np.isclose(at, input_file.bin_file.airmass_type))


@pytest.mark.skipif(not NETCDF4_AVAILABLE, reason="netCDF4 package missing.")
def test_bin_retrieval(test_data):
    """
    Ensure that bin data is correctly converted to retrieval input.
    """
    normalizer = get_normalizer()
    model = get_model()

    input_file = test_data["bin_file"]
    folder = input_file.parent
    input_data = BinInputData(input_file,
                              normalizer)
    results = input_data.run_retrieval(model)
    retrieval_file = input_data.write_retrieval_results(folder, results)

    assert retrieval_file.name[:-2] == input_file.name[:-3]

    data = netCDF4.Dataset(retrieval_file)
    y = data["truth"][:].data
    assert np.all(np.isclose(input_data.bin_file.handle["surface_precip"], y))

