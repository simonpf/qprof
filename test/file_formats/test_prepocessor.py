"""
Tests reading of preprocessor binary files.
"""
import pytest

from qprof.file_formats.preprocessor import GPROFPreprocessorFile

XARRAY_AVAILABLE = False
try:
    import xarray
    XARRAY_AVAILABLE = True
except ImportError:
    pass

def test_reading(test_data):
    """
    Ensures that example preprocessor file can be read.
    """
    preprocessor_file = GPROFPreprocessorFile(test_data["preprocessor_file"])

    assert preprocessor_file.satellite == "GPM"
    assert preprocessor_file.sensor == "GMI"

    start_date = preprocessor_file.start_date
    end_date = preprocessor_file.end_date

    assert start_date.year == 2019
    assert start_date.month == 1
    assert start_date.day == 1
    assert end_date.year == 2019
    assert end_date.month == 1
    assert end_date.day == 1

@pytest.mark.skipif(not XARRAY_AVAILABLE, reason="xarray package missing.")
def test_conversion_to_xarray(test_data):
    """
    Ensures that example preprocessor file can be read.
    """
    preprocessor_file = GPROFPreprocessorFile(test_data["preprocessor_file"])
    data = preprocessor_file.to_xarray_dataset()
    bts = data["brightness_temperatures"]

    assert bts.shape[0] == preprocessor_file.n_scans
    assert bts.shape[1] == preprocessor_file.n_pixels
