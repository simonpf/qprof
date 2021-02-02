"""
Tests reading of retrieval binary files.
"""
import pytest

from qprof.file_formats.retrieval import GPROFRetrievalFile

XARRAY_AVAILABLE = False
try:
    import xarray
    XARRAY_AVAILABLE = True
except ImportError:
    pass

def test_reading(test_data):
    """
    Ensures that example retrieval file can be read.
    """
    retrieval_file = GPROFRetrievalFile(test_data["retrieval_file"])

    assert retrieval_file.satellite == "GPM"
    assert retrieval_file.sensor == "GMI"

    start_date = retrieval_file.start_date
    end_date = retrieval_file.end_date

    assert start_date.year == 2019
    assert start_date.month == 1
    assert start_date.day == 1
    assert end_date.year == 2019
    assert end_date.month == 1
    assert end_date.day == 1

@pytest.mark.skipif(not XARRAY_AVAILABLE, reason="xarray package missing.")
def test_conversion_to_xarray(test_data):
    """
    Ensures that example retrieval file can be read.
    """
    retrieval_file = GPROFRetrievalFile(test_data["retrieval_file"])
    data = retrieval_file.to_xarray_dataset()
    surface_precip = data["surface_precip"]

    assert surface_precip.shape[0] == retrieval_file.n_scans
    assert surface_precip.shape[1] == retrieval_file.n_pixels
