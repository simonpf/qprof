from qprof.input_data import BinInputData
from qprof.driver import RetrievalDriver

def test_bin_retrieval(test_data):
    """
    Ensure that bin data is correctly converted to retrieval input.
    """

    input_file = test_data["bin_file"]
    folder = input_file.parent

    driver = RetrievalDriver(folder, "*.bin", folder, RetrievalDriver)
    driver.run()

    assert len(driver.processed) > 0
