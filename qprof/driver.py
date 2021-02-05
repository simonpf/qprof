"""
============
qprof.driver
============

This module implements a generic retrieval driver to process multiple input files
concurrently.
"""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue
import queue

from tqdm import tqdm

from qprof.input_data import BinInputData
from qprof.models import get_normalizer, get_model


###############################################################################
# Retrieval worker class
###############################################################################

class Worker(Process):
    """
    A retrieval worker process.

    This class implements the basic work functionality: On creation
    it loads the neural network model and normalizer. When started,
    in continues processing the input files from the input queue until
    it is empty.
    """
    def __init__(self,
                 output_path,
                 input_queue,
                 done_queue,
                 input_class):
        """
        Create new worker.

        Args:
            output_path: The folder to which to write the retrieval results.
            input_queue: The queue from which the input files are taken
            done_queue: The queue onto which the processed files are placed.
            input_class: The class used to read and process the input data.
        """
        super().__init__()
        self.output_path = output_path
        self.input_queue = input_queue
        self.done_queue = done_queue
        self.input_class = input_class

        self.normalizer = get_normalizer()
        self.model = get_model()

    def run(self):
        """
        Start the process.
        """
        while True:
            try:
                input_file = self.input_queue.get(False)
                print("Processing: ", input_file)
                input_data = BinInputData(input_file, self.normalizer)
                results = input_data.run_retrieval(self.model)
                retrieval_file = input_data.write_retrieval_results(
                    self.output_path,
                    results
                    )
                self.done_queue.put(retrieval_file)
                print("Done!")
            except queue.Empty:
                break


###############################################################################
# The retrieval driver
###############################################################################

class RetrievalDriver:
    """
    Generic class to run a concurrently on a range of input files.

    Attributes:
        processed: List to which the paths of the files containing the retrieval
            results are stored.
    """
    def __init__(self,
                 path,
                 pattern,
                 output_path,
                 input_class,
                 n_workers=4):
        """
        Create retrieval driver.

        Args:
            path: The folder containing the input files.
            pattern: glob pattern to use to subselect input files.
            output_path: The path to which to write the retrieval
                 results
            input_class: The class to use to read and process the input files.
            n_workers: The number of worker processes to use.
        """

        self.path = path
        self.pattern = pattern

        self._fill_input_queue()
        self.done_queue = Queue()
        self.workers = [Worker(output_path,
                               self.input_queue,
                               self.done_queue,
                               input_class) for i in range(n_workers)]
        self.processed = []

    def _fill_input_queue(self):
        """
        Scans the input folder for matching files and fills the input queue.
        """
        self.input_queue = Queue()

        for f in self.path.iterdir():
            if f.match(self.pattern):
                print("putting on queue: ", f)
                self.input_queue.put(f)

    def run(self):
        """
        Start the processing.

        This will start processing all suitable input files that have been found and
        stores the names of the produced result files in the ``processed`` attribute
        of the driver.
        """
        if len(self.processed) > 0:
            print("This driver already ran.")
            return

        n_files = self.input_queue.qsize()
        [w.start() for w in self.workers]
        for i in tqdm(range(n_files)):
            self.processed.append(self.done_queue.get(True))
