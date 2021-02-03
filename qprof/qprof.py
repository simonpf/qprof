"""
QPROF command line interface.

This script provides the command line interface for QPROF, the
 neural-network-based implementation of the Goddard Profiling Algorithm.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import logging
from pathlib import Path

from qprof.models import get_normalizer, get_model
from qprof.input_data import InputData

LOGGER = logging.getLogger()


def main():
    """
    The qprof command line application.
    """

    # Turn off warnings.

    ###########################################################################
    # Parse input arguments
    ###########################################################################

    parser = argparse.ArgumentParser(description='Run QPROF retrieval.')
    parser.add_argument('input_file', metavar='input_file', type=str, nargs=1,
                        help="The preprocessor file with the input data for"
                        "the retrieval.")
    parser.add_argument('output_file', metavar='output_file', type=str,
                        nargs=1, help="Name of the output file.")
    parser.add_argument('-v', action='store_true')

    args = parser.parse_args()
    input_file = args.input_file[0]
    output_file = args.output_file[0]
    verbose = args.v

    ###########################################################################
    # Configure logging
    ###########################################################################

    logging.basicConfig(format='%(message)s')
    if verbose:
        LOGGER.setlevel(logging.INFO)

    input_file = Path(input_file)
    if not input_file.exists():
        LOGGER.error("Error: The provided input file does not exist.")
        return 1

    ###########################################################################
    # Run retrieval
    ###########################################################################

    normalizer = get_normalizer()
    model = get_model()
    input_data = InputData(input_file, normalizer, scans_per_batch=512)

    results = input_data.run_retrieval(model)
    total_scans = input_data.n_scans
    total_pixels = input_data.n_pixels * total_scans
    LOGGER.info(f"Retrieved precipitation for {total_pixels} in {total_scans} "
                " scans.")

    retrieval_results = input_data.write_retrieval_results(output_file,
                                                           results)
    LOGGER.info(f"Wrote retrieval results to {str(retrieval_results)}.")
