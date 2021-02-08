import argparse
from pathlib import Path

from qprof.input_data import BinInputData
from qprof.driver import RetrievalDriver

###############################################################################
# Parse arguments
###############################################################################

parser = argparse.ArgumentParser(description='Run qprof retrieval on GPROF bin files.')
parser.add_argument('input_path', metavar='input_folder', type=str, nargs=1,
                    help='Path to the folder containing the input bin files.')
parser.add_argument('output_path', metavar='output_folder', type=str, nargs=1,
                    help='Path to the folder to which to store the output files.')
parser.add_argument('--n_procs', type=int, nargs=1, default=1,
                    help='How many worker processes to use.')
args = parser.parse_args()

input_path = Path(args.input_path[0])
output_path = Path(args.output_path[0])
n_workers = args.n_procs

###############################################################################
# Start processing
###############################################################################

driver = RetrievalDriver(input_path,
                         "*.bin",
                         output_path,
                         BinInputData,
                         n_workers=n_workers)
driver.run()
