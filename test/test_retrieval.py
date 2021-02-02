"""
Test running the full retrieval on test data.
"""
from qprof.models import get_normalizer, get_model
from qprof.input_data import InputData

def test_retrieval(test_data):

    normalizer = get_normalizer()
    model = get_model()

    input_file = test_data["preprocessor_file"]
    folder = input_file.parent
    input = InputData(input_file,
                      normalizer,
                      scans_per_batch=8)
    results = input.run_retrieval(model)
    retrieval_results = input.write_retrieval_results(folder, results)

    print(retrieval_results)
    print(test_data["retrieval_file"])




