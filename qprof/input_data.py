"""
================
qprof.input_data
================

This module defines the InputData class, which acts as interface between the
preprocessor data and the neural network model.
"""
import numpy as np
import xarray
import quantnn.quantiles as qq
from torch.utils.data import Dataset
from qprof.file_formats.preprocessor import GPROFPreprocessorFile, N_CHANNELS


class InputData(Dataset):
    """
    This class takes the raw data from a pre-processor file and provides and
    transforms it so that it can be fed into the neural network model. It also
    provides driver functions to execute the retrieval.
    """
    def __init__(self,
                 filename,
                 normalizer,
                 scans_per_batch=4):
        """
        Create a new input data object.

        Args:
            filename: Path to the preprocessor file containing the retrieval
                input data.
            normalizer: Normalizer object to use to normalizer the input data.
            scans_per_batch: How many scans should be combined into a
            batch during processing.
        """
        self.filename = filename
        self.data = GPROFPreprocessorFile(filename).to_xarray_dataset()
        self.normalizer = normalizer

        self.n_scans = self.data.scans.size
        self.n_pixels = self.data.pixels.size
        self.scans_per_batch = scans_per_batch

        self.n_batches = self.n_scans // scans_per_batch
        remainder = self.n_scans // scans_per_batch
        if remainder > 0:
            self.n_batches += 1

        bts = self.data["brightness_temperatures"]
        self.pixel_mask = np.any(np.all(bts < 0.0, axis=0), axis=1)

    def get_batch(self, i):
        """
        Return batch of input pixels to feed into the neural network.

        Args:
            i: The index of the batch to return.

        Returns:
            Input batch of pixel-wise retrieval input data.
        """

        i_start = i * self.scans_per_batch
        i_end = (i + 1) * self.scans_per_batch

        bts = self.data["brightness_temperatures"][i_start:i_end, :, :].data
        bts = bts.reshape(-1, N_CHANNELS)

        # 2m temperature
        t2m = self.data["two_meter_temperature"][i_start:i_end, :].data
        t2m = t2m.reshape(-1, 1)
        # Total precipitable water.
        tcwv = self.data["total_column_water_vapor"][i_start:i_end, :].data
        tcwv = tcwv.reshape(-1, 1)

        # Surface type
        n = bts.shape[0]
        st = self.data["surface_type"][i_start:i_end, :].data
        st = st.reshape(-1, 1).astype(int)
        n_types = 19
        st_1h = np.zeros((n, n_types), dtype=np.float32)
        st_1h[np.arange(n), st.ravel()] = 1.0

        # Airmass type
        am = self.data["airmass_type"][i_start:i_end, :].data
        am = np.maximum(am.reshape(-1, 1).astype(int), 0)
        n_types = 4
        am_1h = np.zeros((n, n_types), dtype=np.float32)
        am_1h[np.arange(n), am.ravel()] = 1.0

        x = np.concatenate([bts, t2m, tcwv, st_1h, am_1h], axis=1)
        return self.normalizer(x)

    def run_retrieval(self, qrnn):
        """
        Run retrieval with given QRNN model.

        Args:
            qrnn: ``quantnn.QRNN`` model to use in the retrieval.

        Returns:
            xarray.Dataset containing the retrieval results.
        """
        quantiles = qrnn.quantiles

        y_pred = np.zeros((self.n_scans, self.n_pixels, len(quantiles)))
        mean = np.zeros((self.n_scans, self.n_pixels))
        first_tertial = np.zeros((self.n_scans, self.n_pixels))
        second_tertial = np.zeros((self.n_scans, self.n_pixels))
        pop = np.zeros((self.n_scans, self.n_pixels))

        for i in range(len(self)):
            x = self[i]
            y = qrnn.predict(x)

            i_start = i * self.scans_per_batch
            i_end = (i + 1) * self.scans_per_batch
            y_pred[i_start:i_end, :, :] = y.reshape(-1, self.n_pixels, len(quantiles))

            means = qq.posterior_mean(y, quantiles, quantile_axis=1)
            mean[i_start:i_end] = means.reshape(-1, self.n_pixels)

            t = qq.posterior_quantiles(y, quantiles, [0.333], quantile_axis=1)
            first_tertial[i_start:i_end] = t.reshape(-1, self.n_pixels)
            t = qq.posterior_quantiles(y, quantiles, [0.666], quantile_axis=1)
            second_tertial[i_start:i_end] = t.reshape(-1, self.n_pixels)

            p =  qq.probability_larger_than(y, quantiles, 0.01, quantile_axis=1)
            pop[i_start:i_end] = p.reshape(-1, self.n_pixels)


        dims = ["scans", "pixels", "quantiles"]

        data = {
            "quantiles": (("quantiles",), quantiles),
            "precip_quantiles": (dims, y_pred),
            "precip_mean": (dims[:2], mean),
            "precip_1st_tertial": (dims[:2], first_tertial),
            "precip_3rd_tertial": (dims[:2], second_tertial),
            "precip_pop": (dims[:2], pop)
        }
        return xarray.Dataset(data)

    def write_retrieval_results(self,
                                path,
                                results):
        """
        Write retrieval results to file.

        Args:
            path: The path of the output file to which to write the retrieval results.
            results: The xarray.Dataset containing the retrieval results.
        """
        preprocessor_file = GPROFPreprocessorFile(self.filename)
        return preprocessor_file.write_retrieval_results(path, results)


    def __len__(self):
        """
        The number of batches in the dataset.
        """
        return self.n_batches

    def __getitem__(self, i):
        """
        Return batch from dataset.
        """
        if i > self.n_batches:
            raise IndexError()
        return self.get_batch(i)
