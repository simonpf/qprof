"""
============
qprof.models
============

This module provides access to the neural network models that can be used
in the retrieval.
"""
from pathlib import Path

from quantnn.qrnn import QRNN
from quantnn.normalizer import Normalizer

MODELS_DIRECTORY = Path(__file__).parent / "data"

def get_normalizer():
    """
    Returns noramlizer for GPROF GMI data.
    """
    return Normalizer.load(MODELS_DIRECTORY / "gprof_gmi_normalizer.pckl")


def get_model():
    """
    Returns the neural network model which is currently used for the QPROF processing.
    """
    return QRNN.load(MODELS_DIRECTORY / "qrnn_gmi_3_256_relu.pt")
