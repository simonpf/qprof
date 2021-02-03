from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gpm-qprof',
    version='0.0.1dev',
    description='The operational QRNN-based implementation of GPROF.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/simonpf/regn',  # Optional
    author='Simon Pfreundschuh',  # Optional
    author_email='simon.pfreundschuh@chalmers.se',  # Optional
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "quantnn",
        "torch",
        "xarray"
    ],
    extras_require={
        "training_data_processing": ["netCDF4",]
    },
    python_requires='>=3.6',
    project_urls={'Source': 'https://github.com/simonpf/qprof/'},
    entry_points={'console_scripts': ['qprof=qprof.qprof:main'],},
    include_package_data=True,
)

