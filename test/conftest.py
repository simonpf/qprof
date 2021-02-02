"""
This module defines a global fixture, which download test data
via SFTP.
"""
from pathlib import Path
import pytest
import os
import paramiko

@pytest.fixture(scope="session")
def test_data(tmpdir_factory):
    """
    Returns:
        Dictionary containing the different test file types and
        their temporary paths.
    """
    local_path = Path(tmpdir_factory.mktemp("data"))
    host = "129.16.35.202"
    user = os.environ["DENDRITE_USER"]
    password = os.environ["DENDRITE_PASSWORD"]

    transport = paramiko.Transport(host)
    transport.connect(username=user,
                      password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    path = Path("array1/share/UserAreas/Simon/qprof/")
    filename = "gpm_308_30_02_17.bin"
    source = str(path / filename)
    bin_file = local_path / filename
    sftp.get(source, str(bin_file))

    filename = "GMI_190101_027510.pp"
    source = str(path / filename)
    preprocessor_file = local_path / filename
    sftp.get(source, str(preprocessor_file))

    filename = "2A.GCORE.GMI.V7.20190101-S001447-E014719.027510.BIN.gz"
    source = str(path / filename)
    retrieval_file = local_path / filename
    sftp.get(source, str(retrieval_file))

    sftp.close()
    transport.close()

    return {"bin_file": bin_file,
            "preprocessor_file": preprocessor_file,
            "retrieval_file": retrieval_file}
