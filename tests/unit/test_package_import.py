from importlib.metadata import version

import conversor_fcf
import conversor_fcf.cobre
import conversor_fcf.decomp
import conversor_fcf.mapping
import conversor_fcf.reporting


def test_package_import() -> None:
    assert conversor_fcf.__version__ == version("conversor-fcf")
