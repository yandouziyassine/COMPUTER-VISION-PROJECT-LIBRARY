"""Compatibility shim: let `rapidocr_openvino` run on current OpenVINO.

rapidocr-openvino 1.4.4 does `from openvino.runtime import Core` and pins openvino<=2024.0. Newer OpenVINO
(2025+) removed the `openvino.runtime` module, and `Core` lives at the top level. Pinning the old version
broke the GPU INT8 benchmark of project 19, so instead the old module path is aliased to the new one.

Import this module before `rapidocr_openvino`.
"""

import sys

import openvino

if "openvino.runtime" not in sys.modules:
    try:
        import openvino.runtime  # noqa: F401  (older OpenVINO: nothing to do)
    except ImportError:
        sys.modules["openvino.runtime"] = openvino
