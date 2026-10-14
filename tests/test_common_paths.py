"""common/paths.py: data and outputs live under $CV_DATA (default: ../cv-data next to the repo)."""

import importlib

import common.paths as paths


def test_cv_data_env_var_controls_data_and_output_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("CV_DATA", str(tmp_path))
    # Keep the module's os.environ.setdefault() calls from leaking into the test session.
    for var in ("HF_HOME", "TORCH_HOME", "YOLO_CONFIG_DIR"):
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
    try:
        p = importlib.reload(paths)
        assert p.DATA_ROOT == tmp_path
        assert p.CACHE_ROOT == tmp_path / ".cache"

        d = p.data_dir("demo")
        assert d == tmp_path / "demo" and d.is_dir()

        o = p.output_dir("demo")
        assert o == tmp_path / "demo" / "outputs" and o.is_dir()

        assert p.data_dir("demo") == d  # calling again is fine (exist_ok)
    finally:
        monkeypatch.undo()
        importlib.reload(paths)


def test_default_data_root_is_sibling_of_repo(monkeypatch):
    monkeypatch.delenv("CV_DATA", raising=False)
    try:
        p = importlib.reload(paths)
        assert p.DATA_ROOT == p.REPO_ROOT.parent / "cv-data"
    finally:
        monkeypatch.undo()
        importlib.reload(paths)
