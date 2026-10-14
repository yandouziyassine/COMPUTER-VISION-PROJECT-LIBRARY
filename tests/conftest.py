"""Shared helpers for the unit tests.

Project folders start with digits (``03-document-scanner-ocr``), so they can't be imported as packages.
Two ways to get at their code:

- ``load_module(project, file)``: import a whole file, for modules whose imports are light
  (OpenCV, NumPy, PyTorch) and have no side effects at import time.
- ``load_defs(project, file, names, namespace)``: pull only the named top-level functions/constants out of a
  file and run them in a fresh namespace. Used for scripts that, at import time, load an OCR model,
  import Ultralytics/Transformers or create data folders: the pure helpers inside them can still be tested
  without any of that.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # for `import common.paths`


def project_dir(project):
    """'03' or '03-document-scanner-ocr' -> the project folder."""
    matches = sorted(REPO.glob(f"{project}*"))
    if not matches:
        raise FileNotFoundError(f"no project folder matching {project!r}")
    return matches[0]


def load_module(project, filename):
    """Import <project>/<filename> under a unique module name (several projects have an evaluate.py)."""
    folder = project_dir(project)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))  # sibling imports inside the project keep working
    path = folder / filename
    name = f"p{folder.name[:2]}_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_defs(project, filename, names, namespace=None):
    """Execute only the named top-level functions, classes and assignments of a file.

    `namespace` provides what those definitions use (e.g. {"np": numpy, "re": re}).
    Raises if a name isn't defined at the top level of the file, so a rename breaks the test loudly."""
    path = project_dir(project) / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted, body, found = set(names), [], set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigned = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if assigned & wanted:
                body.append(node)
                found |= assigned & wanted
    missing = wanted - found
    if missing:
        raise NameError(f"{sorted(missing)} not defined at top level of {path}")
    ns = dict(namespace or {})
    ns["__name__"] = f"extracted_{path.stem}"
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return SimpleNamespace(**{n: ns[n] for n in names})
