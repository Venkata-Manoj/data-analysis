"""pytest configuration: make project analysis modules importable.

Both projects define ``analysis.py``, so plain ``import analysis`` would
collide (the first import wins and the second test module gets the wrong
one). Load each as a distinct module name and register it in sys.modules so
tests can ``import wine_analysis`` / ``import pm25_analysis``.
"""

import importlib.util
import sys
from pathlib import Path

PROJECTS = {
    "wine_analysis": "wine-quality-classification",
    "pm25_analysis": "pm25-air-quality-forecasting",
    "recommender_analysis": "recommender-system-movielens",
    "anomaly_analysis": "anomaly-detection-fraud",
}

for module_name, project_dir in PROJECTS.items():
    analysis_path = Path(__file__).resolve().parent.parent / project_dir / "analysis.py"
    spec = importlib.util.spec_from_file_location(module_name, analysis_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {analysis_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
