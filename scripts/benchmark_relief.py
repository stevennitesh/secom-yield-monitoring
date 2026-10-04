"""Measure exact ReliefF equivalence and uncached fit speed on real SECOM inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from _script_path import ensure_src_on_path

ensure_src_on_path()

import numpy as np
from skrebate import ReliefF

from secom.feature_select._relief_backend import VectorizedReliefF
from secom.io import load_raw_secom, parse_sort_and_label
from secom.preprocess import make_imputer, make_scaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--reference-jobs", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 2 or args.reference_jobs < 1:
        parser.error("rows must be at least two and reference-jobs must be positive")
    loaded = load_raw_secom(args.input_dir)
    data = parse_sort_and_label(loaded.frame).iloc[: args.rows]
    x = make_scaler("RobustScaler").fit_transform(make_imputer(True).fit_transform(data[loaded.feature_columns]))
    y = data.y_bin.to_numpy()
    results = []
    for neighbors in (5, 10, 20):
        times = []
        models = [
            ReliefF(n_neighbors=neighbors, n_jobs=args.reference_jobs),
            VectorizedReliefF(n_neighbors=neighbors, n_jobs=1),
        ]
        for model in models:
            started = perf_counter()
            model.fit(x, y)
            times.append(perf_counter() - started)
        np.testing.assert_array_equal(models[0].feature_importances_, models[1].feature_importances_)
        np.testing.assert_array_equal(models[0].top_features_, models[1].top_features_)
        row = {
            "n_samples": len(x),
            "n_features": x.shape[1],
            "neighbors": neighbors,
            "reference_workers": args.reference_jobs,
            "vectorized_workers": 1,
            "reference_seconds": times[0],
            "vectorized_seconds": times[1],
            "speedup": times[0] / times[1],
            "exact_scores_and_ranks": True,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
