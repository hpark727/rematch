from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


@dataclass
class CosineRunAnalyzer:
    """
    Analyzer for cosine-similarity "runs" shaped like:
      - wide matrix: (n_trajectories, T) where T = (#adjacent edges per trajectory)
      - padded with NaN (recommended) or a sentinel pad_value

    What this class is good for:
      - pooled histograms (with auto-zoom so you don't get 1 bar)
      - trajectory-level histograms (mean/std/jump_rate) to avoid overweighting long trajectories
      - time-series overlays: mean ± std band; optional std line on twin axis
      - quick stats summaries + trajectory-level comparisons (bootstrap CI, permutation test)

    Notes:
      - Cosine sims often cluster near 1.0; use auto_zoom=True for histograms/time series y-limits.
      - If you're running as a script, call `an.show(block=True)` at the end.
    """
    pad_value: float = np.nan
    finite_only: bool = True

    # histogram zoom defaults (quantile clipping)
    clip: Tuple[float, float] = (0.005, 0.995)   # central 99% by default
    pad_frac: float = 0.10                      # expand range by 10%

    # default histogram bins
    default_bins: int = 60

    # -----------------------------
    # Core coercion / extraction
    # -----------------------------
    def to_matrix(self, run: ArrayLike) -> np.ndarray:
        """Coerce run to 2D float matrix, preserving NaNs."""
        X = run.to_numpy() if hasattr(run, "to_numpy") else np.asarray(run)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = X.astype(float, copy=False)

        # convert pad_value to NaN (if pad_value is not already NaN)
        if not (isinstance(self.pad_value, float) and np.isnan(self.pad_value)):
            X = np.where(X == self.pad_value, np.nan, X)

        return X

    def pooled_values(self, run: ArrayLike) -> np.ndarray:
        """Flatten to 1D pooled vector; drops NaN/inf if configured."""
        X = self.to_matrix(run).ravel()
        if self.finite_only:
            X = X[np.isfinite(X)]
        return X

    # -----------------------------
    # Zoom / bins helpers
    # -----------------------------
    def auto_range(
        self,
        vals: np.ndarray,
        clip: Optional[Tuple[float, float]] = None,
        pad_frac: Optional[float] = None,
    ) -> Tuple[float, float]:
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (-1.0, 1.0)

        qlo, qhi = clip if clip is not None else self.clip
        lo = float(np.quantile(vals, qlo))
        hi = float(np.quantile(vals, qhi))

        if lo == hi:
            eps = max(1e-4, abs(lo) * 1e-4)
            return (lo - eps, hi + eps)

        pad = (hi - lo) * (pad_frac if pad_frac is not None else self.pad_frac)
        return (lo - pad, hi + pad)

    def auto_bins(
        self,
        vals: np.ndarray,
        bins: Optional[Union[int, Sequence[float]]] = None,
        clip: Optional[Tuple[float, float]] = None,
        pad_frac: Optional[float] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Return (bin_edges, (xmin, xmax)) with data-driven range."""
        if bins is None:
            bins = self.default_bins

        if isinstance(bins, int):
            xmin, xmax = self.auto_range(vals, clip=clip, pad_frac=pad_frac)
            edges = np.linspace(xmin, xmax, bins + 1)
            return edges, (xmin, xmax)

        # explicit edges
        edges = np.asarray(list(bins), dtype=float)
        return edges, (float(edges[0]), float(edges[-1]))

    # -----------------------------
    # Trajectory-level summaries
    # -----------------------------
    def trajectory_summaries(
        self,
        run: ArrayLike,
        jump_threshold: float = 0.90,
    ) -> pd.DataFrame:
        """
        Per-trajectory stats (each row = trajectory):
          n, mean, std, min, max, p10, p50, p90, jump_rate, max_run_low
        """
        X = self.to_matrix(run)
        rows = []
        for r in X:
            rr = r[np.isfinite(r)]
            if rr.size == 0:
                rows.append(
                    dict(
                        n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                        p10=np.nan, p50=np.nan, p90=np.nan,
                        jump_rate=np.nan, max_run_low=np.nan
                    )
                )
                continue

            low = (rr < jump_threshold).astype(int)
            max_run = 0
            runlen = 0
            for v in low:
                runlen = runlen + 1 if v else 0
                max_run = max(max_run, runlen)

            rows.append(
                dict(
                    n=int(rr.size),
                    mean=float(np.mean(rr)),
                    std=float(np.std(rr, ddof=0)),
                    min=float(np.min(rr)),
                    max=float(np.max(rr)),
                    p10=float(np.quantile(rr, 0.10)),
                    p50=float(np.quantile(rr, 0.50)),
                    p90=float(np.quantile(rr, 0.90)),
                    jump_rate=float(np.mean(rr < jump_threshold)),
                    max_run_low=float(max_run),
                )
            )
        return pd.DataFrame(rows)

    def summary(self, run: ArrayLike) -> Dict[str, Any]:
        """Quick pooled summary (can be dominated by longer trajectories)."""
        x = self.pooled_values(run)
        if x.size == 0:
            return {"n": 0}
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=0)),
            "min": float(np.min(x)),
            "p10": float(np.quantile(x, 0.10)),
            "p50": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
            "max": float(np.max(x)),
        }

    # -----------------------------
    # Plotting: Histograms
    # -----------------------------
    def plot_hist(
        self,
        run: ArrayLike,
        label: str,
        *,
        bins: Optional[Union[int, Sequence[float]]] = None,
        ax: Optional[plt.Axes] = None,
        density: bool = True,
        alpha: float = 0.6,
        auto_zoom: bool = True,
        clip: Optional[Tuple[float, float]] = None,
        pad_frac: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: str = "Adjacent-step cosine similarity",
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 4))

        vals = self.pooled_values(run)
        if auto_zoom:
            edges, (xmin, xmax) = self.auto_bins(vals, bins=bins, clip=clip, pad_frac=pad_frac)
            ax.set_xlim(xmin, xmax)
        else:
            edges = bins if bins is not None else self.default_bins

        ax.hist(vals, bins=edges, alpha=alpha, density=density, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if density else "Count")
        if title:
            ax.set_title(title)
        ax.legend()
        return ax

    def plot_overlay_hist(
        self,
        runs: Dict[str, ArrayLike],
        *,
        bins: Optional[Union[int, Sequence[float]]] = None,
        ax: Optional[plt.Axes] = None,
        density: bool = True,
        alpha: float = 0.45,
        auto_zoom: bool = True,
        clip: Optional[Tuple[float, float]] = None,
        pad_frac: Optional[float] = None,
        title: str = "Overlay histogram",
        xlabel: str = "Adjacent-step cosine similarity",
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 4))

        # compute shared edges based on pooled data across all runs
        all_vals = []
        for r in runs.values():
            v = self.pooled_values(r)
            if v.size:
                all_vals.append(v)
        all_vals = np.concatenate(all_vals) if all_vals else np.array([])

        if auto_zoom:
            edges, (xmin, xmax) = self.auto_bins(all_vals, bins=bins, clip=clip, pad_frac=pad_frac)
            ax.set_xlim(xmin, xmax)
        else:
            edges = bins if bins is not None else self.default_bins

        for label, r in runs.items():
            vals = self.pooled_values(r)
            ax.hist(vals, bins=edges, alpha=alpha, density=density, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(title)
        ax.legend()
        return ax

    def plot_trajectory_stat_hist(
        self,
        run: ArrayLike,
        stat: str,
        label: str,
        *,
        bins: Optional[Union[int, Sequence[float]]] = None,
        ax: Optional[plt.Axes] = None,
        density: bool = True,
        alpha: float = 0.6,
        auto_zoom: bool = True,
        clip: Optional[Tuple[float, float]] = None,
        pad_frac: Optional[float] = None,
        title: Optional[str] = None,
        jump_threshold: float = 0.90,
    ) -> plt.Axes:
        """
        Histogram of a trajectory-level statistic: mean/std/min/p10/jump_rate/max_run_low/etc.
        """
        summ = self.trajectory_summaries(run, jump_threshold=jump_threshold)
        if stat not in summ.columns:
            raise ValueError(f"Unknown stat '{stat}'. Available: {summ.columns.tolist()}")

        vals = summ[stat].to_numpy()
        vals = vals[np.isfinite(vals)]

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 4))

        if auto_zoom:
            edges, (xmin, xmax) = self.auto_bins(vals, bins=bins, clip=clip, pad_frac=pad_frac)
            ax.set_xlim(xmin, xmax)
        else:
            edges = bins if bins is not None else self.default_bins

        ax.hist(vals, bins=edges, alpha=alpha, density=density, label=label)
        ax.set_xlabel(f"Per-trajectory {stat}")
        ax.set_ylabel("Density" if density else "Count")
        if title:
            ax.set_title(title)
        ax.legend()
        return ax

    # -----------------------------
    # Plotting: Time series
    # -----------------------------
    def plot_time_series_mean_std(
        self,
        run: ArrayLike,
        label: str,
        *,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        show_band: bool = True,
        band_alpha: float = 0.20,
        clip_y: Optional[Tuple[float, float]] = None,
    ) -> plt.Axes:
        """
        Mean over trajectories at each time index; optional ±1 std shaded band.
        """
        X = self.to_matrix(run)
        t = np.arange(X.shape[1])

        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(t, mean, label=label)
        if show_band:
            ax.fill_between(t, mean - std, mean + std, alpha=band_alpha)

        ax.set_xlabel("Adjacent-step index (t)")
        ax.set_ylabel("Cosine similarity (mean)")
        if title:
            ax.set_title(title)
        if clip_y is not None:
            ax.set_ylim(*clip_y)
        ax.legend()
        return ax

    def plot_overlay_time_series_mean_std(
        self,
        runs: Dict[str, ArrayLike],
        *,
        title: str = "Mean ± 1σ over time",
        show_band: bool = True,
        band_alpha: float = 0.20,
        clip_y: Optional[Tuple[float, float]] = None,
    ) -> plt.Axes:
        """
        Overlay multiple runs: each run is a mean line; optionally each has a ±std band.
        """
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
        for label, run in runs.items():
            self.plot_time_series_mean_std(
                run,
                label=label,
                ax=ax,
                title=None,
                show_band=show_band,
                band_alpha=band_alpha,
                clip_y=None,
            )
        ax.set_title(title)
        if clip_y is not None:
            ax.set_ylim(*clip_y)
        plt.tight_layout()
        return ax

    def plot_time_series_mean_with_std_line(
        self,
        run: ArrayLike,
        label: str,
        *,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        clip_mean_y: Optional[Tuple[float, float]] = None,
        clip_std_y: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Axes, plt.Axes]:
        """
        Mean on left axis; std (dashed) on right axis (twin y).
        """
        X = self.to_matrix(run)
        t = np.arange(X.shape[1])

        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax2 = ax.twinx()

        ax.plot(t, mean, label=f"{label} mean")
        ax2.plot(t, std, linestyle="--", label=f"{label} std")

        ax.set_xlabel("Adjacent-step index (t)")
        ax.set_ylabel("Cosine similarity (mean)")
        ax2.set_ylabel("Std dev across trajectories")

        if title:
            ax.set_title(title)
        if clip_mean_y is not None:
            ax.set_ylim(*clip_mean_y)
        if clip_std_y is not None:
            ax2.set_ylim(*clip_std_y)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")

        plt.tight_layout()
        return ax, ax2

    # -----------------------------
    # Comparisons (trajectory-level)
    # -----------------------------
    def compare_trajectory_stat_bootstrap(
        self,
        run_a: ArrayLike,
        run_b: ArrayLike,
        *,
        stat: str = "mean",
        n_boot: int = 2000,
        seed: int = 0,
        jump_threshold: float = 0.90,
    ) -> Dict[str, Any]:
        """
        Bootstrap CI on difference in average per-trajectory statistic (A - B).
        """
        rng = np.random.default_rng(seed)
        A = self.trajectory_summaries(run_a, jump_threshold=jump_threshold)[stat].to_numpy()
        B = self.trajectory_summaries(run_b, jump_threshold=jump_threshold)[stat].to_numpy()
        A = A[np.isfinite(A)]
        B = B[np.isfinite(B)]
        if A.size == 0 or B.size == 0:
            return {"ok": False, "reason": "empty after filtering"}

        obs = float(np.mean(A) - np.mean(B))

        boots = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            aa = rng.choice(A, size=A.size, replace=True)
            bb = rng.choice(B, size=B.size, replace=True)
            boots[i] = float(np.mean(aa) - np.mean(bb))

        lo, hi = np.quantile(boots, [0.025, 0.975])
        return {
            "ok": True,
            "stat": stat,
            "observed_diff": obs,
            "ci95": (float(lo), float(hi)),
            "n_a": int(A.size),
            "n_b": int(B.size),
            "n_boot": int(n_boot),
        }

    def permutation_test_trajectory_stat(
        self,
        run_a: ArrayLike,
        run_b: ArrayLike,
        *,
        stat: str = "mean",
        n_perm: int = 5000,
        seed: int = 0,
        jump_threshold: float = 0.90,
    ) -> Dict[str, Any]:
        """
        Permutation test on difference in average per-trajectory statistic (A - B).
        """
        rng = np.random.default_rng(seed)
        A = self.trajectory_summaries(run_a, jump_threshold=jump_threshold)[stat].to_numpy()
        B = self.trajectory_summaries(run_b, jump_threshold=jump_threshold)[stat].to_numpy()
        A = A[np.isfinite(A)]
        B = B[np.isfinite(B)]
        if A.size == 0 or B.size == 0:
            return {"ok": False, "reason": "empty after filtering"}

        obs = float(np.mean(A) - np.mean(B))
        combined = np.concatenate([A, B])
        nA = A.size

        more_extreme = 0
        for _ in range(n_perm):
            rng.shuffle(combined)
            aa = combined[:nA]
            bb = combined[nA:]
            stat_val = float(np.mean(aa) - np.mean(bb))
            if abs(stat_val) >= abs(obs):
                more_extreme += 1

        p = (more_extreme + 1) / (n_perm + 1)
        return {
            "ok": True,
            "stat": stat,
            "observed_diff": obs,
            "p_value": float(p),
            "n_perm": int(n_perm),
            "n_a": int(A.size),
            "n_b": int(B.size),
        }

    # -----------------------------
    # Convenience
    # -----------------------------
    def show(self, block: bool = True) -> None:
        """Use this at end of scripts to prevent windows closing instantly."""
        plt.show(block=block)
