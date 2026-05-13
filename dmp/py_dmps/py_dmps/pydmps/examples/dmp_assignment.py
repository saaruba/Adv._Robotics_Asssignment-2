import sys
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

# Force Python to use the professor's local pydmps package
script_path = Path(__file__).resolve()
pydmps_root = script_path.parent.parent
sys.path.insert(0, str(pydmps_root))

import pydmps
import pydmps.dmp_discrete
print("Using pydmps from:", pydmps.__file__)

def load_demo_csv(csv_path: Path):
    """
    Load a row-wise trajectory CSV file.
    Expected format:
    - row 0: x values
    - row 1: y values
    """
    data = np.loadtxt(csv_path, delimiter=",", ndmin=2)

    # Main expected format is (2, N).
    if data.shape[0] >= 2:
        x = data[0]
        y = data[1]
    # Fallback in case file is (N, 2).
    elif data.shape[1] >= 2:
        x = data[:, 0]
        y = data[:, 1]
    else:
        raise ValueError(f"Unsupported CSV shape: {data.shape}")

    # Keep only valid (x, y) pairs.
    valid = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        raise ValueError("No valid points found after NaN filtering.")

    return x, y


def to_xy_matrix(path_data):
    """
    Convert path data to shape (N, 2) when possible.
    """
    arr = np.asarray(path_data)

    if arr.ndim != 2:
        raise ValueError("Path data must be a 2D array.")

    if arr.shape[1] == 2:
        return arr

    if arr.shape[0] == 2:
        return arr.T

    raise ValueError(f"Could not convert array to (N, 2). Input shape: {arr.shape}")


def interpolate_xy(path_xy: np.ndarray, target_len: int) -> np.ndarray:
    """
    Interpolate a trajectory path (N,2) to a target number of points.
    """
    current_len = len(path_xy)
    if current_len == target_len:
        return path_xy

    old_t = np.linspace(0.0, 1.0, current_len)
    new_t = np.linspace(0.0, 1.0, target_len)

    x_interp = np.interp(new_t, old_t, path_xy[:, 0])
    y_interp = np.interp(new_t, old_t, path_xy[:, 1])
    return np.column_stack((x_interp, y_interp))


def to_time_major_matrix(values):
    """
    Convert forcing values to shape (timesteps, n_dmps) when possible.
    """
    arr = np.asarray(values, dtype=float)

    if arr.ndim == 1:
        return arr[:, None]

    if arr.ndim != 2:
        raise ValueError(f"Forcing term must be 1D or 2D. Got shape: {arr.shape}")

    # Prefer orientation where columns are DMPS dimensions (usually 2).
    if arr.shape[1] <= arr.shape[0]:
        return arr

    # If likely transposed, flip it.
    return arr.T


def interpolate_time_series(matrix: np.ndarray, target_len: int) -> np.ndarray:
    """
    Interpolate a time-series matrix (T, D) to target_len rows.
    """
    current_len = matrix.shape[0]
    if current_len == target_len:
        return matrix

    old_t = np.linspace(0.0, 1.0, current_len)
    new_t = np.linspace(0.0, 1.0, target_len)

    interpolated = np.zeros((target_len, matrix.shape[1]), dtype=float)
    for d in range(matrix.shape[1]):
        interpolated[:, d] = np.interp(new_t, old_t, matrix[:, d])
    return interpolated


def align_forcing_terms(f_target, f_learned):
    """
    Convert, clean, and align forcing terms for comparison.
    Returns aligned target and learned forcing matrices (T, D), or (None, None).
    """
    if f_target is None or f_learned is None:
        return None, None

    try:
        target = to_time_major_matrix(f_target)
        learned = to_time_major_matrix(f_learned)
    except Exception:
        return None, None

    # Keep same number of forcing dimensions.
    n_dims = min(target.shape[1], learned.shape[1])
    if n_dims == 0:
        return None, None
    target = target[:, :n_dims]
    learned = learned[:, :n_dims]

    # Match number of timesteps using interpolation.
    learned = interpolate_time_series(learned, target_len=target.shape[0])

    # Remove rows that contain invalid values.
    valid_rows = np.all(np.isfinite(target), axis=1) & np.all(np.isfinite(learned), axis=1)
    target = target[valid_rows]
    learned = learned[valid_rows]

    if len(target) == 0:
        return None, None

    return target, learned


def compute_forcing_mse(f_target, f_learned):
    """
    Compute forcing term MSE with robust alignment.
    Returns (forcing_mse, target_aligned, learned_aligned).
    """
    target_aligned, learned_aligned = align_forcing_terms(f_target, f_learned)
    if target_aligned is None or learned_aligned is None:
        return np.nan, None, None

    forcing_mse = float(np.mean((target_aligned - learned_aligned) ** 2))
    return forcing_mse, target_aligned, learned_aligned


def main():
    # Build paths relative to this file.
    script_path = Path(__file__).resolve()
    pydmps_root = script_path.parent.parent
    demos_dir = pydmps_root / "demos"
    results_dir = pydmps_root / "results" / "dmp"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not demos_dir.exists():
        print(f"Demos folder not found: {demos_dir}")
        return

    csv_files = sorted(demos_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {demos_dir}")
        return

    rbf_list = [5, 10, 20, 30, 50, 75, 100]
    summary_rows = []
    csv_rows = []

    for csv_file in csv_files:
        try:
            x, y = load_demo_csv(csv_file)
        except Exception as exc:
            print(f"\nSkipping {csv_file.name}: {exc}")
            continue

        dataset_name = csv_file.stem
        original_xy = np.column_stack((x, y))

        # Requirement: demo_path shape (n_dmps, timesteps).
        demo_path = np.vstack([x, y])

        dataset_results = []
        all_rollouts = []

        for n_bfs in rbf_list:
            dmp = pydmps.dmp_discrete.DMPs_discrete(
                n_dmps=2,
                n_bfs=n_bfs,
                y0=demo_path[:, 0].copy(),
                goal=demo_path[:, -1].copy()

            )    



            # Professor's custom API:
            # imitate_path() returns target forcing term.
            f_target = dmp.imitate_path(y_des=demo_path)

            # rollout() returns trajectory, derivatives, and learned forcing term.
            rollout_output = dmp.rollout(tau=1.0)

            if len(rollout_output) == 4:
                y_track, dy_track, ddy_track, f_learned = rollout_output
            else:
                y_track, dy_track, ddy_track = rollout_output
                f_learned = None

            _ = dy_track, ddy_track        

            rollout_xy = to_xy_matrix(y_track)

            # Interpolate rollout if length differs from original.
            rollout_xy_aligned = interpolate_xy(rollout_xy, target_len=len(original_xy))

            # Remove invalid trajectory rows before MSE.
            valid_traj = np.all(np.isfinite(original_xy), axis=1) & np.all(
                np.isfinite(rollout_xy_aligned), axis=1
            )
            original_xy_valid = original_xy[valid_traj]
            rollout_xy_valid = rollout_xy_aligned[valid_traj]

            if len(original_xy_valid) == 0:
                traj_mse = np.nan
            else:
                # Trajectory MSE.
                traj_mse = float(np.mean((original_xy_valid - rollout_xy_valid) ** 2))

            # Forcing term MSE with robust shape handling and interpolation.
            forcing_mse, f_target_aligned, f_learned_aligned = compute_forcing_mse(
                f_target, f_learned
            )

            result = {
                "n_bfs": n_bfs,
                "traj_mse": traj_mse,
                "forcing_mse": forcing_mse,
                "rollout_xy": rollout_xy_aligned,
                "f_target_aligned": f_target_aligned,
                "f_learned_aligned": f_learned_aligned,
            }
            dataset_results.append(result)
            all_rollouts.append((n_bfs, rollout_xy_aligned))

            csv_rows.append(
                {
                    "dataset": dataset_name,
                    "rbf_count": n_bfs,
                    "trajectory_mse": traj_mse,
                    "forcing_mse": forcing_mse,
                    "is_best_rbf": False,  # updated later once best is known
                }
            )

        # Select best RBF by lowest trajectory MSE.
        valid_results = [row for row in dataset_results if np.isfinite(row["traj_mse"])]
        if valid_results:
            best_result = min(valid_results, key=lambda row: row["traj_mse"])
        else:
            # Fallback if everything is invalid.
            best_result = dataset_results[0]

        summary_rows.append(
            {
                "dataset": dataset_name,
                "best_rbf": best_result["n_bfs"],
                "best_traj_mse": best_result["traj_mse"],
                "best_forcing_mse": best_result["forcing_mse"],
            }
        )

        # Update CSV rows for this dataset to mark best RBF.
        for row in csv_rows:
            if row["dataset"] == dataset_name and row["rbf_count"] == best_result["n_bfs"]:
                row["is_best_rbf"] = True

        # Plot A: Original vs best rollout.
        plt.figure(figsize=(8, 6))
        plt.plot(original_xy[:, 0], original_xy[:, 1], linewidth=2, label="Original")
        plt.plot(
            best_result["rollout_xy"][:, 0],
            best_result["rollout_xy"][:, 1],
            "--",
            linewidth=2,
            label=f"Best DMP (RBF={best_result['n_bfs']})",
        )
        plt.title(f"DMP Comparison: {dataset_name}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / f"{dataset_name}_comparison.png", dpi=200)
        plt.close()

        # Plot B: All RBF rollouts.
        plt.figure(figsize=(8, 6))
        plt.plot(original_xy[:, 0], original_xy[:, 1], color="black", linewidth=2.5, label="Original")
        for n_bfs, rollout_xy in all_rollouts:
            plt.plot(rollout_xy[:, 0], rollout_xy[:, 1], linewidth=1.5, alpha=0.85, label=f"RBF={n_bfs}")
        plt.title(f"All DMP Rollouts: {dataset_name}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(results_dir / f"{dataset_name}_all_rollouts.png", dpi=200)
        plt.close()

        # Plot C: RBF vs trajectory MSE.
        plt.figure(figsize=(8, 6))
        traj_mse_values = [row["traj_mse"] for row in dataset_results]
        plt.plot(rbf_list, traj_mse_values, marker="o", linewidth=2)
        plt.title(f"Trajectory MSE vs RBF: {dataset_name}")
        plt.xlabel("RBF count")
        plt.ylabel("Trajectory MSE")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"{dataset_name}_traj_mse.png", dpi=200)
        plt.close()

        # Plot D: RBF vs forcing term MSE.
        plt.figure(figsize=(8, 6))
        forcing_mse_values = [row["forcing_mse"] for row in dataset_results]
        forcing_plot_values = [np.nan if not np.isfinite(v) else v for v in forcing_mse_values]
        plt.plot(rbf_list, forcing_plot_values, marker="o", linewidth=2, color="tab:orange")
        plt.title(f"Forcing Term MSE vs RBF: {dataset_name}")
        plt.xlabel("RBF count")
        plt.ylabel("Forcing term MSE")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"{dataset_name}_forcing_mse.png", dpi=200)
        plt.close()

        # Plot E: Forcing term comparison for best RBF model.
        if (
            best_result["f_target_aligned"] is not None
            and best_result["f_learned_aligned"] is not None
        ):
            f_t = best_result["f_target_aligned"]
            f_l = best_result["f_learned_aligned"]
            time = np.arange(f_t.shape[0])

            plt.figure(figsize=(9, 6))
            for dim in range(f_t.shape[1]):
                plt.plot(time, f_t[:, dim], linewidth=2, label=f"Target forcing (dim {dim + 1})")
                plt.plot(
                    time,
                    f_l[:, dim],
                    "--",
                    linewidth=2,
                    label=f"Learned forcing (dim {dim + 1})",
                )
            plt.title(f"Forcing Term Comparison - {dataset_name}")
            plt.xlabel("Timestep")
            plt.ylabel("Forcing value")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / f"{dataset_name}_forcing_comparison.png", dpi=200)
            plt.close()
        else:
            print(f"Note: forcing comparison plot skipped for {dataset_name} (missing/invalid forcing data).")

    if not summary_rows:
        print("\nNo valid datasets were processed.")
        return

    # Save per-RBF metrics CSV.
    csv_path = results_dir / "dmp_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "rbf_count", "trajectory_mse", "forcing_mse", "is_best_rbf"],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # Print summary table.
    print("\n" + "=" * 78)
    print("DMP ASSIGNMENT SUMMARY")
    print("=" * 78)
    print(f"{'Dataset':<16}{'Best RBF':<12}{'Best Trajectory MSE':<24}{'Best Forcing MSE':<24}")
    print("-" * 78)
    for row in summary_rows:
        forcing_text = f"{row['best_forcing_mse']:.10f}" if np.isfinite(row["best_forcing_mse"]) else "N/A"
        print(
            f"{row['dataset']:<16}{row['best_rbf']:<12}{row['best_traj_mse']:<24.10f}{forcing_text:<24}"
        )
    print("-" * 78)
    print(f"Saved plots to: {results_dir}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()

