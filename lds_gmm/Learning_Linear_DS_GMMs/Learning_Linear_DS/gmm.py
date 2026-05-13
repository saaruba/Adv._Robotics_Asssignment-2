import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


def load_trajectory(csv_path: Path):
    """
    Load one 2D trajectory from CSV and return data as shape (N, 2).
    Supports:
    - row-wise: (2, N)
    - column-wise: (N, 2)
    """
    raw = np.loadtxt(csv_path, delimiter=",", dtype=float)
    raw = np.atleast_2d(raw)

    # Format 1: row-wise trajectory.
    if raw.shape[0] == 2:
        data = raw.T
    # Format 2: column-wise trajectory.
    elif raw.shape[1] == 2:
        data = raw
    else:
        raise ValueError(
            f"Unsupported shape {raw.shape}. Expected (2, N) or (N, 2)."
        )

    # Remove rows containing NaN or inf values.
    valid_rows = np.all(np.isfinite(data), axis=1)
    data = data[valid_rows]

    if len(data) == 0:
        raise ValueError("No valid points after NaN filtering.")

    return data


def main():
    # Resolve paths relative to this script folder.
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent

    # Required datasets in the same folder as part_b.py.
    dataset_files = ["CShape.csv", "Line.csv", "Sshape.csv", "WShape.csv"]

    # Output folder for CSV and plots.
    results_dir = base_dir / "results" / "gmm"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Number of Gaussian components to test.
    component_list = [1, 2, 3, 4, 5, 6, 8, 10]

    all_rows = []
    summary_rows = []

    for dataset_name in dataset_files:
        csv_path = base_dir / dataset_name
        if not csv_path.exists():
            print(f"Skipping {dataset_name}: file not found.")
            continue

        try:
            data = load_trajectory(csv_path)
        except Exception as exc:
            print(f"Skipping {dataset_name}: {exc}")
            continue

        dataset_results = []
        best_bic = np.inf
        best_components = None
        best_model = None

        # Fit GMM for each component count.
        for n_components in component_list:
            model = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=42,
            )
            model.fit(data)

            log_likelihood = float(model.score(data))  # average log-likelihood
            aic_value = float(model.aic(data))
            bic_value = float(model.bic(data))

            dataset_results.append(
                {
                    "dataset": dataset_name.replace(".csv", ""),
                    "n_components": n_components,
                    "log_likelihood": log_likelihood,
                    "aic": aic_value,
                    "bic": bic_value,
                    "model": model,
                }
            )

            # Keep track of best model by lowest BIC.
            if bic_value < best_bic:
                best_bic = bic_value
                best_components = n_components
                best_model = model

        # Add rows for final CSV.
        for row in dataset_results:
            all_rows.append(
                {
                    "dataset": row["dataset"],
                    "n_components": row["n_components"],
                    "log_likelihood": row["log_likelihood"],
                    "aic": row["aic"],
                    "bic": row["bic"],
                    "is_best_bic": row["n_components"] == best_components,
                }
            )

        summary_rows.append((dataset_name.replace(".csv", ""), best_components, best_bic))

        # Plot 1: best-model cluster visualization.
        labels = best_model.predict(data)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            data[:, 0],
            data[:, 1],
            c=labels,
            cmap="tab10",
            s=18,
            alpha=0.9,
        )
        plt.title(f"GMM Clusters - {dataset_name.replace('.csv', '')} (k={best_components})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        legend = plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
        plt.gca().add_artist(legend)
        plt.tight_layout()
        plt.savefig(results_dir / f"{dataset_name.replace('.csv', '')}_gmm_clusters.png", dpi=200)
        plt.close()

        # Prepare metric arrays for plots.
        bic_values = [row["bic"] for row in dataset_results]
        aic_values = [row["aic"] for row in dataset_results]

        # Plot 2: BIC vs components.
        plt.figure(figsize=(8, 6))
        plt.plot(component_list, bic_values, marker="o", linewidth=2)
        plt.title(f"BIC vs Components - {dataset_name.replace('.csv', '')}")
        plt.xlabel("Number of Components")
        plt.ylabel("BIC")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"{dataset_name.replace('.csv', '')}_bic.png", dpi=200)
        plt.close()

        # Plot 3: AIC vs components.
        plt.figure(figsize=(8, 6))
        plt.plot(component_list, aic_values, marker="o", linewidth=2, color="tab:orange")
        plt.title(f"AIC vs Components - {dataset_name.replace('.csv', '')}")
        plt.xlabel("Number of Components")
        plt.ylabel("AIC")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"{dataset_name.replace('.csv', '')}_aic.png", dpi=200)
        plt.close()

    # Save results table to CSV.
    csv_path = results_dir / "gmm_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "n_components",
                "log_likelihood",
                "aic",
                "bic",
                "is_best_bic",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Print terminal summary table.
    print("\n" + "=" * 60)
    print("GMM SUMMARY (BEST BY LOWEST BIC)")
    print("=" * 60)
    print(f"{'Dataset':<20}{'Best Components':<18}{'Best BIC':<18}")
    print("-" * 60)
    for dataset, best_k, best_bic in summary_rows:
        print(f"{dataset:<20}{best_k:<18}{best_bic:<18.6f}")
    print("-" * 60)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plots in: {results_dir}")


if __name__ == "__main__":
    main()
