from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from scipy import optimize
from sklearn import metrics


class LinearDS:
    def __init__(self):
        # Demonstrated positions and velocities are stored as (N, 2) arrays.
        self.demo_positions = []
        self.demo_velocities = []

        # Initialize a stable A matrix guess (negative definite).
        tmp = np.random.rand(1, 2)
        A_matrix = -(np.dot(tmp.T, tmp) + np.eye(2))
        self.init_A_indices = A_matrix

    def motion_model(self, A_matrix, pos):
        # Predict velocities using linear DS: vel = A * pos
        # Input:
        #   A_matrix: 2x2 matrix
        #   pos: N x 2 positions
        # Return:
        #   pred_vel: N x 2 predicted velocities
        pred_vel = (A_matrix @ pos.T).T
        return pred_vel

    def lyapunov_constrains(self, A_indices):
        # Stability constraint function:
        # Return real parts of eigenvalues of A.
        # To be stable, all real parts should be negative.
        A_matrix = np.reshape(A_indices, (2, 2))
        eigenvalues = np.linalg.eigvals(A_matrix)
        return np.real(eigenvalues)

    def objective_fun(self, A_indices):
        # Objective: prediction MSE between demonstrated and predicted velocities.
        A_matrix = np.reshape(A_indices, (2, 2))
        pred_vel = self.motion_model(A_matrix, self.demo_positions)
        error = metrics.mean_squared_error(self.demo_velocities, pred_vel)
        return error

    def train_ds(self):
        # Optimize matrix A under stability constraints using trust-constr.
        A_indices = self.init_A_indices.flatten()

        # Enforce negative real eigenvalues.
        stability_cons = NonlinearConstraint(
            self.lyapunov_constrains, -1000, -1e-6
        )
        optimizations_opts = {"disp": False, "maxiter": 10000}
        res = optimize.minimize(
            self.objective_fun,
            A_indices,
            constraints=[stability_cons],
            options=optimizations_opts,
            method="trust-constr",
        )

        optimal_A_indices = res.x
        return optimal_A_indices

    def import_demonstration(self, demo_name):
        # Read one demonstration and calculate velocities.
        demo = np.loadtxt(demo_name, delimiter=",", dtype=float)
        demo = np.atleast_2d(demo)

        # Support both row-wise (2, N) and column-wise (N, 2) trajectories.
        if demo.shape[0] == 2:
            positions = demo.T
        elif demo.shape[1] == 2:
            positions = demo
        else:
            raise ValueError(
                f"Unsupported demonstration shape {demo.shape}. Expected (2, N) or (N, 2)."
            )

        # Remove invalid rows.
        valid_rows = np.all(np.isfinite(positions), axis=1)
        positions = positions[valid_rows]

        if len(positions) < 2:
            raise ValueError("Demonstration needs at least 2 valid points.")

        self.demo_positions = positions
        dt = 0.002
        self.demo_velocities = self.getDerivative(self.demo_positions, dt)

    def getDerivative(self, positions, time):
        # Calculate derivative using the five-point stencil method.
        rows_num = positions.shape[0]
        dt = time
        der = np.zeros(positions.shape)
        for n in range(0, rows_num - 1):
            if n == 0:
                der[n, :] = np.zeros(positions.shape[1])
            elif n == 1 or n == rows_num - 2:
                der[n, :] = (positions[n, :] - positions[n - 1, :]) / dt
            elif 1 < n < rows_num - 2:
                der[n, :] = (
                    -positions[n + 2, :]
                    + 8 * positions[n + 1, :]
                    - 8 * positions[n - 1, :]
                    + positions[n - 2, :]
                ) / (12 * dt)
        return der

    def plot_ds(self, optimal_A_indices, demo_name=None):
        # Create streamplot of the learned DS and overlay demonstration.
        A_matrix = np.reshape(optimal_A_indices, (2, 2))

        # Build a plotting range using demonstration bounds.
        x_min, x_max = np.min(self.demo_positions[:, 0]), np.max(self.demo_positions[:, 0])
        y_min, y_max = np.min(self.demo_positions[:, 1]), np.max(self.demo_positions[:, 1])
        x_margin = 0.2 * (x_max - x_min if x_max > x_min else 1.0)
        y_margin = 0.2 * (y_max - y_min if y_max > y_min else 1.0)

        u = np.linspace(x_min - x_margin, x_max + x_margin, 100)
        v = np.linspace(y_min - y_margin, y_max + y_margin, 100)
        uu, vv = np.meshgrid(u, v)
        u_vel = np.empty_like(uu)
        v_vel = np.empty_like(vv)

        for i in range(uu.shape[0]):
            for j in range(uu.shape[1]):
                pos = np.array([[uu[i, j], vv[i, j]]])
                vel = np.dot(A_matrix, pos.T).T
                u_vel[i, j] = vel[0, 0]
                v_vel[i, j] = vel[0, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.streamplot(uu, vv, u_vel, v_vel, density=1.2, color="tab:blue", linewidth=1.0)
        ax.plot(
            self.demo_positions[:, 0],
            self.demo_positions[:, 1],
            "-r",
            linewidth=2,
            label="Demonstration",
        )
        ax.scatter(
            self.demo_positions[0, 0],
            self.demo_positions[0, 1],
            color="green",
            s=40,
            label="Start",
        )

        title_name = demo_name if demo_name is not None else "Demo"
        ax.set_title(f"Learned Linear DS - {title_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        results_dir = Path("results") / "lds"
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_name = title_name.replace(".csv", "")
        output_path = results_dir / f"{safe_name}_streamplot.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def resolve_demo_path(demo_filename):
    # Try common locations for demo files.
    candidates = [
        Path(demo_filename),
        Path("datasets") / demo_filename,
        Path("dmp") / "py_dmps" / "py_dmps" / "pydmps" / "demos" / demo_filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    datasets = ["CShape.csv", "Line.csv", "Sshape.csv", "WShape.csv"]

    results_dir = Path("results") / "lds"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / "lds_results.csv"

    rows = []

    print("\n" + "=" * 90)
    print("LINEAR DYNAMICAL SYSTEM LEARNING SUMMARY")
    print("=" * 90)

    for dataset_name in datasets:
        demo_path = resolve_demo_path(dataset_name)
        if demo_path is None:
            print(f"\nDataset: {dataset_name}")
            print("Status: skipped (file not found)")
            continue

        dynamic_system = LinearDS()

        try:
            dynamic_system.import_demonstration(str(demo_path))
            optimal_params = dynamic_system.train_ds()

            A_matrix = np.reshape(optimal_params, (2, 2))
            eigenvalues = np.linalg.eigvals(A_matrix)
            stable = bool(np.all(np.real(eigenvalues) < 0))
            mse = float(dynamic_system.objective_fun(optimal_params))

            dynamic_system.plot_ds(optimal_params, demo_name=dataset_name)

            print(f"\nDataset: {dataset_name}")
            print("Learned A matrix:")
            print(A_matrix)
            print(f"Eigenvalues: {eigenvalues}")
            print(f"Training MSE: {mse:.10f}")
            print(f"Stable: {stable}")

            rows.append(
                {
                    "dataset": dataset_name,
                    "mse": mse,
                    "a11": A_matrix[0, 0],
                    "a12": A_matrix[0, 1],
                    "a21": A_matrix[1, 0],
                    "a22": A_matrix[1, 1],
                    "eigenvalue_1": eigenvalues[0],
                    "eigenvalue_2": eigenvalues[1],
                    "stable": stable,
                }
            )
        except Exception as exc:
            print(f"\nDataset: {dataset_name}")
            print(f"Status: skipped ({exc})")

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "mse",
                "a11",
                "a12",
                "a21",
                "a22",
                "eigenvalue_1",
                "eigenvalue_2",
                "stable",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "-" * 90)
    print(f"Saved results CSV: {results_csv}")
    print(f"Saved plots folder: {results_dir}")
