# ADV_ROBOTICS_ASSIGNMENT-2

# Robot Learning from Demonstration Using Gaussian Mixture Models and Stable Dynamical Systems

This repository contains the complete implementation, output graphs, datasets, MATLAB files, Python codes, and supporting documents related to Advanced Robotics Assignment 2.

The assignment focuses on Robot Learning from Demonstration (LfD) using:

- Linear Dynamical Systems (LDS)
- Gaussian Mixture Models (GMM)
- Dynamic Movement Primitives (DMP)
- Stable Estimator of Dynamical Systems (SEDS)

The implementation was developed using both Python and MATLAB environments by modifying and extending workshop codes provided during laboratory sessions.

---

## Work Environment

### Python Environment

The Python implementation was developed and tested using:

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- pydmps

Python virtual environment:

```bash
.venv
```

### MATLAB Environment

The SEDS implementation was developed using MATLAB workshop codes provided during the laboratory sessions.

MATLAB libraries used:

- `SEDS_lib`
- `GMR_lib`

---

## Repository Structure

### Main Repository Structure

```bash
ADV_ROBOTICS_ASSIGNMENT-2/
```

Contains the full assignment implementation and outputs.

---

## Folder Description

### 1. DMP Implementation

```bash
dmp/
```

Contains the Dynamic Movement Primitive (DMP) implementation files.

#### Important Files and Folders

```bash
py_dmps/
pydmps/
examples/
demos/
results/
```

#### Output Graphs

The DMP output graphs and reproduced trajectory results can be found inside:

```bash
dmp/results/
```

---

### 2. LDS and GMM Implementation

```bash
lds_gmm/Learning_Linear_DS_GMMS/Learning_Linear_DS/
```

Contains implementations for:

- Linear Dynamical Systems (LDS)
- Gaussian Mixture Models (GMM)

#### Important Files

```bash
part_a.py
part_b.py
```

#### Dataset Files

```bash
Angle.csv
CShape.csv
Line.csv
Sshape.csv
WShape.csv
```

These datasets were used for trajectory learning experiments.

#### LDS Output Graphs

LDS result outputs can be found inside:

```bash
results/lds/
```

This folder contains:

- LDS vector field outputs
- LDS streamline graphs
- Stability visualizations

#### GMM Output Graphs

GMM result outputs can be found inside:

```bash
results/gmm/
```

This folder contains:

- AIC graphs
- BIC graphs
- GMM clustering outputs
- Component analysis plots

---

### 3. SEDS MATLAB Implementation

```bash
SEDS_MATLAB_code/seds/
```

Contains the MATLAB implementation for:

- Stable Estimator of Dynamical Systems (SEDS)

#### Important MATLAB Files

```bash
SEDS_Learning.m
demo_Plot_Results.m
```

#### MATLAB Libraries

```bash
SEDS_lib/
GMR_lib/
models/
```

#### SEDS Output Graphs

Generated SEDS output graphs and streamline visualizations are available in:

```bash
SEDS_Graph_outputs.pdf
```

This PDF contains:

- SEDS vector field outputs
- Streamline plots
- Stable trajectory convergence outputs
- Learned motion behavior visualizations

---

## Dataset Information

The experiments were performed using the following trajectory datasets:

| Dataset | Description |
| --- | --- |
| Angle | Angled trajectory used for LDS/GMM workshop tasks |
| CShape | Curved C-shaped trajectory |
| Line | Linear trajectory |
| Sshape | Nonlinear S-shaped trajectory |
| WShape | Complex W-shaped trajectory |

---

## Methods Implemented

### Linear Dynamical Systems (LDS)

Used for learning stable linear motion dynamics.

### Gaussian Mixture Models (GMM)

Used for probabilistic trajectory modeling and clustering.

Includes:

- AIC analysis
- BIC analysis
- Gaussian component selection

### Dynamic Movement Primitives (DMP)

Used for trajectory reproduction using nonlinear forcing functions.

### Stable Estimator of Dynamical Systems (SEDS)

Used for learning globally stable nonlinear dynamical systems using Lyapunov stability constraints.

---

## Output Results

The repository contains:

- LDS graphs
- GMM clustering outputs
- AIC/BIC analysis graphs
- DMP reproduced trajectories
- SEDS streamline outputs
- Stable vector field visualizations

---

## Supporting Documents

The repository also contains:

- Assignment report
- Supporting PDFs
- MATLAB output documents
- Graph output files

---

## How to Run

Run scripts from project root.

### DMP Workshop Script

```bash
python dmp/py_dmps/py_dmps/pydmps/examples/dmp_assignment.py
```

### LDS Workshop Script (Part A)

```bash
python part_a.py
```

### GMM Workshop Script (Part B)

```bash
python lds_gmm/Learning_Linear_DS_GMMS/Learning_Linear_DS/part_b.py
```

---

## Author

Saarunathan Thuviprakash  
MSc Robotics and Artificial Intelligence  
University of Lincoln
