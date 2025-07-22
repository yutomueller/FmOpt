
**A surrogate-model-based optimizer for mixed (continuous + discrete) variables using Factorization Machine (FM) regression and QUBO (Quantum/Simulated Annealing) optimization.**

---

## Overview

FMQAOptimizer is a Python implementation of a **surrogate-assisted optimization algorithm** designed for problems with mixed continuous, discrete, and categorical variables.

* Builds a **Factorization Machine (FM)** surrogate model using PyTorch
* Transforms the FM model into a **QUBO (Quadratic Unconstrained Binary Optimization)** problem
* Solves the QUBO using either a **quantum annealer (D-Wave QPU)** or classical **simulated annealing (neal)**
* Handles **arbitrary combinations of continuous and discrete/categorical variables**

---

## Features

* **Mixed-variable optimization:** Simultaneous support for continuous, discrete, and categorical parameters
* **Surrogate learning:** Fast FM-based regression of expensive or black-box objectives
* **Automatic encoding:** All variables are encoded into a bitstring (binary vector) for QUBO mapping
* **Optimization backend:** Switch between quantum (D-Wave) or classical (neal) annealing
* **CSV logging and optimization progress plots**

---

## Installation

```bash
pip install numpy pandas matplotlib torch tqdm neal
# Optional: For real quantum annealing (D-Wave, requires account)
pip install dwave-system
```

---

## Quick Example

### 1. Define your search domain

```python
domain = [
    {'name': 'temp',    'type': 'continuous', 'domain': (500, 900)},         # Temperature
    {'name': 'time',    'type': 'continuous', 'domain': (1, 8)},             # Time (hours)
    {'name': 'press',   'type': 'continuous', 'domain': (0.5, 3)},           # Pressure (MPa)
    {'name': 'catalyst','type': 'discrete',   'domain': ('Ni', 'Pt', 'Fe')}, # Catalyst
    {'name': 'method',  'type': 'discrete',   'domain': ('A', 'B', 'C')},    # Processing Method
]
```

### 2. Define your objective function (black-box OK)

```python
def func(x):
    temp, time, press, catalyst, method = x
    # Simulate a nonlinear, multimodal property (e.g., strength)
    strength = -((temp-720)**2)/800 - ((time-5)**2)/4 - ((press-1.5)**2)*2
    if catalyst == 'Pt':
        strength += 2.0
    if catalyst == 'Ni' and method == 'B':
        strength += 1.5
    if method == 'C':
        strength += np.sin(press*2) * 1.2
    # Simulate cost (higher temp, longer time, Pt catalyst, method C is cheaper)
    cost = ((temp-500)/400)*2 + (time-1)/7 + (1 if catalyst=='Pt' else 0.5) - (0.7 if method=='C' else 0)
    # Maximize (strength - cost) → FMQA minimizes, so return negative
    return -(strength - cost)
```

### 3. Run the optimization

```python
optimizer = FMQAOptimizer(domain, bits=5, k=4, epochs=400, verbose=True, use_qpu=False)

best_x, best_y, history_df = optimizer.run_optimization(func, n_init=10, n_iter=20, csv_path="history_material.csv")

print("Best X:", best_x)
print("Best (strength - cost):", -best_y)
```

* `bits`: Number of quantization bits for each continuous variable (higher = finer resolution)
* `k`: Number of FM factors (for quadratic terms)
* `epochs`: Number of epochs for FM regression learning
* `use_qpu`: Set to True to use a real D-Wave quantum annealer (requires account)

---

## Output

* **Best parameters (`Best X`)** and their evaluation (`Best Y`) printed to console
* **All evaluated points and results** are saved to CSV (e.g., `history_material.csv`)
* **Optimization history plot** displayed automatically

---

## Applications

* Materials/process optimization with mixed variables
* Expensive or black-box function optimization (simulation, lab experiment, etc.)
* AI model hyperparameter tuning with categorical and numerical options
* Industrial design, scheduling, or allocation with mixed constraints

---

## Limitations & Notes

* FM surrogates model up to **quadratic (pairwise) interactions** only; highly nonlinear or high-order dependencies are harder to capture
* QUBO size grows quickly with the number of variables and bits per variable—be cautious with large problems
* Exploration of uncertainty is limited compared to Bayesian Optimization (BO)
* For pure continuous, smooth problems, consider GP-based BO as an alternative

## Reference
tsudalab/fmqa inspired
https://unit.aist.go.jp/g-quat/ja/events/img/CAE_20240509-10/20240510_01_Tamura.pdf
---
