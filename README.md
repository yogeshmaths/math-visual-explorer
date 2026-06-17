# Mathematical Concepts Visual Explorer

An interactive Streamlit application showcasing advanced mathematical concepts through stunning visual explorations.

**Built for:** Yogesh Kumar Singh  
**Affiliation:** PhD Scholar, Department of Computer Science & Engineering, IIT Kharagpur  
**Background:** M.Sc. Mathematics (BHU) | M.Tech CSE (IIT Kharagpur)

---

## Overview

This application bridges pure mathematics and computational implementation across five interactive modules:

| Page | Concept | Key Math |
|------|---------|----------|
| Fourier Transform | Fourier Series Decomposition | Harmonic Analysis, Signal Processing |
| Eigenvalues | Linear Transformation Explorer | Linear Algebra, Spectral Theory |
| Gradient Descent | Optimization on Loss Surfaces | Calculus, Convex/Non-convex Optimization |
| Probability Distributions | Distribution Explorer | Probability Theory, Statistics |
| Regression | Regression Analysis Visual | Statistical Inference, Correlation |

---

## Installation

```bash
# Clone or navigate to the project directory
cd math-explorer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Pages

### 1. Fourier Series Decomposition
Visualize how any periodic signal (square wave, sawtooth, triangle) can be decomposed into a sum of sinusoids. Adjust the number of harmonics to see Gibbs phenomenon and convergence in real time.

### 2. Linear Transformation Explorer
Explore how 2×2 matrices transform the plane. Visualize eigenvectors as the directions that remain invariant under transformation (only scaled by their eigenvalues).

### 3. Gradient Descent on Loss Surfaces
Watch gradient descent navigate optimization landscapes including the notorious Rosenbrock banana function and Himmelblau's multi-minima surface. Compare behavior across learning rates.

### 4. Probability Distribution Explorer
Interactively explore six distributions (Normal, Poisson, Binomial, Exponential, Beta, Gamma) with real-time PDF/CDF plots and analytical statistical moments.

### 5. Regression Analysis Visual
Generate correlated data and perform OLS regression. Inspect residual plots and Q-Q plots to verify classical regression assumptions.

---

## Technical Stack
- **Streamlit** — UI framework
- **Plotly** — Interactive visualizations
- **NumPy** — Numerical computation
- **SciPy** — Statistical distributions and functions
- **Pandas** — Data manipulation

---

*Portfolio project demonstrating the intersection of mathematical theory and computational visualization.*
