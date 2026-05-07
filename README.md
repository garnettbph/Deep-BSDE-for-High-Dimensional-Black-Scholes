# Deep BSDE Solver: High-Dimensional Option Pricing via Deep Learning

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Standard finite difference methods suffer from the "curse of dimensionality" when pricing derivatives with multiple underlying assets. A 100-asset portfolio creates a computational grid that is mathematically hard to solve using classical numerical PDE methods.

This project implements the **Deep Backward Stochastic Differential Equation (Deep BSDE)** method (introduced by *Han, Jentzen, and E, 2018*) to solve the high-dimensional Black-Scholes-Barenblatt PDE. By reformulating the PDE into a system of Forward-Backward SDEs, we use PyTorch neural networks to approximate the spatial gradients (the Delta hedging strategy), completely bypassing the dimensional bottleneck.

## Key Features
- **100-Dimensional State Space:** Solves the pricing PDE for a 100-asset basket option simultaneously.
- **Market Non-Linearity:** Incorporates differential rates for borrowing ($r_b$) and lending ($r_l$), a problem classical analytical formulas cannot solve.
- **Mathematical Validation:** Includes an exact closed-form analytical benchmark for the linear geometric basket option to rigorously prove the model's convergence and measure the non-linear premium.
- **Time-Sequential Neural Networks:** Utilizes independent sub-networks per time step with Batch Normalization to prevent vanishing/exploding gradients during the Euler-Maruyama forward propagation.

---

## Mathematical Formulation

### 1. The Forward SDE (Market Dynamics)
The underlying 100 assets are simulated via generalized Geometric Brownian Motion:
$$dX_t = \mu X_t dt + \sigma X_t dW_t$$

### 2. The Backward SDE (Pricing and Hedging)
The option price ($Y_t$) and the Delta hedging strategy ($Z_t$) are governed by the BSDE:
$$dY_t = -f(t, X_t, Y_t, Z_t)dt + Z_t^T dW_t$$

Where the **non-linear driver** $f(t, X_t, Y_t, Z_t)$ reflects the differential borrowing and lending rates:
$$f(t, X_t, Y_t, Z_t) = -r_l Y_t - (r_b - r_l) \max\left(Y_t - \sum \frac{Z_t}{\sigma}, 0\right)$$

### 3. Deep Learning Reformulation
The model initializes the unknown starting price $Y_0$ as a trainable parameter. At each time step, a neural network approximates the hedging gradient $Z_t \approx \mathcal{N}\mathcal{N}_{\theta_t}(X_t)$. The loss function is the Mean Squared Error between the terminal predicted portfolio value $Y_T$ and the actual mathematical payoff $g(X_T)$.

---

## Repository Structure

```text
deep-bsde-pricing/
├── README.md                   # Project overview and instructions
├── requirements.txt            # Python dependencies
├── notebooks/                  
│   └── Deep_BSDE_Solver.ipynb  # Interactive walkthrough notebook
├── src/                        
│   ├── __init__.py
│   ├── environment.py          # SDE generator and non-linear driver
│   ├── model.py                # PyTorch architecture (DeepBSDE)
│   └── train.py                # Execution engine and training loop
├── tests/                      
│   ├── __init__.py
│   └── test_pricing.py         # Unit tests for the mathematical benchmark
└── results/                    
    └── results.png             # Output graphs
```
---

## Key Findings:

1. **The Non-Linear Premium:** The model converges to a price of ~3.18, while the standard Black-Scholes analytical benchmark sits at ~2.12. This is a correct and expected result. Because the model is forced to borrow at a higher rate (6%) than it lends (4%), the replication strategy is more expensive. The Deep BSDE successfully prices this non-linear friction.

2. **Learned Symmetries (Deltas):** Plot 3 shows the initial hedging weights ($Z_0$) for all 100 assets. Because the assets share identical initial states and volatilities, the neural network independently learned to assign a perfectly uniform Delta weight (~0.18) to every asset. It learned the true structural physics of the market, not just noise.
---

## Quickstart Guide
### 1. Clone the repository
```bash
git clone [https://github.com/garnettbph/Deep-BSDE-for-High-Dimensional-Black-Scholes.git](https://github.com/garnettbph/Deep-BSDE-for-High-Dimensional-Black-Scholes.git)
cd deep-bsde-pricing
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Unit Tests
Before training, verify that the mathematical baseline and analytical pricing formula are correct.
```bash
python -m unittest tests/test_pricing.py
```

### 4. Train the Model
Executes the Monte Carlo paths, trains the neural networks, and saves the resulting plots to the /results directory.
```bash
python src/train.py
```

## References
- Han, J., Jentzen, A., & E, W. (2018). Solving high-dimensional partial differential equations using deep learning. Proceedings of the National Academy of Sciences, 115(34), 8505-8510.
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637-654.
