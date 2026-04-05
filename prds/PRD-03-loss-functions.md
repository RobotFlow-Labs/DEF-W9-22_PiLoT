# PRD-03: Loss Functions

## Objective
Implement all loss functions from the PiLoT paper.

## Losses

### 1. Barron's Robust Loss (Training Loss, Eq.3)
```
L = sum_j rho_B(||p_j^q - p_tilde_j^q||_2^2)
```
- Generalized robust loss from "A General and Adaptive Robust Loss Function" (Barron, 2019)
- Parameterized by alpha (shape) and scale
- alpha=0 -> Cauchy, alpha=1 -> Charbonnier, alpha=2 -> L2

### 2. Photometric Cost (Eq.7)
```
C_photo = sum_j rho_huber(w_l(j) * ||r_{j,m}^l||_2^2)
```
- Huber robust loss on feature residuals
- w_l(j) = learned uncertainty weight per pixel per level
- r = feature difference between query and warped reference

### 3. SE(3) Motion Regularization (Eq.10)
```
C_motion = lambda * ||log(T_pred^{-1} * T_m')||_2^2
```
- Geodesic distance in SE(3) Lie algebra
- Penalizes large deviations from Kalman-predicted pose
- lambda controls regularization strength

### 4. Total Cost (Eq.10)
```
C_total = C_photo + C_motion
```

## Deliverables
- `src/pilot/losses.py`:
  - `BarronRobustLoss(alpha, scale)` -- differentiable Barron loss
  - `PhotometricCost(huber_delta)` -- uncertainty-weighted feature residual
  - `SE3MotionRegularization(lam)` -- geodesic on SE(3)
  - `PiLoTLoss(config)` -- combined training loss

## Acceptance Criteria
- Barron loss matches reference for alpha in {0, 1, 2}
- All losses differentiable (gradient flows)
- SE(3) log map numerically stable for small rotations
- Unit tests with known inputs/outputs
