"""
Custom AdaGrad Optimizers from Master Thesis:
"Adaptive Subgradient Methods for Online Deep Learning"

Implements:
- DiagonalAdaGradCOMID (Algorithm 9): AdaGrad with Composite Mirror Descent
- DiagonalAdaGradRDA (Algorithm 13): AdaGrad with Regularized Dual Averaging

Both optimizers solve the unified update (Thesis §6.1, p.71):

    w_{t+1} = argmin_{w ∈ K} { ⟨u_t, w⟩ + ηφ(w) + ½⟨w, H_t^x w⟩ }

where u_t and H_t^x depend on the method:
    COMID:  u_t = ηg_t − H_t w_t,   H_t^x = H_t           (eq. 139)
    RDA:    u_t = (η/t)·ḡ_t,        H_t^x = H_t / t       (eq. 140)

Regularization φ  (Thesis §6.2):  'none', 'l1', 'l2'
Feasible sets K   (Thesis §6.3–6.5): 'none' (ℝⁿ), 'box', 'l2_ball', 'l1_ball'
"""

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
#  Projection operators  (Thesis §6.3–6.5)
# ---------------------------------------------------------------------------

def _project_box(w, bound):
    """Box projection Π_{[-c,c]}(w)  (Thesis §6.3).

    Π_{[l,r]}(w_i) = min(max(w_i, l_i), r_i)   — element-wise clipping.
    Here symmetric bounds l = -c, r = c are used.
    """
    return w.clamp(-bound, bound)


def _project_l2_ball(w, radius):
    """ℓ₂-ball projection Π_{ℓ₂,R}(w)  (Thesis §6.4).

              ⎧ w,              if ‖w‖₂ ≤ R
    Π(w) =    ⎨
              ⎩ R · w/‖w‖₂,    if ‖w‖₂ > R
    """
    norm = w.norm()
    if norm.item() > radius:
        w = w * (radius / norm)
    return w


def _project_l1_ball_weighted(w, h, radius, max_iter=50, tol=1e-8):
    """Weighted ℓ₁-ball projection with diagonal metric H^x  (Thesis §6.5).

    Solves:  argmin_{‖z‖₁ ≤ R}  ½ Σᵢ h_i^x (z_i − w̃_i)²

    where w̃ is the unconstrained/regularized solution from §6.2.

    Solution (Thesis §6.5, via KKT with Lagrange multiplier ν ≥ 0):

        z_i(ν) = sign(w̃_i) · max(|w̃_i| − ν/h_i^x, 0)

    This is equivalent to the thesis's joint formula (combining L1-reg
    threshold ηλ and constraint threshold ν):

        w_i(ν) = (1/h_ii^x) · sign(−u_i) · max(|u_i| − (ηλ + ν), 0)

    since |w̃_i| = (|u_i| − ηλ)/h_i^x  and the thresholds add:
        |w̃_i| − ν/h_i^x = (|u_i| − ηλ − ν) / h_i^x

    The function g(ν) = ‖z(ν)‖₁ is continuous, piecewise linear, and
    monotone non-increasing. ν* is found via bisection on [ν_lo, ν_hi].
    """
    # Thesis §6.5: "If g(0) ≤ R, the constraint is inactive, ν* = 0"
    if w.norm(1).item() <= radius:
        return w

    abs_w = w.abs()

    # Bisection bounds (Thesis §6.5):
    #   ν_lo = 0:          g(0) = ‖w̃‖₁ > R  (constraint violated)
    #   ν_hi = max_i b_i:  g(ν_hi) = 0 ≤ R   (all components zeroed)
    # where b_i = |u_i| − ηλ = h_i · |w̃_i|, so ν_hi = max(h · |w̃|)
    nu_lo = 0.0
    nu_hi = (h * abs_w).max().item()

    # Bisection: find ν* ∈ [ν_lo, ν_hi] such that g(ν*) = R
    # Thesis §6.5: "monotonicity and continuity make bisection a robust choice"
    for _ in range(max_iter):
        nu = (nu_lo + nu_hi) / 2.0

        # g(ν) = Σ max(|w̃_i| − ν/h_i, 0)   (Thesis §6.5)
        # Equivalent to: Σ (1/h_i) · max(|u_i| − (ηλ + ν), 0)
        projected = torch.clamp(abs_w - nu / h, min=0)
        l1_norm = projected.sum().item()

        if abs(l1_norm - radius) < tol:
            break
        if l1_norm > radius:
            # g(ν) > R → ν too small → increase lower bound
            nu_lo = nu
        else:
            # g(ν) < R → ν too large → decrease upper bound
            nu_hi = nu

    # z_i = sign(w̃_i) · max(|w̃_i| − ν*/h_i, 0)   (Thesis §6.5)
    return torch.sign(w) * torch.clamp(abs_w - nu / h, min=0)


def _apply_constraint(w, constraint, constraint_param, h=None):
    """Apply the feasible-set projection Π_K after the proximal step.

    Thesis §6.3–6.5: the constrained solution is obtained by projecting
    the unconstrained/regularized solution onto K.
    """
    if constraint == 'none':
        return w
    elif constraint == 'box':
        return _project_box(w, constraint_param)
    elif constraint == 'l2_ball':
        return _project_l2_ball(w, constraint_param)
    elif constraint == 'l1_ball':
        return _project_l1_ball_weighted(w, h, constraint_param)
    return w


# ---------------------------------------------------------------------------
#  ℓ₂-norm proximal operator  (Thesis §6.2, p.72–73)
# ---------------------------------------------------------------------------

def _l2_norm_proximal(h, u, tau, max_iter=10, tol=1e-8):
    """
    Proximal operator for φ(w) = λ‖w‖₂ with diagonal metric (Thesis §6.2).

    Solves the ℓ₂-regularized subproblem from the unified update (§6.1):

        argmin_w { ⟨u, w⟩ + τ·‖w‖₂ + ½⟨w, H^x w⟩ }

    where H^x = diag(h), τ = ηλ, and u is the method-specific direction
    (eq. 139 for COMID, eq. 140 for RDA).

    Case w = 0  (Thesis p.73):
        ‖u‖₂ ≤ τ  ⟹  w* = 0
        (subdifferential condition: ∃s with ‖s‖₂ ≤ 1 s.t. u + τs = 0)

    Case w ≠ 0  (Thesis p.73):
        Introduce β := τ/‖w‖₂ > 0, then (H^x + βI)w = −u, giving:
            w = −(H^x + βI)⁻¹ u
        For diagonal H^x:  w_i = −u_i / (h_i + β)

        β is found via Newton's method on (Thesis eq. 142):
            θ(β) = β/(ηλ) − 1/‖(H^x + βI)⁻¹ u‖₂

        with derivative (Thesis p.73):
            θ'(β) = 1/(ηλ) − ‖(H^x + βI)^{−3/2} u‖₂² / ‖(H^x + βI)⁻¹ u‖₂³

        Step-size damping c ∈ (0,1) is applied when |θ| increases
        (Thesis p.73: "the previous Newton step was too long and must be
        discarded and repeated with a step size damping c ∈ (0,1)").
    """
    # ── Case w = 0: ‖u‖₂ ≤ τ  (Thesis p.73) ──
    if torch.norm(u).item() <= tau:
        return torch.zeros_like(u)

    # ── Initial guess: β₀ = τ/‖w⁰‖₂ where w⁰ = −(H^x)⁻¹u ──
    w0_norm = torch.norm(u / h)
    if w0_norm.item() < 1e-12:
        return torch.zeros_like(u)
    beta = tau / w0_norm

    # ── Newton iteration with damping (Thesis p.73) ──
    for _ in range(max_iter):
        # Compute inv = (H^x + βI)⁻¹ once, reuse for θ and θ'
        inv = 1.0 / (h + beta)
        inv_u = inv * u                                # (H^x+βI)⁻¹ u
        norm_inv_u = torch.norm(inv_u)                  # ‖(H^x+βI)⁻¹ u‖₂

        # ── θ(β) = β/τ − 1/‖(H^x+βI)⁻¹u‖₂  (Thesis eq. 142) ──
        th = beta / tau - 1.0 / norm_inv_u
        if abs(th.item()) < tol:
            break

        # ── θ'(β) = 1/τ − ‖(H^x+βI)^{-3/2}u‖₂² / ‖(H^x+βI)⁻¹u‖₂³  (Thesis p.73) ──
        # ‖(H^x+βI)^{-3/2}u‖₂² = Σ u_i²/(h_i+β)³ = Σ inv_u_i² · inv_i
        s3 = torch.sum(inv_u * inv_u * inv)
        th_prime = 1.0 / tau - s3 / (norm_inv_u ** 3)
        if abs(th_prime.item()) < 1e-14:
            break

        step = th / th_prime
        th_abs = abs(th.item())

        # Damped Newton (Thesis p.73): "if the function value increases,
        # the step must be discarded and repeated with damping c ∈ (0,1)"
        c = 1.0
        for _ in range(10):
            beta_new = beta - c * step
            if beta_new.item() > 0:
                th_new = beta_new / tau - 1.0 / torch.norm(u / (h + beta_new))
                if abs(th_new.item()) < th_abs:
                    break
            c *= 0.5
        beta = beta_new if beta_new.item() > 0 else beta * 0.5

    # ── Solution: w_i = −u_i / (h_i + β)  (Thesis p.73) ──
    return -u / (h + beta)


# ---------------------------------------------------------------------------
#  DiagonalAdaGradCOMID  (Algorithm 9)
# ---------------------------------------------------------------------------

class DiagonalAdaGradCOMID(Optimizer):
    """
    Diagonal AdaGrad with Composite Mirror Descent (Algorithm 9).

    Unified update (Thesis §6.1):
        w_{t+1} = Π_K( argmin_w { ⟨u_t, w⟩ + ηφ(w) + ½⟨w, H_t^x w⟩ } )

    with u_t = ηg_t − H_t w_t  and  H_t^x = H_t   (Thesis eq. 139).
    """

    def __init__(self, params, lr=0.01, delta=1e-10,
                 reg_type='none', reg_lambda=0.0,
                 constraint='none', constraint_param=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if delta < 0.0:
            raise ValueError(f"Invalid delta: {delta}")
        if reg_type not in ('none', 'l1', 'l2'):
            raise ValueError(f"Invalid reg_type: {reg_type}")
        if reg_lambda < 0.0:
            raise ValueError(f"Invalid reg_lambda: {reg_lambda}")
        if constraint not in ('none', 'box', 'l2_ball', 'l1_ball'):
            raise ValueError(f"Invalid constraint: {constraint}")
        if constraint_param <= 0.0 and constraint != 'none':
            raise ValueError(f"constraint_param must be > 0, got {constraint_param}")

        defaults = dict(lr=lr, delta=delta, reg_type=reg_type,
                        reg_lambda=reg_lambda, constraint=constraint,
                        constraint_param=constraint_param)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']           # η  (step size)
            delta = group['delta']     # δ  (numerical stability)
            reg_type = group['reg_type']
            lam = group['reg_lambda']  # λ  (regularization strength)
            constraint = group['constraint']
            cparam = group['constraint_param']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad          # g_t ∈ ∂f_t(w_t)

                state = self.state[p]
                if len(state) == 0:
                    # Alg. 9, line 1: initialize s̃₀ = 0
                    state['sum_sq_grad'] = torch.zeros_like(p.data)
                    state['step'] = 0
                state['step'] += 1

                # ── Alg. 9, step 4: s̃_t = s̃_{t-1} + g_t ⊙ g_t ──
                state['sum_sq_grad'].add_(grad * grad)

                # ── Alg. 9, steps 5–6: s_t = √s̃_t,  h_t = δ·1 + s_t ──
                # For COMID:  H_t^x = H_t = diag(h_t)   (eq. 139)
                h = delta + torch.sqrt(state['sum_sq_grad'])

                # ── Thesis eq. 139: u_t = ηg_t − H_t^x · w_t ──
                u = lr * grad - h * p.data

                # ── Alg. 9, step 7: solve the unified subproblem (§6.2) ──
                # argmin_w { ⟨u, w⟩ + ηφ(w) + ½⟨w, H^x w⟩ }
                if reg_type == 'none':
                    # §6.2 case 1:  w = −(H^x)⁻¹ u
                    # Equivalent to:  w = w_t − η·g_t / h_t
                    w = -u / h

                elif reg_type == 'l1':
                    # §6.2 case 2, eq. 141:
                    # w_i = (1/h_ii^x) · sign(−u_i) · max(|u_i| − ηλ, 0)
                    w = (1 / h) * torch.sign(-u) * torch.clamp(
                        torch.abs(u) - lr * lam, min=0)

                elif reg_type == 'l2':
                    # §6.2 case 3, eq. 142 + Newton (p.73):
                    # w = −(H^x + βI)⁻¹ u  with β from θ(β) = 0
                    w = _l2_norm_proximal(h, u, lr * lam)

                # ── Projection onto K (Thesis §6.3–6.5) ──
                p.data.copy_(_apply_constraint(w, constraint, cparam, h=h))

        return loss


# ---------------------------------------------------------------------------
#  DiagonalAdaGradRDA  (Algorithm 13)
# ---------------------------------------------------------------------------

class DiagonalAdaGradRDA(Optimizer):
    """
    Diagonal AdaGrad with Regularized Dual Averaging (Algorithm 13).

    Unified update (Thesis §6.1):
        w_{t+1} = Π_K( argmin_w { ⟨u_t, w⟩ + ηφ(w) + ½⟨w, H_t^x w⟩ } )

    with u_t = (η/t)·ḡ_t  and  H_t^x = H_t / t   (Thesis eq. 140).
    """

    def __init__(self, params, lr=0.01, delta=1e-10,
                 reg_type='none', reg_lambda=0.0,
                 constraint='none', constraint_param=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if delta < 0.0:
            raise ValueError(f"Invalid delta: {delta}")
        if reg_type not in ('none', 'l1', 'l2'):
            raise ValueError(f"Invalid reg_type: {reg_type}")
        if reg_lambda < 0.0:
            raise ValueError(f"Invalid reg_lambda: {reg_lambda}")
        if constraint not in ('none', 'box', 'l2_ball', 'l1_ball'):
            raise ValueError(f"Invalid constraint: {constraint}")
        if constraint_param <= 0.0 and constraint != 'none':
            raise ValueError(f"constraint_param must be > 0, got {constraint_param}")

        defaults = dict(lr=lr, delta=delta, reg_type=reg_type,
                        reg_lambda=reg_lambda, constraint=constraint,
                        constraint_param=constraint_param)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']           # η  (step size)
            delta = group['delta']     # δ  (numerical stability)
            reg_type = group['reg_type']
            lam = group['reg_lambda']  # λ  (regularization strength)
            constraint = group['constraint']
            cparam = group['constraint_param']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad          # g_t ∈ ∂f_t(w_t)

                state = self.state[p]
                if len(state) == 0:
                    # Alg. 13, line 1: initialize s̃₀ = 0, ḡ = 0
                    state['sum_sq_grad'] = torch.zeros_like(p.data)
                    state['sum_grad'] = torch.zeros_like(p.data)
                    state['step'] = 0
                state['step'] += 1
                t = state['step']

                # ── Alg. 13, step 4: s̃_t = s̃_{t-1} + g_t ⊙ g_t ──
                state['sum_sq_grad'].add_(grad * grad)

                # ── Alg. 13, step 5: ḡ = ḡ + g_t ──
                state['sum_grad'].add_(grad)

                # ── Alg. 13, steps 6–7: h_t = δ·1 + s_t ──
                # For RDA:  H_t^x = H_t / t = diag(h_t) / t   (eq. 140)
                h = (delta + torch.sqrt(state['sum_sq_grad'])) / t
                g_bar = state['sum_grad']

                # ── Thesis eq. 140: u_t = (η/t) · ḡ ──
                u = (lr / t) * g_bar

                # ── Alg. 13, step 8: solve the unified subproblem (§6.2) ──
                # argmin_w { ⟨u, w⟩ + ηφ(w) + ½⟨w, H^x w⟩ }
                if reg_type == 'none':
                    # §6.2 case 1:  w = −(H^x)⁻¹ u = −ηḡ / h_t
                    # (the 1/t in u and h cancel out)
                    w = -u / h

                elif reg_type == 'l1':
                    # §6.2 case 2, eq. 141:
                    # w_i = (1/h_ii^x) · sign(−u_i) · max(|u_i| − ηλ, 0)
                    # Expanding: (t/h_i)·sign(−ḡ_i)·max((η/t)|ḡ_i| − ηλ, 0)
                    #          = (η/h_i)·sign(−ḡ_i)·max(|ḡ_i| − λt, 0)
                    w = (1 / h) * torch.sign(-u) * torch.clamp(
                        torch.abs(u) - lr * lam, min=0)

                elif reg_type == 'l2':
                    # §6.2 case 3, eq. 142 + Newton (p.73):
                    # w = −(H^x + βI)⁻¹ u  with β from θ(β) = 0
                    w = _l2_norm_proximal(h, u, lr * lam)

                # ── Projection onto K (Thesis §6.3–6.5) ──
                p.data.copy_(_apply_constraint(w, constraint, cparam, h=h))

        return loss
