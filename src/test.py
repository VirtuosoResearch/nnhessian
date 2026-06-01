"""
Compare Hutchinson trace estimator vs Hutch++ at equal HVP query budgets.

Both methods estimate Tr(H) of the loss Hessian w.r.t. model parameters.
- hutchinson_trace(num_samples=m): uses m random probe vectors
- hutch_pp_trace_estimator(m):    uses m HVPs but allocates them more carefully

Ground truth is computed by applying H to each standard basis vector (d HVP calls),
feasible only for small models. We report mean estimate, absolute error, std, and
relative error across n_runs independent trials at each budget.

Note: Hutch++ requires m ≡ 2 (mod 4) due to its internal split s=(m+2)//4, g=(m-2)//2.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from nnhessian.hessian import NNHessianCalculator


class TinyMLP(nn.Module):
    def __init__(self, input_dim=4, hidden=8, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_dataloader(input_dim, output_dim, n_samples=200, batch_size=50, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    Y = torch.randn(n_samples, output_dim)
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=False)


def load_batch(batch, device):
    x, y = batch
    return x.to(device), y.to(device), x.size(0)


def exact_trace(calc):
    """Exact Tr(H) via d basis-vector HVPs. Only feasible for small models."""
    d = calc.total_params
    eye = torch.eye(d)
    trace = sum(calc._hessian_vector_product(eye[i])[i].item() for i in range(d))
    return trace


def run_experiment(n_runs=20, budgets=(10, 18, 26, 50, 102)):
    torch.manual_seed(42)

    model = TinyMLP(input_dim=4, hidden=8, output_dim=2)
    dataloader = make_dataloader(input_dim=4, output_dim=2)

    calc = NNHessianCalculator(
        model=model,
        loss_fn=nn.MSELoss(),
        dataloader=dataloader,
        external_load_batch_func=load_batch,
        device="cpu",
    )

    d = calc.total_params
    print(f"Model parameters: {d}")
    print(f"Computing exact trace via {d} HVP calls...")
    gt = exact_trace(calc)
    print(f"Exact trace: {gt:.6f}\n")

    col = f"{'Budget':<8} {'Method':<12} {'Mean':<14} {'|Error|':<12} {'Std':<12} {'Rel Err':<10} {'Time/call':>10}"
    print(col)
    print("-" * len(col))

    for m in budgets:
        # Hutchinson: seed controls the probe vectors so each run is independent
        hutch_ests, hutch_times = [], []
        for run in range(n_runs):
            t0 = time.perf_counter()
            est = calc.hutchinson_trace(num_samples=m, distribution="rademacher", seed=run)
            hutch_times.append(time.perf_counter() - t0)
            hutch_ests.append(est)
        _report("Hutchinson", m, hutch_ests, hutch_times, gt)

        # Hutch++: no seed parameter — set torch global seed before each call
        hpp_ests, hpp_times = [], []
        for run in range(n_runs):
            torch.manual_seed(run)
            t0 = time.perf_counter()
            hpp_ests.append(calc.hutch_pp_trace_estimator(m=m).item())
            hpp_times.append(time.perf_counter() - t0)
        _report("Hutch++", m, hpp_ests, hpp_times, gt)

        print()

    run_budget_to_accuracy(calc, gt, n_runs=n_runs, budgets=budgets)


def run_budget_to_accuracy(calc, gt, n_runs=20, budgets=(10, 18, 26, 50, 102),
                           std_targets=(0.8, 0.6, 0.4)):
    """
    For each target std, find the minimum budget each method needs to achieve it.
    This shows Hutch++'s real advantage: fewer HVP calls for the same precision.
    """
    print("\n--- Budget needed to reach target std ---\n")

    # Collect (budget -> std) for each method across n_runs
    results = {"Hutchinson": {}, "Hutch++": {}}

    for m in budgets:
        hutch_ests = [
            calc.hutchinson_trace(num_samples=m, distribution="rademacher", seed=run)
            for run in range(n_runs)
        ]
        results["Hutchinson"][m] = np.std(hutch_ests)

        hpp_ests = []
        for run in range(n_runs):
            torch.manual_seed(run)
            hpp_ests.append(calc.hutch_pp_trace_estimator(m=m).item())
        results["Hutch++"][m] = np.std(hpp_ests)

    col = f"{'Target Std':<14} {'Hutchinson':>12} {'Hutch++':>10} {'Speedup':>10}"
    print(col)
    print("-" * len(col))

    for target in std_targets:
        row = {method: None for method in results}
        for method, std_by_m in results.items():
            for m in sorted(std_by_m):
                if std_by_m[m] <= target:
                    row[method] = m
                    break

        h_m = row["Hutchinson"]
        hpp_m = row["Hutch++"]
        h_str = f"{h_m} HVPs" if h_m else ">max"
        hpp_str = f"{hpp_m} HVPs" if hpp_m else ">max"
        speedup = f"{h_m/hpp_m:.2f}×" if (h_m and hpp_m) else "n/a"
        print(f"{target:<14.1f} {h_str:>12} {hpp_str:>10} {speedup:>10}")

    # Also print the raw std curve so the trend is visible
    print(f"\n--- Std vs budget (lower is better) ---\n")
    col2 = f"{'Budget':<8} {'Hutchinson Std':<18} {'Hutch++ Std':<16} {'Std ratio H/H++'}"
    print(col2)
    print("-" * len(col2))
    for m in sorted(budgets):
        h_std = results["Hutchinson"][m]
        hpp_std = results["Hutch++"][m]
        ratio = h_std / hpp_std if hpp_std > 0 else float("inf")
        print(f"{m:<8} {h_std:<18.4f} {hpp_std:<16.4f} {ratio:.2f}×")


def _report(method, m, estimates, times, gt):
    mean = np.mean(estimates)
    std = np.std(estimates)
    err = abs(mean - gt)
    rel = err / max(abs(gt), 1e-8)
    mean_time = np.mean(times)
    time_str = f"{mean_time*1000:.1f} ms" if mean_time < 1 else f"{mean_time:.2f} s"
    print(f"{m:<8} {method:<12} {mean:<14.4f} {err:<12.4f} {std:<12.4f} {rel:<10.4f} {time_str:>10}")


##############################################################
# Deflated SLQ: Hutch++ idea applied to spectrum estimation
##############################################################

def lanczos_tridiag(hvp_fn, v0, n_iter):
    """
    n_iter steps of the Lanczos 3-term recurrence starting from unit vector v0.
    Returns a symmetric tridiagonal matrix T as a numpy array of shape (n_iter, n_iter).
    """
    alpha = np.zeros(n_iter)
    beta = np.zeros(n_iter - 1)

    v_prev = torch.zeros_like(v0)
    v = v0 / v0.norm()
    b = 0.0  # β_1 = 0 by convention

    for j in range(n_iter):
        Hv = hvp_fn(v)
        a = torch.dot(Hv, v).item()
        alpha[j] = a
        r = Hv - a * v - b * v_prev

        if j < n_iter - 1:
            b = r.norm().item()
            beta[j] = b
            if b > 1e-10:
                v_next = r / b
            else:
                # Near-happy breakdown: restart with a vector orthogonal to current v
                v_next = torch.randn_like(v)
                v_next = v_next - v * torch.dot(v_next, v)
                norm = v_next.norm()
                v_next = v_next / norm if norm > 1e-10 else torch.zeros_like(v)
            v_prev, v = v, v_next

    return np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)


def slq_estimate(hvp_fn, d, n_v, n_iter, deflate_Q=None):
    """
    Estimate the spectral density via SLQ.
    Returns (values, weights) with total weight ≈ 1.

    deflate_Q: optional (d, s) orthonormal matrix. If given, each probe vector
               is projected out of span(deflate_Q) before running Lanczos,
               so the estimate targets the residual (deflated) spectrum.
    """
    all_eigs, all_weights = [], []
    for _ in range(n_v):
        v = torch.randn(d)
        if deflate_Q is not None:
            v = v - deflate_Q @ (deflate_Q.t() @ v)
            norm = v.norm()
            if norm < 1e-10:
                continue
            v = v / norm
        else:
            v = v / v.norm()

        T = lanczos_tridiag(hvp_fn, v, n_iter)
        eigs, U = np.linalg.eigh(T)
        all_eigs.append(eigs)
        all_weights.append(U[0] ** 2)

    n_valid = max(len(all_eigs), 1)
    # Average over probes so total weight ≈ 1 (each probe's weights sum to 1)
    return np.concatenate(all_eigs), np.concatenate(all_weights) / n_valid


def kde_density(values, weights, grid, sigma):
    """Weighted Gaussian KDE evaluated on grid."""
    density = np.zeros_like(grid, dtype=float)
    for v, w in zip(values, weights):
        density += w * np.exp(-0.5 * ((grid - v) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return density


def _deflated_slq_l1(calc, d, exact_density, grid, sigma, dx, s, n_v, n_iter):
    """Single trial of deflated SLQ; returns L1 error against exact_density."""
    # Phase 1: build Q ≈ top-s eigenspace (s HVPs), compute HQ (s more HVPs)
    S = torch.randn(d, s)
    Y = torch.stack([calc._hessian_vector_product(S[:, i]) for i in range(s)], dim=1)
    Q, _ = torch.linalg.qr(Y, mode='reduced')                    # (d, s)
    HQ = torch.stack([calc._hessian_vector_product(Q[:, i]) for i in range(s)], dim=1)
    QTHQ = Q.t() @ HQ                                             # (s, s)
    large_eigs = torch.linalg.eigvalsh(QTHQ).numpy()

    # Deflated HVP: H'v = (I−QQᵀ)H(I−QQᵀ)v
    #             = Hv − HQ(Qᵀv) − Q(QᵀHv) + Q(QᵀHQ Qᵀv)
    def deflated_hvp(v):
        Hv = calc._hessian_vector_product(v)
        QTv  = Q.t() @ v
        QTHv = Q.t() @ Hv
        return Hv - HQ @ QTv - Q @ QTHv + Q @ (QTHQ @ QTv)

    # Phase 2: SLQ on deflated operator with probes projected out of span(Q)
    slq_vals, slq_wts = slq_estimate(deflated_hvp, d, n_v, n_iter, deflate_Q=Q)

    # Combine: large part contributes s/d, deflated SLQ part contributes (d-s)/d
    all_vals = np.concatenate([large_eigs, slq_vals])
    all_wts  = np.concatenate([np.ones(s) / d, slq_wts * (d - s) / d])
    density = kde_density(all_vals, all_wts, grid, sigma)
    return np.sum(np.abs(density - exact_density)) * dx


def run_slq_experiment(n_runs=8, n_iter=10,
                       budgets=(60, 120, 200),
                       s_values=(5, 10)):
    """
    Sweep over HVP budgets and subspace sizes s to compare standard SLQ vs
    deflated SLQ (Hutch++-SLQ) for spectral density estimation.

    Budget allocation:
      Standard SLQ:  n_v = budget // n_iter probes
      Deflated SLQ:  2s setup HVPs + n_v' = (budget − 2s) // n_iter probes

    Quality metric: L1 distance between estimated KDE density and exact density
    computed from the full d×d Hessian eigendecomposition.
    """
    print("\n" + "=" * 62)
    print("SLQ vs Deflated SLQ (Hutch++-SLQ) — Spectrum Estimation")
    print("=" * 62)

    torch.manual_seed(42)
    model = TinyMLP(input_dim=4, hidden=8, output_dim=2)
    dataloader = make_dataloader(input_dim=4, output_dim=2)
    calc = NNHessianCalculator(
        model=model,
        loss_fn=nn.MSELoss(),
        dataloader=dataloader,
        external_load_batch_func=load_batch,
        device="cpu",
    )
    d = calc.total_params

    # Exact Hessian (d HVPs)
    print(f"\nBuilding exact {d}×{d} Hessian ({d} HVP calls)...")
    eye = torch.eye(d)
    H_mat = torch.stack([calc._hessian_vector_product(eye[i]) for i in range(d)], dim=1)
    exact_eigs = np.linalg.eigvalsh(H_mat.numpy())

    lam_min, lam_max = exact_eigs.min(), exact_eigs.max()
    span = lam_max - lam_min
    grid  = np.linspace(lam_min - 0.15 * span, lam_max + 0.15 * span, 300)
    sigma = 0.05 * span
    dx    = grid[1] - grid[0]
    exact_density = kde_density(exact_eigs, np.ones(d) / d, grid, sigma)

    print(f"\nEigenvalue range: [{lam_min:.3f}, {lam_max:.3f}]  "
          f"(top-5: {sorted(exact_eigs)[-5:][::-1]})\n")

    # ── header ──────────────────────────────────────────────────────────
    methods = ["SLQ"] + [f"Def-{s}" for s in s_values]
    col_w = 14
    hdr = f"{'Budget':<8}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(hdr)
    print("-" * len(hdr))

    for budget in budgets:
        n_v_std = max(1, budget // n_iter)

        # ── standard SLQ ─────────────────────────────────────────────
        std_l1 = []
        for run in range(n_runs):
            torch.manual_seed(run * 13)
            vals, wts = slq_estimate(calc._hessian_vector_product, d, n_v_std, n_iter)
            density = kde_density(vals, wts, grid, sigma)
            std_l1.append(np.sum(np.abs(density - exact_density)) * dx)

        cells = [f"{np.mean(std_l1):.4f}±{np.std(std_l1):.4f}"]

        # ── deflated SLQ for each s ───────────────────────────────────
        for s in s_values:
            n_v_def = max(1, (budget - 2 * s) // n_iter)
            if n_v_def < 1:
                cells.append(f"{'—':>{col_w}}")
                continue

            def_l1 = []
            for run in range(n_runs):
                torch.manual_seed(run * 13)
                l1 = _deflated_slq_l1(calc, d, exact_density, grid, sigma, dx,
                                      s=s, n_v=n_v_def, n_iter=n_iter)
                def_l1.append(l1)
            cells.append(f"{np.mean(def_l1):.4f}±{np.std(def_l1):.4f}")

        print(f"{budget:<8}" + "".join(f"{c:>{col_w}}" for c in cells))

    print(f"\nColumns: mean L1 ± std over {n_runs} runs  (lower is better)")
    print(f"SLQ = standard,  Def-s = Deflated SLQ with subspace size s")
    print(f"Budget = total HVP calls; n_iter = {n_iter} Lanczos steps per probe")


def make_outlier_bulk_hvp(d, s_true, gap, seed=0):
    """
    Synthetic PSD matrix with:
      - s_true outlier eigenvalues near `gap`  (separated cluster)
      - d - s_true bulk eigenvalues uniformly spaced in [0.1, 1.0]  (hard continuous bulk)

    The bulk is intentionally dense and continuous — Lanczos needs many steps to
    resolve it, unlike a two-level spectrum where it converges in 2 steps.

    Returns (hvp_fn, exact_eigs).
    """
    torch.manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(d, d))
    bulk = torch.linspace(0.1, 1.0, d - s_true)
    outliers = torch.linspace(gap * 0.85, gap * 1.15, s_true)
    eigs = torch.cat([outliers, bulk])           # large first, then bulk
    exact_eigs = eigs.numpy()

    def hvp_fn(v):
        return U @ (eigs * (U.t() @ v))

    return hvp_fn, exact_eigs


def run_slq_crossover(d=60, s=5, budget=100, n_iter=5, n_runs=40,
                      gaps=(1, 2, 5, 10, 20, 50)):
    """
    Sweep the spectral gap to find where deflated SLQ outperforms standard SLQ.

    Uses a realistic spectrum: s outlier eigenvalues near `gap` plus a continuous
    bulk in [0.1, 1.0].  With n_iter small, standard Lanczos wastes steps
    separating the outliers, while the deflated Lanczos spends all steps on the
    bulk.  The crossover appears once the gap is large enough that:
      (a) Q reliably recovers the outlier subspace, AND
      (b) removing outliers makes the deflated Lanczos significantly cheaper.

    Budget:
      Standard SLQ  : n_v  = budget // n_iter probes
      Deflated SLQ  : 2s setup + n_v' = (budget − 2s) // n_iter probes
    """
    print("\n" + "=" * 72)
    print("Crossover Analysis: when does Deflated SLQ beat standard SLQ?")
    print("=" * 72)
    print(f"\nd={d}, s={s}, budget={budget}, n_iter={n_iter} (intentionally small), n_runs={n_runs}")
    print(f"Spectrum: {s} outliers near `gap`, {d-s} bulk eigenvalues in [0.1, 1.0]\n")

    n_v_std = max(1, budget // n_iter)
    n_v_def = max(1, (budget - 2 * s) // n_iter)
    print(f"  Standard SLQ : {n_v_std} probes × {n_iter} steps = {n_v_std * n_iter} HVPs")
    print(f"  Deflated SLQ : 2×{s} setup + {n_v_def} probes × {n_iter} steps "
          f"= {2*s + n_v_def*n_iter} HVPs\n")

    hdr = (f"{'Gap':>6}  {'Conc%':>7}  "
           f"{'SLQ (mean±std)':>18}  {'Def-SLQ (mean±std)':>20}  {'Winner':>10}")
    print(hdr)
    print("-" * len(hdr))

    for gap in gaps:
        hvp_fn, exact_eigs = make_outlier_bulk_hvp(d, s, gap, seed=42)

        # Trace concentration: fraction of Tr(H) held by the outlier eigenvalues
        conc_pct = 100 * exact_eigs[:s].sum() / exact_eigs.sum()

        # KDE: bandwidth relative to bulk width (not the full range), so the
        # bulk density is well-resolved regardless of gap size
        sigma = 0.05 * (exact_eigs[s:].max() - exact_eigs[s:].min() + 1e-6)
        grid  = np.linspace(-0.2, gap * 1.3, 500)
        dx    = grid[1] - grid[0]
        exact_density = kde_density(exact_eigs, np.ones(d) / d, grid, sigma)

        std_l1, def_l1 = [], []

        for run in range(n_runs):
            torch.manual_seed(run)

            # ── Standard SLQ ─────────────────────────────────────────
            vals, wts = slq_estimate(hvp_fn, d, n_v_std, n_iter)
            density = kde_density(vals, wts, grid, sigma)
            std_l1.append(np.sum(np.abs(density - exact_density)) * dx)

            # ── Deflated SLQ ──────────────────────────────────────────
            S  = torch.randn(d, s)
            Y  = torch.stack([hvp_fn(S[:, i]) for i in range(s)], dim=1)
            Q, _ = torch.linalg.qr(Y, mode='reduced')
            HQ   = torch.stack([hvp_fn(Q[:, i]) for i in range(s)], dim=1)
            QTHQ = Q.t() @ HQ
            large_eigs = torch.linalg.eigvalsh(QTHQ).numpy()

            def deflated_hvp(v, _Q=Q, _HQ=HQ, _QTHQ=QTHQ):
                Hv   = hvp_fn(v)
                QTv  = _Q.t() @ v
                QTHv = _Q.t() @ Hv
                return Hv - _HQ @ QTv - _Q @ QTHv + _Q @ (_QTHQ @ QTv)

            slq_vals, slq_wts = slq_estimate(deflated_hvp, d, n_v_def, n_iter,
                                             deflate_Q=Q)
            all_vals = np.concatenate([large_eigs, slq_vals])
            all_wts  = np.concatenate([np.ones(s) / d, slq_wts * (d - s) / d])
            density  = kde_density(all_vals, all_wts, grid, sigma)
            def_l1.append(np.sum(np.abs(density - exact_density)) * dx)

        s_mean, s_std = np.mean(std_l1), np.std(std_l1)
        d_mean, d_std = np.mean(def_l1), np.std(def_l1)
        winner = "Def-SLQ ✓" if d_mean < s_mean else "SLQ"
        print(f"{gap:>6}  {conc_pct:>6.1f}%  "
              f"{s_mean:>8.4f}±{s_std:.4f}  "
              f"{d_mean:>9.4f}±{d_std:.4f}  {winner:>10}")

    print(f"\nConc% = fraction of Tr(H) held by the top-{s} outlier eigenvalues.")
    print(f"n_iter={n_iter} is intentionally small: standard Lanczos is under-converged,")
    print("so deflation's advantage (all steps go to the bulk) shows up at large gaps.")


if __name__ == "__main__":
    # run_experiment()
    # run_slq_experiment()
    run_slq_crossover()
