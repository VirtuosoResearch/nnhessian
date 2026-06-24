"""
    python src/estimate_top_quantile_spectrum.py \
        --replot_from src/logs/top_quantile_density_data.npz
"""

import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nnhessian.hessian import hutchpp_top_quantile_density, slq_estimate
from nnhessian.utils import kde_density
import matplotlib as mpl
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

# --------------------------------------------------------------------------- #
# Synthetic Hessian with a known (planted) spectrum                            #
# --------------------------------------------------------------------------- #
def make_synthetic_spectrum(d, n_top, lam, seed=0):
    """
    Build a planted spectrum with a heavy bulk near 0 and ``n_top`` outliers
    above ``lam``.

    Returns the sorted eigenvalues (numpy float64, length d).
    """
    rng = np.random.default_rng(seed)
    n_bulk = d - n_top

    # Bulk: skewed, concentrated below lam (a couple of values creep up toward
    # lam to make the count-above-lam boundary non-trivial).
    bulk = rng.gamma(shape=1.6, scale=0.55, size=n_bulk)          # ~[0, 4]
    bulk = np.clip(bulk, 0.0, lam - 1.0)

    # Top quantile: n_top outliers spread geometrically in (lam, top_max],
    # so they decay -- the regime where SLQ's tail variance bites.
    top = np.geomspace(lam + 1.0, 60.0, n_top)

    eig = np.concatenate([bulk, top])
    eig.sort()
    return eig.astype(np.float64)


def make_hvp(eigvals, seed=0):
    """
    Matrix-free HVP for H = U diag(eigvals) U^T with a random orthogonal U.

    We never need H explicitly; v -> U (Lambda * (U^T v)) is two matvecs. Returns
    (hvp_fn, d). hvp_fn consumes/returns 1-D float64 torch tensors.
    """
    d = len(eigvals)
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    U, _ = torch.linalg.qr(A)                                     # random orthogonal
    Lam = torch.from_numpy(eigvals).to(torch.float64)

    def hvp_fn(v):
        v = v.to(torch.float64)
        return U @ (Lam * (U.t() @ v))

    return hvp_fn, d


# --------------------------------------------------------------------------- #
# Metrics                                                                       #
# --------------------------------------------------------------------------- #
def mass_above(values, weights, lam):
    """Estimated fraction of eigenvalues strictly above ``lam`` (a hard count)."""
    return float(np.asarray(weights)[np.asarray(values) > lam].sum())


def density_l1_above(true_density, values, weights, grid, sigma, lam):
    """L1 distance between the (sigma-smoothed) estimated and true densities,
    integrated over the top-quantile region t > lam."""
    est_density = kde_density(np.asarray(values), np.asarray(weights), grid, sigma)
    mask = grid > lam
    dx = grid[1] - grid[0]
    return float(np.sum(np.abs(est_density[mask] - true_density[mask])) * dx)


# --------------------------------------------------------------------------- #
# Left figure: top-quantile density profile (drawing + reload-from-data)       #
# --------------------------------------------------------------------------- #
def draw_density_panel(ax, grid, true_density, slq_curve, dfl_curve, top_eigs,
                       lam, curve_budget, seeds):
    """Render the top-quantile density figure into ``ax`` from plain arrays.

    Zooms into t > lam (the bulk near 0 sits off-axis to the left) so the
    y-scale is set by the outlier bumps we care about.
    """
    region = grid >= lam
    ymax = max(true_density[region].max(), slq_curve[region].max(),
               dfl_curve[region].max())
    ax.fill_between(grid, true_density, color="0.8", alpha=0.5, label=r"$\mathrm{True~density}$")
    ax.plot(grid, slq_curve, color="tab:orange", lw=3, ls="dashdot", label=r"$\mathrm{Standard~SLQ}$")
    ax.plot(grid, dfl_curve, color="tab:blue", lw=2, label=r"$\mathrm{Deflated~SLQ}$")
    ax.axvline(lam, color="k", ls="--", lw=1)
    eig_heights = np.interp(top_eigs, grid, true_density)
    ax.vlines(top_eigs, 0.0, eig_heights, color="k", lw=0.6, ls="--", alpha=0.4,
              zorder=1, label=r"$\mathrm{True~top~eigenvalues}$")
    xmax = (float(top_eigs.max()) if len(top_eigs) else float(grid.max())) + 3
    ax.set_xlim(lam - 2.00, 63)
    ax.set_xticks([lam, 20, 40, 60])                    # sparse ticks, keep one at lambda
    ax.set_xlim(lam - 2.00, 63)                         # set_xticks rescales the view; restore it
    ax.set_ylim(-0.00005, ymax * 1.3)
    ax.locator_params(axis="y", nbins=5)               # fewer y ticks
    ax.text(lam + 1.05, ymax * 1.14, r"$\lambda$", fontsize=26)
    ax.tick_params(axis="both", labelsize=24)
    ax.set_xlabel(r"$\mathrm{Eigenvalue}$", fontsize=26)
    ax.set_ylabel(r"$\mathrm{Spectral~density}$", fontsize=26)
    #ax.set_title(f"Top-quantile density (eigenvalues > {lam}, ~{curve_budget} "
    #             f"HVPs, avg of {seeds} seeds)")
    ax.legend(fontsize=25)
    ax.grid(True, alpha=0.3)


def plot_density_from_data(npz_path, out_pdf):
    """Reproduce the top-quantile density figure purely from a saved *_data.npz.

    Demonstrates that the stored data is self-contained for the left figure.
    """
    z = np.load(npz_path)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    draw_density_panel(
        ax, z["grid"], z["true_density"], z["slq_curve"], z["dfl_curve"],
        z["top_eigenvalues"], float(z["lam"]), int(z["curve_budget"]),
        int(z["seeds"]),
    )
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150)
    plt.close(fig)
    return out_pdf


# --------------------------------------------------------------------------- #
# Budget-matched estimators                                                     #
# --------------------------------------------------------------------------- #
def run_standard_slq(hvp_fn, d, budget, n_iter, dtype, generator):
    """Standard SLQ using `budget` HVPs (n_v = budget // n_iter probes)."""
    n_v = max(1, budget // n_iter)
    vals, wts = slq_estimate(
        hvp_fn, d, n_v, n_iter, dtype=dtype, reorth=True, generator=generator
    )
    return {"values": vals, "weights": wts, "n_hvp": n_v * n_iter}


def run_deflated_slq(hvp_fn, d, lam, budget, n_iter, n_power, frac_deflate,
                     dtype, generator):
    """Hutch++-style deflated SLQ using ~`budget` HVPs.

    Splits the budget: a fraction ``frac_deflate`` builds/reads the deflation
    sketch (rank s costs s*(2+n_power) HVPs), the remainder runs residual SLQ.
    """
    deflate_budget = int(round(frac_deflate * budget))
    s = max(1, deflate_budget // (2 + n_power))
    s = min(s, d - 1)
    residual_budget = budget - s * (2 + n_power)
    n_v = max(1, residual_budget // n_iter)
    return hutchpp_top_quantile_density(
        hvp_fn, d, lam, sketch_rank=s, n_v=n_v, n_iter=n_iter,
        n_power=n_power, dtype=dtype, reorth=True, generator=generator,
    )


# --------------------------------------------------------------------------- #
# Experiment                                                                    #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=1500)
    p.add_argument("--n_top", type=int, default=40, help="# planted eigenvalues > lam")
    p.add_argument("--lam", type=float, default=5.0, help="top-quantile threshold")
    p.add_argument("--n_iter", type=int, default=30, help="Lanczos steps per probe")
    p.add_argument("--n_power", type=int, default=1, help="subspace-iteration steps")
    p.add_argument("--frac_deflate", type=float, default=0.7,
                   help="fraction of budget spent on the deflation sketch")
    p.add_argument("--budgets", type=int, nargs="+",
                   default=[300, 600, 1200, 2400], help="HVP budgets to compare")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.6, help="density KDE bandwidth")
    p.add_argument("--curve_budget", type=int, default=900,
                   help="budget used for the density-profile figure")
    p.add_argument("--outdir", type=str,
                   default=os.path.join(os.path.dirname(__file__), "logs"),
                   help="directory for figures and saved data")
    p.add_argument("--replot_from", type=str, default=None,
                   help="path to a saved *_data.npz; if given, just redraw the "
                        "density figure from it (no recomputation) and exit")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Fast path: reproduce the left (density) figure purely from stored data.
    if args.replot_from is not None:
        out_pdf = os.path.join(args.outdir, "top_quantile_density.pdf")
        plot_density_from_data(args.replot_from, out_pdf)
        print(f"Redrew density figure from {args.replot_from} -> {out_pdf}")
        return

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    dtype = torch.float64

    # ---- ground truth -----------------------------------------------------
    eigvals = make_synthetic_spectrum(args.d, args.n_top, args.lam, seed=12345)
    hvp_fn, d = make_hvp(eigvals, seed=12345)
    true_mass = float((eigvals > args.lam).mean())
    quantile = 1.0 - true_mass
    print(f"d={d}  threshold lam={args.lam}  ->  {int(round(true_mass*d))} "
          f"eigenvalues above lam  (true mass={true_mass:.4f}, "
          f"i.e. the top {100*true_mass:.2f}% / {quantile:.4f}-quantile)")
    print(f"top eigenvalues: max={eigvals.max():.2f}, "
          f"min-above-lam={eigvals[eigvals > args.lam].min():.2f}")

    grid = np.linspace(eigvals.min() - 1.0, eigvals.max() + 3.0, 800)
    true_density = kde_density(eigvals, np.full(d, 1.0 / d), grid, args.sigma)

    # ---- accuracy vs budget ----------------------------------------------
    methods = {"Standard SLQ": run_standard_slq, "Deflated SLQ (Hutch++)": run_deflated_slq}
    # results[method][budget] = list over seeds of (mass_err, l1_err, mass_hat, n_hvp)
    results = {m: {b: [] for b in args.budgets} for m in methods}

    for budget in args.budgets:
        for seed in range(args.seeds):
            gen = torch.Generator().manual_seed(1000 + seed)
            slq = run_standard_slq(hvp_fn, d, budget, args.n_iter, dtype, gen)
            gen = torch.Generator().manual_seed(1000 + seed)
            dfl = run_deflated_slq(hvp_fn, d, args.lam, budget, args.n_iter,
                                   args.n_power, args.frac_deflate, dtype, gen)
            for name, out in (("Standard SLQ", slq), ("Deflated SLQ (Hutch++)", dfl)):
                mh = mass_above(out["values"], out["weights"], args.lam)
                l1 = density_l1_above(true_density, out["values"], out["weights"],
                                      grid, args.sigma, args.lam)
                results[name][budget].append((abs(mh - true_mass), l1, mh, out["n_hvp"]))

    # ---- print table ------------------------------------------------------
    print("\n" + "=" * 84)
    print("Accuracy of the top-quantile density estimate (eigenvalues > "
          f"{args.lam}), mean +/- std over {args.seeds} seeds")
    print(f"true mass above lam = {true_mass:.4f}  "
          f"(= {int(round(true_mass*d))}/{d} eigenvalues)")
    print("=" * 84)
    header = (f"{'method':<24}{'~HVPs':>7}{'mass_hat':>14}"
              f"{'|mass err|':>13}{'density-L1>lam':>16}")
    for budget in args.budgets:
        print(f"\n-- budget ~ {budget} HVPs " + "-" * 60)
        print(header)
        for name in methods:
            arr = np.array(results[name][budget])               # (seeds, 4)
            mass_err = arr[:, 0]; l1 = arr[:, 1]; mass_hat = arr[:, 2]
            n_hvp = int(arr[0, 3])
            print(f"{name:<24}{n_hvp:>7}"
                  f"{mass_hat.mean():>9.4f}+-{mass_hat.std():<4.3f}"
                  f"{mass_err.mean():>8.4f}+-{mass_err.std():<4.3f}"
                  f"{l1.mean():>11.4f}+-{l1.std():<4.3f}")

    # summary at the largest budget (use density-L1, which is robust to the
    # deflated mass error collapsing to floating-point zero)
    big = args.budgets[-1]
    m_slq = np.array(results["Standard SLQ"][big])[:, 0].mean()
    m_dfl = np.array(results["Deflated SLQ (Hutch++)"][big])[:, 0].mean()
    l_slq = np.array(results["Standard SLQ"][big])[:, 1].mean()
    l_dfl = np.array(results["Deflated SLQ (Hutch++)"][big])[:, 1].mean()
    print(f"\nAt ~{big} HVPs:")
    dfl_mass = "exact to numerical precision" if m_dfl < 1e-6 else f"{m_dfl:.4f}"
    print(f"  top-quantile mass error : standard SLQ {m_slq:.4f}  vs  "
          f"deflated SLQ {dfl_mass}")
    if l_dfl > 0:
        print(f"  density-L1 above lam    : standard SLQ {l_slq:.4f}  vs  "
              f"deflated SLQ {l_dfl:.4f}  ({l_slq / l_dfl:.1f}x smaller)")

    # ===================================================================== #
    # Figure 1 (left): top-quantile density profile at curve_budget          #
    # ===================================================================== #
    cb = args.curve_budget
    slq_curve = np.zeros_like(grid)
    dfl_curve = np.zeros_like(grid)
    for seed in range(args.seeds):
        gen = torch.Generator().manual_seed(2000 + seed)
        slq = run_standard_slq(hvp_fn, d, cb, args.n_iter, dtype, gen)
        gen = torch.Generator().manual_seed(2000 + seed)
        dfl = run_deflated_slq(hvp_fn, d, args.lam, cb, args.n_iter,
                               args.n_power, args.frac_deflate, dtype, gen)
        slq_curve += kde_density(slq["values"], slq["weights"], grid, args.sigma)
        dfl_curve += kde_density(dfl["values"], dfl["weights"], grid, args.sigma)
    slq_curve /= args.seeds
    dfl_curve /= args.seeds
    top_eigs = eigvals[eigvals > args.lam]

    # --- persist the data needed to reproduce this figure ---
    data_npz = os.path.join(args.outdir, "top_quantile_density_data.npz")
    np.savez(
        data_npz,
        grid=grid, true_density=true_density,
        slq_curve=slq_curve, dfl_curve=dfl_curve,
        top_eigenvalues=top_eigs,
        lam=args.lam, sigma=args.sigma, curve_budget=cb,
        seeds=args.seeds, true_mass=true_mass, d=d,
    )
    data_csv = os.path.join(args.outdir, "top_quantile_density_curves.csv")
    np.savetxt(
        data_csv,
        np.column_stack([grid, true_density, slq_curve, dfl_curve]),
        delimiter=",", header="grid,true_density,slq_curve,dfl_curve", comments="",
    )
    print(f"\nSaved left-figure data to:\n  {data_npz}\n  {data_csv}")

    # --- draw figure 1 from those arrays ---
    fig1, ax1 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    draw_density_panel(ax1, grid, true_density, slq_curve, dfl_curve, top_eigs,
                       args.lam, cb, args.seeds)
    out_density = os.path.join(args.outdir, "top_quantile_density.pdf")
    fig1.savefig(out_density, dpi=150)
    plt.close(fig1)
    print(f"Saved density figure to {out_density}")

    # ===================================================================== #
    # Figure 2 (right): top-quantile mass error vs budget                    #
    # ===================================================================== #
    fig2, ax2 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = {"Standard SLQ": "tab:orange", "Deflated SLQ (Hutch++)": "tab:blue"}
    for name in methods:
        xs = [int(np.array(results[name][b])[0, 3]) for b in args.budgets]
        errs = np.array([np.array(results[name][b])[:, 0] for b in args.budgets])
        ax2.errorbar(xs, errs.mean(1), yerr=errs.std(1), marker="o", capsize=3,
                     color=colors[name], label=name)
    ax2.set_yscale("log")
    ax2.set_xlabel("HVP budget")
    ax2.set_ylabel(r"|estimated $-$ true mass above $\lambda$|")
    ax2.set_title(f"Top-quantile mass error vs. budget (eigenvalues > {args.lam}, "f"true mass {true_mass:.3f})")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)
    out_error = os.path.join(args.outdir, "top_quantile_mass_error.pdf")
    fig2.savefig(out_error, dpi=150)
    plt.close(fig2)
    print(f"Saved mass-error figure to {out_error}")


if __name__ == "__main__":
    main()
