from typing import Iterable, Callable, Optional, Union, List, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import math
import re
from fnmatch import fnmatch


ParamSelector = Union[str, re.Pattern, Callable[[str, nn.Parameter], bool], nn.Parameter]


def lanczos_tridiag(
    hvp_fn: Callable, v0: torch.Tensor, n_iter: int, reorth: bool = False
) -> np.ndarray:
    """
    Run n_iter steps of the Lanczos 3-term recurrence starting from unit vector v0.

    Args:
        hvp_fn: callable v -> H v (returns a CPU tensor of the same shape as v).
        v0: starting vector (need not be unit-norm).
        n_iter: number of Lanczos steps.
        reorth: if True, full reorthogonalization of each new Lanczos vector
                against all previous ones (run twice for numerical stability).
                Costs O(n_iter^2 * d) but keeps the Ritz values trustworthy at
                larger n_iter / in low precision; the default (False) keeps the
                cheap 3-term recurrence used elsewhere.

    Returns:
        T: symmetric tridiagonal numpy array of shape (n_iter, n_iter).
    """
    alpha = np.zeros(n_iter)
    beta  = np.zeros(max(n_iter - 1, 0))

    basis = [] if reorth else None
    v_prev = torch.zeros_like(v0)
    v = v0 / v0.norm()
    b = 0.0

    for j in range(n_iter):
        if reorth:
            basis.append(v)
        Hv = hvp_fn(v)
        a  = torch.dot(Hv, v).item()
        alpha[j] = a
        r = Hv - a * v - b * v_prev

        if reorth:
            # Full reorthogonalization (twice) against the stored basis.
            for _ in range(2):
                for q in basis:
                    r = r - q * torch.dot(q, r)

        if j < n_iter - 1:
            b = r.norm().item()
            beta[j] = b
            if b > 1e-10:
                v_next = r / b
            else:
                # Near-happy breakdown: restart orthogonal to the basis built so far
                v_next = torch.randn_like(v)
                ortho = basis if reorth else [v]
                for q in ortho:
                    v_next = v_next - q * torch.dot(q, v_next)
                nrm = v_next.norm()
                v_next = v_next / nrm if nrm > 1e-10 else torch.zeros_like(v)
            v_prev, v = v, v_next

    return np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)


def slq_estimate(
    hvp_fn: Callable,
    d: int,
    n_v: int,
    n_iter: int,
    deflate_Q: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    reorth: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Lanczos Quadrature spectral density estimator.

    Args:
        hvp_fn: callable v -> H v (1-D CPU tensor in, 1-D CPU tensor out).
        d: parameter-space dimension.
        n_v: number of random probe vectors.
        n_iter: Lanczos steps per probe.
        deflate_Q: optional (d, s) orthonormal matrix. When provided each probe
                   is projected out of span(deflate_Q) before running Lanczos,
                   targeting the residual spectrum.
        dtype: probe dtype (must match what hvp_fn consumes/returns).
        reorth: full reorthogonalization inside Lanczos (see lanczos_tridiag).
        generator: optional torch.Generator for reproducible probes.

    Returns:
        values:  (n_valid * n_iter,) Ritz values.
        weights: (n_valid * n_iter,) quadrature weights, averaged over probes
                 so the total sums to ≈ 1.
    """
    all_eigs, all_weights = [], []
    for _ in range(n_v):
        v = torch.randn(d, dtype=dtype, generator=generator)
        if deflate_Q is not None:
            v = v - deflate_Q @ (deflate_Q.t() @ v)
            nrm = v.norm()
            if nrm < 1e-10:
                continue
            v = v / nrm
        else:
            v = v / v.norm()

        T = lanczos_tridiag(hvp_fn, v, n_iter, reorth=reorth)
        eigs, U = np.linalg.eigh(T)
        all_eigs.append(eigs)
        all_weights.append(U[0] ** 2)

    if not all_eigs:
        return np.array([]), np.array([])
    n_valid = len(all_eigs)
    return np.concatenate(all_eigs), np.concatenate(all_weights) / n_valid


def randomized_subspace(
    hvp_fn: Callable,
    d: int,
    rank: int,
    n_power: int = 0,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized sketch of the dominant eigenspace of H (the Hutch++ "Q" step).

    Draws a Gaussian test matrix Omega (d, rank), forms Q = orth(H Omega), and
    optionally refines it with `n_power` steps of subspace (power) iteration so
    Q better captures the top eigenvectors of H.

    Args:
        hvp_fn: callable v -> H v.
        d: dimension.
        rank: sketch size s (number of columns of Q).
        n_power: subspace-iteration steps (0 = plain Hutch++ sketch).
        dtype: working dtype.
        generator: optional torch.Generator for reproducibility.

    Returns:
        Q:  (d, rank) orthonormal basis approximating the top-rank eigenspace.
        HQ: (d, rank) equal to H @ Q.
        B:  (rank, rank) symmetric matrix Q^T H Q whose eigenvalues are the
            Ritz approximations of the top eigenvalues of H.

    HVP cost: rank * (2 + n_power).
    """
    def matmat(M: torch.Tensor) -> torch.Tensor:
        return torch.stack([hvp_fn(M[:, i].contiguous()) for i in range(M.shape[1])], dim=1)

    Omega = torch.randn(d, rank, dtype=dtype, generator=generator)
    Y = matmat(Omega)
    for _ in range(n_power):
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        Y = matmat(Q)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    HQ = matmat(Q)
    B = Q.t() @ HQ
    B = 0.5 * (B + B.t())
    return Q, HQ, B


def hutchpp_top_quantile_density(
    hvp_fn: Callable,
    d: int,
    lam: float,
    sketch_rank: int,
    n_v: int,
    n_iter: int,
    n_power: int = 1,
    dtype: torch.dtype = torch.float64,
    reorth: bool = True,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """
    Hutch++-style estimator of the spectral density restricted to the TOP
    QUANTILE -- the eigenvalues above a threshold ``lam``.

    Based on Hutch++ (Meyer, Musco, Musco, Woodruff, arXiv:2010.09649): deflate
    the dominant eigenspace with a randomized sketch, resolve the large
    eigenvalues *exactly* as the Ritz values of B = Q^T H Q (each an exact point
    mass of weight 1/d), then run SLQ only on the deflated residual
    ``P H P`` with ``P = I - Q Q^T`` to fill in the rest of the density.

    The combined spectral density estimate is
        phi_hat(t) = (1/d) sum_j delta(t - theta_j)            [exact top part]
                   + ((d - s)/d) sum_k w_k delta(t - nu_k)     [residual SLQ]
    and the top-quantile mass (fraction of eigenvalues above ``lam``) is
        mu_hat(lam) = (#{theta_j > lam})/d
                    + ((d - s)/d) sum_{nu_k > lam} w_k.
    Because the sketch captures the eigenvalues above ``lam`` whenever
    ``s`` exceeds their count, the residual term there vanishes and the
    top-quantile estimate becomes essentially exact and low-variance -- unlike
    plain SLQ, whose variance is dominated by exactly these few large outliers.

    Args:
        hvp_fn: callable v -> H v (1-D tensor in/out, dtype == ``dtype``).
        d: dimension.
        lam: threshold; the "top quantile" is {eigenvalue > lam}.
        sketch_rank: deflation rank s.
        n_v: residual SLQ probe count.
        n_iter: residual SLQ Lanczos steps.
        n_power: subspace-iteration steps for the sketch (default 1).
        dtype: working dtype (float64 recommended for accurate Ritz values).
        reorth: full reorthogonalization inside residual Lanczos.
        generator: optional torch.Generator for reproducibility.

    Returns:
        dict with keys:
          values, weights        -- full combined density (weights sum to ~1)
          ritz_vals              -- the s exact top Ritz values
          residual_values/weights-- residual SLQ part (weights already scaled)
          mass_above             -- estimated fraction of eigenvalues > lam
          top_mass_above         -- contribution from the exact top part only
          n_hvp                  -- total Hessian-vector products used
    """
    Q, HQ, B = randomized_subspace(hvp_fn, d, sketch_rank, n_power, dtype, generator)
    ritz_vals = np.linalg.eigvalsh(B.cpu().numpy().astype(np.float64))

    def deflated_hvp(v, _Q=Q, _HQ=HQ, _B=B):
        # (I - QQ^T) H (I - QQ^T) v  without ever forming H.
        Hv = hvp_fn(v)
        Qtv = _Q.t() @ v
        QtHv = _Q.t() @ Hv
        return Hv - _HQ @ Qtv - _Q @ QtHv + _Q @ (_B @ Qtv)

    res_vals, res_wts = slq_estimate(
        deflated_hvp, d, n_v, n_iter, deflate_Q=Q,
        dtype=dtype, reorth=reorth, generator=generator,
    )

    s = sketch_rank
    top_wts = np.full(s, 1.0 / d)
    res_wts_scaled = res_wts * (d - s) / d

    values = np.concatenate([ritz_vals, res_vals])
    weights = np.concatenate([top_wts, res_wts_scaled])

    return {
        "values": values,
        "weights": weights,
        "ritz_vals": ritz_vals,
        "residual_values": res_vals,
        "residual_weights": res_wts_scaled,
        "mass_above": float(weights[values > lam].sum()),
        "top_mass_above": float(top_wts[ritz_vals > lam].sum()),
        "n_hvp": s * (2 + n_power) + n_v * n_iter,
    }


class NNHessianCalculator():
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        dataloader: Optional[Iterable] = None,
        external_load_batch_func: Optional[Callable] = None,
        assigned_parameters: Optional[Iterable[ParamSelector]] = None,
        device: Union[str, torch.device] = "cpu",
        aggregate_method: str = "mean",
    ):
        """
        Args:
            model: PyTorch model. Will be moved to `device` and set to eval().
            loss_fn: Loss function taking (pred, target, *extras).
            dataloader: Iterable yielding batches.
            external_load_batch_func: Optional function (batch, device) -> (inputs, targets, *extras).
            assigned_parameters: Optional selectors limiting which parameters we differentiate w.r.t.
                Supports:
                  - exact name: "layer1.weight"
                  - glob: "encoder.*.weight"
                  - regex: re.compile(r"bias$")
                  - callable: lambda name, p: condition
                  - direct nn.Parameter object
            device: device string or torch.device.
            aggregate_method: how to aggregate per-example losses ("mean" or "sum").
        """
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()  # ensure device + eval mode
        self.loss_fn = loss_fn
        self.aggregate_method = aggregate_method

        self.dataloader = dataloader
        self.load_batch_func = external_load_batch_func or self._default_load_batch_func

        # Parameter selection setup
        self.assigned_parameters: Optional[List[ParamSelector]] = (
            list(assigned_parameters) if assigned_parameters is not None else None
        )

        # Cache named parameters according to selection
        self.named_params: "OrderedDict[str, nn.Parameter]" = self._get_assigned_parameters(require_grad=True)

        # Bookkeeping: sizes
        self.total_params_all: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.total_params: int = sum(p.numel() for p in self.named_params.values())

    ##############################################
    # Utils
    ##############################################

    def _get_assigned_parameters(
        self,
        require_grad: bool = True,
    ) -> "OrderedDict[str, nn.Parameter]":
        """
        Return an OrderedDict of (name -> parameter) filtered by `assigned_parameters`.
        If no selectors are provided, returns all trainable (or all, if require_grad=False) named parameters.

        Selection semantics:
          - str: exact match OR glob pattern using fnmatch (e.g., "encoder.*.weight").
          - re.Pattern: include if pattern.search(name) is True.
          - callable(name, param) -> bool.
          - nn.Parameter: include if `param is selector`.
        """
        def matches(name: str, param: nn.Parameter) -> bool:
            # No filter -> include
            if not self.assigned_parameters:
                return True

            for sel in self.assigned_parameters:
                # direct parameter identity
                if isinstance(sel, nn.Parameter):
                    if param is sel:
                        return True
                # regex
                elif isinstance(sel, re.Pattern):
                    if sel.search(name):
                        return True
                # callable
                elif callable(sel):
                    try:
                        if bool(sel(name, param)):
                            return True
                    except TypeError:
                        # allow callables that take only name
                        if bool(sel(name)):
                            return True
                # string: exact or glob
                elif isinstance(sel, str):
                    if name == sel or fnmatch(name, sel):
                        return True
            return False

        selected: List[Tuple[str, nn.Parameter]] = []
        for name, p in self.model.named_parameters():
            if require_grad and not p.requires_grad:
                continue
            if matches(name, p):
                selected.append((name, p))

        # Maintain model-defined order
        return OrderedDict(selected)

    def _hessian_vector_product(self, v=None, dataloader=None) -> torch.Tensor:
        """
        compute hessian-vector product, takes a flattened tensor as input
        shape: (sum_selected_params, )

        Uses:
            data, target, batch_size = self.load_batch_func(batch, self.device)
            output = self.model(data)
        """
        if dataloader is None:
            if self.dataloader is None:
                raise ValueError("No dataloader provided.")
            dataloader = self.dataloader

        # Choose parameter list (prefer previously selected subset if available)
        if hasattr(self, "named_params") and len(self.named_params) > 0:
            params = list(self.named_params.values())
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]

        if len(params) == 0:
            raise ValueError("No trainable parameters to differentiate.")

        device = self.device
        p_dtype = params[0].dtype
        total_param_elems = sum(p.numel() for p in params)

        d_flat = v.view(-1)
        if d_flat.numel() != total_param_elems:
            raise ValueError(
                f"Flattened vector length {d_flat.numel()} does not match "
                f"selected parameters size {total_param_elems}."
            )
        d_flat = d_flat.to(device=device, dtype=p_dtype)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        total_examples = 0
        hvp_sum_flat = torch.zeros(total_param_elems, device=device, dtype=p_dtype)

        # Clear cache before starting
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for batch in dataloader:
            # Required contract from you:
            data, target, batch_size = self.load_batch_func(batch, device)
            output = self.model(data)
            loss = self.loss_fn(output, target)

            # Ensure scalar loss; keep it simple: mean if it's per-example
            if hasattr(loss, "dim") and loss.dim() > 0:
                loss = loss.mean()

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Invalid loss detected: {loss.item()}")

            # First backward (create graph for second derivative)
            self.model.zero_grad(set_to_none=True)
            loss.backward(create_graph=True)

            # Collect first-order grads as a flat vector (zeros for unused params)
            g_chunks = []
            for p in params:
                if p.grad is None:
                    g_chunks.append(torch.zeros(p.numel(), device=device, dtype=p_dtype))
                else:
                    # Check for invalid gradients
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        raise RuntimeError(
                            f"Invalid gradient detected after first backward: "
                            f"NaN={torch.isnan(p.grad).sum().item()}, "
                            f"Inf={torch.isinf(p.grad).sum().item()}"
                        )
                    g_chunks.append(p.grad.reshape(-1))
            g_flat = torch.cat(g_chunks, dim=0)

            # Dot(g, v) and second backward to get H·v in .grad of params
            self.model.zero_grad(set_to_none=True)
            (g_flat * d_flat).sum().backward()  # builds second-order grads

            # Read out H·v and accumulate weighted by batch size
            hv_chunks = []
            for p in params:
                if p.grad is None:
                    hv_chunks.append(torch.zeros(p.numel(), device=device, dtype=p_dtype))
                else:
                    hv_chunks.append(p.grad.reshape(-1))
            hv_flat = torch.cat(hv_chunks, dim=0)

            hvp_sum_flat += hv_flat * float(batch_size)
            total_examples += int(batch_size)

            # Clear for next batch
            self.model.zero_grad(set_to_none=True)

            # Free memory
            del output, loss, g_flat, hv_flat
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if total_examples == 0:
            raise RuntimeError("Empty dataloader: total_examples is 0.")

        # Average to correspond to dataset mean loss
        hvp_mean_flat = hvp_sum_flat / float(total_examples)
        return hvp_mean_flat.detach().cpu()

    def evaluate_loss(self, model, dataloader, loss_fn, device):
        """Compute the average loss over a dataloader."""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        return total_loss / total_samples

    def hutchinson_trace(self, num_samples: int = 50, distribution: str = "rademacher",
                     dataloader=None, seed: int = None, return_std: bool = False):
        """
        Estimate Tr(H) via Hutchinson:
            Tr(H) ≈ (1/K) * Σ_i z_i^T (H z_i),
        where z_i are Rademacher (±1) or standard normal vectors.

        Args:
            num_samples: number of probe vectors K.
            distribution: "rademacher" or "normal".
            dataloader: optional dataloader to override self.dataloader.
            seed: optional RNG seed for reproducibility.
            return_std: if True, also return (sample_std, stderr).

        Returns:
            mean_estimate[, sample_std, stderr]
        """
        if dataloader is None:
            if self.dataloader is None:
                raise ValueError("No dataloader provided.")
            dataloader = self.dataloader

        # Gather the parameter list we differentiate w.r.t.
        if hasattr(self, "named_params") and len(self.named_params) > 0:
            params = list(self.named_params.values())
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]

        if len(params) == 0:
            raise ValueError("No trainable/selected parameters found.")

        device = self.device
        p_dtype = params[0].dtype
        n_params = sum(p.numel() for p in params)

        # RNG (on correct device for reproducibility across devices)
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)

        estimates = []

        for _ in range(num_samples):
            if distribution.lower() in ("rademacher", "rad"):
                z = torch.randint(0, 2, (n_params,), generator=g, device=device).to(dtype=p_dtype)
                z = z * 2 - 1  # {0,1} -> {-1,+1}
            elif distribution.lower() in ("normal", "gaussian"):
                z = torch.randn(n_params, generator=g, device=device, dtype=p_dtype)
            else:
                raise ValueError("distribution must be 'rademacher' or 'normal'.")

            Hz = self._hessian_vector_product(z, dataloader=dataloader)  # returns CPU tensor
            # Move z to CPU for the dot-product (Hz is already on CPU)
            est = float((z.detach().cpu() * Hz).sum().item())
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)

        if return_std:
            if len(estimates) > 1:
                m = mean_est
                var = sum((e - m) ** 2 for e in estimates) / (len(estimates) - 1)
                std = math.sqrt(var)
                stderr = std / math.sqrt(len(estimates))
            else:
                std, stderr = float("nan"), float("nan")
            return mean_est, std, stderr

        return mean_est

    def slq_spectrum(
        self, n_v: int, n_iter: int, dataloader=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the Hessian spectral density via Stochastic Lanczos Quadrature.

        Args:
            n_v: number of random probe vectors.
            n_iter: Lanczos steps per probe.  Total HVP cost: n_v * n_iter.
            dataloader: optional override of self.dataloader.

        Returns:
            values:  (n_v * n_iter,) Ritz values.
            weights: (n_v * n_iter,) quadrature weights (sum ≈ 1).
        """
        if dataloader is None:
            if self.dataloader is None:
                raise ValueError("No dataloader provided.")
            dataloader = self.dataloader

        hvp_fn = lambda v: self._hessian_vector_product(v, dataloader=dataloader)
        return slq_estimate(hvp_fn, self.total_params, n_v, n_iter)

    def deflated_slq_spectrum(
        self, n_v: int, n_iter: int, s: int, dataloader=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if dataloader is None:
            if self.dataloader is None:
                raise ValueError("No dataloader provided.")
            dataloader = self.dataloader

        d = self.total_params
        hvp_fn = lambda v: self._hessian_vector_product(v, dataloader=dataloader)

        S  = torch.randn(d, s)
        Y  = torch.stack([hvp_fn(S[:, i]) for i in range(s)], dim=1)
        Q, _ = torch.linalg.qr(Y, mode='reduced')                   # (d, s)
        HQ   = torch.stack([hvp_fn(Q[:, i]) for i in range(s)], dim=1)
        QTHQ = Q.t() @ HQ                                            # (s, s)
        large_eigs = torch.linalg.eigvalsh(QTHQ).numpy()

        def deflated_hvp(v, _Q=Q, _HQ=HQ, _QTHQ=QTHQ):
            Hv   = hvp_fn(v)
            QTv  = _Q.t() @ v
            QTHv = _Q.t() @ Hv
            return Hv - _HQ @ QTv - _Q @ QTHv + _Q @ (_QTHQ @ QTv)

        slq_vals, slq_wts = slq_estimate(deflated_hvp, d, n_v, n_iter, deflate_Q=Q)

        all_vals = np.concatenate([large_eigs, slq_vals])
        all_wts  = np.concatenate([np.ones(s) / d, slq_wts * (d - s) / d])
        return all_vals, all_wts

    def hutchpp_top_quantile_spectrum(
        self, lam: float, sketch_rank: int, n_v: int, n_iter: int,
        n_power: int = 1, dataloader=None,
    ) -> dict:
        """
        Hutch++-style spectral density of the top-quantile eigenvalues (> lam).

        Thin wrapper around :func:`hutchpp_top_quantile_density` using this
        model's Hessian-vector product. Runs in the model's parameter dtype
        (typically float32) without Lanczos reorthogonalization, matching the
        rest of this class; see the standalone function for the algorithm.
        """
        if dataloader is None:
            if self.dataloader is None:
                raise ValueError("No dataloader provided.")
            dataloader = self.dataloader

        params = list(self.named_params.values()) if self.named_params else \
            [p for p in self.model.parameters() if p.requires_grad]
        p_dtype = params[0].dtype

        hvp_fn = lambda v: self._hessian_vector_product(v, dataloader=dataloader)
        return hutchpp_top_quantile_density(
            hvp_fn, self.total_params, lam, sketch_rank, n_v, n_iter,
            n_power=n_power, dtype=p_dtype, reorth=False,
        )

    def max_eigenvalue_power(self, num_iters: int = 50, tol: float = 1e-5,
                         dataloader=None, init_vec: torch.Tensor = None,
                         distribution: str = "rademacher", seed: int = None,
                         which: str = "lm",  # "lm" = largest magnitude, "la" = largest algebraic
                         return_vec: bool = False):
        """
        Estimate the extreme eigenvalue of the Hessian via power iteration using HVP.

        Args:
            num_iters: maximum number of power iterations.
            tol: relative change tolerance on eigenvalue estimate for early stop.
            dataloader: optional override of self.dataloader.
            init_vec: optional 1D init vector (size == sum_selected_params).
            distribution: "rademacher" or "normal" (used if init_vec is None).
            seed: RNG seed for reproducibility.
            which: "lm" (default, largest magnitude) or "la" (largest algebraic).
            return_vec: if True, also return the (approx.) top eigenvector (unit-norm).

        Returns:
            lambda_est  [ , v_hat ]
        """
        # figure out parameter dimensions
        if hasattr(self, "named_params") and len(self.named_params) > 0:
            params = list(self.named_params.values())
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise ValueError("No trainable/selected parameters found.")

        n_params = sum(p.numel() for p in params)
        p_dtype = params[0].dtype

        if dataloader is None:
            if self.dataloader is None:
                raise ValueError("No dataloader provided.")
            dataloader = self.dataloader

        # init on CPU (HVP returns CPU tensor in this implementation)
        if init_vec is not None:
            v = init_vec.detach().flatten().to(dtype=p_dtype, device="cpu")
        else:
            g = torch.Generator(device="cpu")
            if seed is not None:
                g.manual_seed(seed)
            if distribution.lower() in ("rademacher", "rad"):
                v = torch.randint(0, 2, (n_params,), generator=g, device="cpu").to(dtype=p_dtype)
                v = v * 2 - 1
            elif distribution.lower() in ("normal", "gaussian"):
                v = torch.randn(n_params, generator=g, device="cpu", dtype=p_dtype)
            else:
                raise ValueError("distribution must be 'rademacher' or 'normal'.")

        if v.norm() == 0:
            v = torch.ones(n_params, dtype=p_dtype)

        v = v / v.norm()
        lam_prev = None

        for _ in range(num_iters):
            Hz = self._hessian_vector_product(v, dataloader=dataloader)  # CPU tensor, dtype=p_dtype

            # Rayleigh quotient (v is unit-norm)
            lam = float(torch.dot(v, Hz).item())
            lam_eval = abs(lam) if which == "lm" else lam

            if lam_prev is not None:
                rel_change = abs(lam_eval - lam_prev) / max(1.0, abs(lam_prev))
                if rel_change <= tol:
                    if return_vec:
                        return lam_eval, v
                    return lam_eval
            lam_prev = lam_eval

            # next iterate
            v = Hz
            nrm = v.norm()
            if nrm == 0:
                if return_vec:
                    return 0.0, torch.zeros_like(Hz)
                return 0.0
            v = v / nrm

        if return_vec:
            return lam_prev if lam_prev is not None else float("nan"), v
        return lam_prev if lam_prev is not None else float("nan")

    def hutch_pp_trace_estimator(self, m: int):
        """
        Estimate tr(H) with Hutch++ using your class's HVP:
            self.hessian_vector_product_with_tensor_input(v, dataloader=self.dataloader)

        Notes:
        - Uses the parameter subset in `self.named_params` (if set), otherwise all trainable params.
        - HVP is computed w.r.t. the dataset loss defined by (self.model, self.loss_fn, self.dataloader).

        Args:
            m: total number of Hessian-vector queries. Requires 2*s + g = m with
            s = (m + 2) // 4  and  g = (m - 2) // 2.

        Returns:
            A scalar tensor (CPU) estimating trace(H).
        """
        if self.dataloader is None:
            raise ValueError("self.dataloader is None. Provide a dataloader before calling Hutch++.")

        # Pick parameter list from class selection (if any)
        if hasattr(self, "named_params") and len(self.named_params) > 0:
            params = list(self.named_params.values())
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise ValueError("No trainable/selected parameters found.")

        # Problem size
        d = sum(p.numel() for p in params)
        p_dtype = params[0].dtype

        # Hutch++ split
        s = (m + 2) // 4
        g_num = (m - 2) // 2
        if s <= 0:
            raise ValueError("m too small: s must be >= 1. Choose m ≥ 2 (and typically m ≡ 2 mod 4).")
        if 2 * s + g_num != m:
            raise ValueError(f"Invalid m for Hutch++ split: need 2*s + g = m with s=(m+2)//4, g=(m-2)//2. Got s={s}, g={g_num}, 2s+g={2*s+g_num} != m={m}.")

        # We'll keep everything on CPU to avoid device ping-pong, since HVP returns CPU in your impl.
        device_cpu = torch.device("cpu")

        # Sample S ~ N(0,1) in R^(d x s)
        S = torch.randn(d, s, device=device_cpu, dtype=p_dtype)

        # Sample G ~ Rademacher in R^(d x g_num)
        if g_num > 0:
            G = (torch.randint(0, 2, (d, g_num), device=device_cpu) * 2 - 1).to(dtype=p_dtype)
        else:
            G = torch.empty(d, 0, device=device_cpu, dtype=p_dtype)

        # Helper that routes through your HVP (returns CPU tensor)
        def hvp_call(v_flat_cpu: torch.Tensor) -> torch.Tensor:
            # v_flat_cpu: shape (d,)
            return self._hessian_vector_product(v_flat_cpu, dataloader=self.dataloader)  # CPU

        # Y = H S
        Y_cols = []
        for i in range(s):
            v = S[:, i].contiguous()
            Y_cols.append(hvp_call(v).unsqueeze(1))
        Y = torch.cat(Y_cols, dim=1)  # (d, s)

        # Q = orth(Y)
        Q, _ = torch.linalg.qr(Y, mode="reduced")  # (d, s)

        # term1 = tr(Q^T H Q) = Σ_i q_i^T H q_i
        term1 = torch.zeros((), dtype=p_dtype, device=device_cpu)
        for i in range(s):
            q_i = Q[:, i].contiguous()
            Hq = hvp_call(q_i)
            term1 = term1 + torch.dot(q_i, Hq)

        # term2 = (2/(m-2)) * tr(G^T (I - QQ^T) H (I - QQ^T) G)
        term2_sum = torch.zeros((), dtype=p_dtype, device=device_cpu)
        if g_num > 0:
            # Precompute Q^T once
            Qt = Q.t()  # (s, d)
            for j in range(g_num):
                g_vec = G[:, j].contiguous()
                # r = (I - QQ^T) g
                proj = Q @ (Qt @ g_vec)
                r = g_vec - proj
                Hr = hvp_call(r)
                term2_sum = term2_sum + torch.dot(r, Hr)

        term2 = (2.0 / (m - 2)) * term2_sum if m > 2 else torch.zeros((), dtype=p_dtype, device=device_cpu)

        trace_estimate = term1 + term2

        return trace_estimate
