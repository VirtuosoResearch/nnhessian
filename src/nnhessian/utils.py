from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import math


def print_gpu_utilization():
    nvmlInit()
    memory = 0
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    memory += info.used//1024**2

def add_noise_to_model(model, noise_vector):
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        noise = noise_vector[offset: offset + numel].view_as(param)
        param.data.add_(noise)
        offset += numel

def compute_model_norm(model, p=2):
    norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in model.parameters() if p.requires_grad]), 2)
    return norm

def load_batch_func(batch, device='cpu'):
    batch = batch[0].to(device)
    inputs = batch[:, :-1]
    targets = batch
    batch_size = batch.shape[0]
    return inputs, targets, batch_size

def filter_eigenvalues(eigen_list, weight_list, threshold=None):
    filtered_eigen = []
    filtered_weight = []
    for eig, w in zip(eigen_list, weight_list):
        if threshold is not None:
            if eig >= threshold and w >= 1e-7:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
        else:
            if w >= 1e-10:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
    return filtered_eigen, filtered_weight

def renormalize_weights(filtered_weight, epsilon=1e-12):
    total = sum(filtered_weight)
    if total > 0:
        renormalized_weight = [w / (total + epsilon) for w in filtered_weight]
    else:
        renormalized_weight = [0.0 for _ in filtered_weight]
    return renormalized_weight

def construct_spectral_density(flat_eigen, flat_weight, lambdas, sigma=0.1):
    density = np.zeros_like(lambdas)
    for eig, w in zip(flat_eigen, flat_weight):
        density += w * norm.pdf(lambdas, loc=eig, scale=sigma)

    density_sum = np.sum(density) * (lambdas[1] - lambdas[0])
    density /= density_sum + 1e-12
    return density

def sqrt_with_neg_handling(arr):
    result = np.where(arr < 0, 0, np.sqrt(arr))
    return result

def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(p.requires_grad for p in module.parameters(recurse=False)):
            if "LayerNorm" not in name and "ln" not in name and "pooler" not in name:
                layers[name] = module
    return layers

def weighted_quantile(values, weights, quantile):
    """
    Compute the weighted quantile of a tensor.
    Args:
      values: 1D tensor of eigenvalues.
      weights: 1D tensor of corresponding weights.
      quantile: desired quantile (between 0 and 1).
    Returns:
      The eigenvalue threshold corresponding to the weighted quantile.
    """
    sorted_vals, sorted_indices = torch.sort(values)
    sorted_weights = weights[sorted_indices]
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = sorted_weights.sum()
    normalized_cum_weights = cumulative_weights / total_weight
    idx = torch.nonzero(normalized_cum_weights >= quantile, as_tuple=False)[0]
    threshold = sorted_vals[idx]
    return threshold

def tail_mass_fraction(values, weights, quantile=0.9):
    """
    Compute the tail mass fraction: the fraction of the total weighted mass
    (∑ p_i λ_i) that comes from eigenvalues above the weighted quantile threshold.
    """
    weights = weights / weights.sum()
    tau = weighted_quantile(values, weights, quantile)
    mask = values >= tau
    numerator = torch.sum(weights[mask] * values[mask])
    denominator = torch.sum(weights * values)
    return (numerator / denominator).item()

def weighted_gini(values, weights):
    """
    Compute the weighted Gini coefficient for the eigenvalue distribution.
    First, normalize weights to obtain a probability distribution:
      q_i = weights_i / (∑_j weights_j).
    Then, compute:
      G = (∑_{i,j} q_i q_j |λ_i - λ_j|) / (2 μ),
    where μ = ∑_i q_i λ_i.
    """
    q = weights / weights.sum()
    mu = (values * q).sum()
    diff_matrix = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
    q_matrix = q.unsqueeze(0) * q.unsqueeze(1)
    gini = torch.sum(diff_matrix * q_matrix) / (2 * mu)
    return gini.item()

def weighted_skewness(values, weights, eps=1e-8):
    """
    Compute the weighted skewness for the eigenvalue distribution.
    Using normalized weights q_i = weights_i / (∑_j weights_j), we have:
      μ   = ∑_i q_i λ_i,
      σ²  = ∑_i q_i (λ_i - μ)²,
      skew = ∑_i q_i (λ_i - μ)³ / (σ³ + eps).
    """
    q = weights / weights.sum()
    mu = (values * q).sum()
    diff = values - mu
    variance = (q * diff**2).sum()
    std = torch.sqrt(variance + eps)
    skew = (q * diff**3).sum() / (std**3 + eps)
    return skew.item()

def compute_sigma_from_weights(state_dict, factor=1.0):
    """
    Compute sigma as a factor times the average standard deviation of the floating-point parameters.
    """
    sigmas = []
    for key, param in state_dict.items():
        if param.requires_grad:
            sigmas.append(param.std().item())
    if sigmas:
        return factor * (sum(sigmas) / len(sigmas))
    else:
        return factor

def compute_kl_divergence_initial_state(final_state_dict, init_state_dict):
    """
    Compute KL(Q||P) where
        Q = N(w_T, sigma^2 I) is the posterior (final weights),
        P = N(w_0, sigma0^2 I) is the prior (initial weights).
    """
    sigma = compute_sigma_from_weights(final_state_dict, factor=0.5)
    sigma0 = compute_sigma_from_weights(init_state_dict, factor=1.0)

    sigma2 = sigma ** 2
    sigma0_2 = sigma0 ** 2
    kl_total = 0.0

    for key in final_state_dict:
        param_final = final_state_dict[key]
        param_init = init_state_dict[key].to(param_final.device)

        if not torch.is_floating_point(param_final) or not torch.is_floating_point(param_init):
            continue

        d = param_final.numel()
        diff_norm_sq = torch.sum((param_final - param_init) ** 2)

        kl_tensor = 0.5 * (d * math.log(sigma0_2 / sigma2) +
                           diff_norm_sq / sigma0_2 +
                           d * (sigma2 / sigma0_2) - d)
        kl_total += kl_tensor

    return kl_total

def pac_bayes_term(kl_div, n, delta):
    """
    Computes the PAC-Bayes bound of the form:
        E[L(f)] <= E[hat{L}(f)] + sqrt((KL(Q||P) + log(1/delta_prime)) / (2n))

    Args:
        empirical_loss (float or torch.Tensor): Empirical loss (averaged over n samples).
        kl_div (float or torch.Tensor): KL(Q||P) already computed.
        n (int): Number of samples in the dataset.
        delta_prime (float): Confidence parameter (e.g., 0.05).

    Returns:
        torch.Tensor: The PAC-Bayes upper bound on the true (expected) loss.
    """
    if not isinstance(kl_div, torch.Tensor):
        kl_div = torch.tensor(float(kl_div), dtype=torch.float32)

    n_t = torch.tensor(float(n), dtype=torch.float32)
    delta_t = torch.tensor(float(delta), dtype=torch.float32)

    complexity_term = torch.sqrt((kl_div + torch.log(1.0 / delta_t)) / (2.0 * n_t))

    return complexity_term

def plot_curves(log, data_names, path_name, file_name=None, yabel='Hessian', save_dir="./results/", x_log=True, y_log=True):
    if file_name is None:
        file_name = path_name
    try:
        train_converge = log["train_converge"]["value"]
        val_converge = log["val_converge"]["value"]
    except:
        train_converge = 0
        val_converge = 0
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    for i, name in enumerate(data_names):
        plt.plot(log[name]["iter"], log[name]["value"], label=name)

    if train_converge > 0:
        plt.axvline(x=train_converge, color='blue', linestyle='--', linewidth=1, label='train convergence')
    if val_converge > 0:
        plt.axvline(x=val_converge, color='orange', linestyle='--', linewidth=1, label='val convergence')
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Hessian")
    if x_log:
        plt.xscale("log", base=10)
    if y_log:
        plt.yscale("log", base=10)
    plt.grid()
    plt.annotate(time_str, xy=(0.2, 0.5), xycoords='axes fraction', fontsize=12, color='purple', ha='center')
    plt.savefig(f"{save_dir}{path_name}/{file_name}_{log.label}.png", dpi=150)
    plt.draw()
    plt.close()

def sqrt_sum_nonnegative(arr):
    arr = np.array(arr)
    arr = np.where(arr < 0, 0, arr)
    return np.sum(np.sqrt(arr))

def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if (type(module) == torch.nn.Linear) and \
        ("LayerNorm" not in name and "embeddings" not in name and "pooler" not in name):
            layers[name] = module
    return layers