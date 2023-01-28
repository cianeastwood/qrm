import numpy as np
import random
import torch
import itertools
import pandas as pd

pd.set_option('colheader_justify', 'center')
ERRORS = ["MSE", "FVU", "R2"]     # r^2 = 1 - FVU, where FVU is the fraction of variance unexplained
ERROR = ERRORS[0]
USE_FVU = ERROR in ["MSE", "FVU"]
EPS = 1e-6


def analytical_results(x1_var, y_var, x2_var):
    """
    TODO: Add r_sq_x1x2
    """
    x1_res = x1_var / (x1_var + y_var)
    x2_res = (x1_var + y_var) / (x1_var + y_var + x2_var)

    if ERROR == "MSE":
        x1_res, x2_res = (1. - x1_res) * y_var, (1. - x2_res) * y_var
    elif ERROR == "FVU":
        x1_res, x2_res = 1. - x1_res, 1. - x2_res

    print(f"Analytical OLS results\nX1:{ERROR}={x1_res:.2f};\tX2:{ERROR}={x2_res:.2f}")


def solve_linear_system(x, y, error="MSE"):
    x, y = np.array(x), np.array(y)
    coeff = np.linalg.inv(x.T @ x) @ x.T.dot(y)
    mse = np.mean((y - x.dot(coeff)) ** 2)
    if error == "MSE":
        return coeff, mse
    elif error == "FVU":
        return coeff, mse / y.var()
    else:  # r^2
        return coeff, 1. - mse / y.var()


def empirical_results(env_xs, env_ys, per_env_results=True, print_coeffs=True):
    """
    TODO: fix bug -- results seem slightly off compared to analytical and SGD.
    """
    env_x1s = [x[:, 0:1].numpy() for x in env_xs]
    env_x2s = [x[:, 1:].numpy() for x in env_xs]

    if per_env_results:
        x1_e_res = [solve_linear_system(x, y, ERROR) for x, y in zip(env_x1s, env_ys)]
        x2_e_res = [solve_linear_system(x, y, ERROR) for x, y in zip(env_x2s, env_ys)]
        x1x2_e_res = [solve_linear_system(x, y, ERROR) for x, y in zip(env_xs, env_ys)]

        print("Empirical OLS results (per env):")
        for i, ((x1_c, x1_r2), (x2_c, x2_r2), (x1x2_c, x1x2_r2)) in enumerate(zip(x1_e_res, x2_e_res, x1x2_e_res)):
            print(f"Env {i}")
            print(f"X1: {ERROR}={x1_r2:.2f}" + f"; coeff={x1_c}" * print_coeffs)
            print(f"X2: {ERROR}={x2_r2:.2f}" + f"; coeff={x2_c}" * print_coeffs)
            print(f"[X1,X2]: {ERROR}={x1x2_r2:.2f}" + f"; coeffs={x1x2_c}" * print_coeffs)

    # Pooled envs
    pooled_x1s, pooled_ys, pooled_x2s = np.vstack(env_x1s), np.vstack(env_ys), np.vstack(env_x2s)
    x1_pooled_res = solve_linear_system(pooled_x1s, pooled_ys, ERROR)
    x2_pooled_res = solve_linear_system(pooled_x2s, pooled_ys, ERROR)
    x1x2_pooled_res = solve_linear_system(np.hstack([pooled_x1s, pooled_x2s]), pooled_ys, ERROR)

    print("Empirical OLS results (pooled envs):")
    print(f"X1: {ERROR}={x1_pooled_res[1]:.2f}" + f"; coeff={x1_pooled_res[0]}" * print_coeffs)
    print(f"X2: {ERROR}={x2_pooled_res[1]:.2f}" + f"; coeff={x2_pooled_res[0]}" * print_coeffs)
    print(f"[X1,X2]: {ERROR}={x1x2_pooled_res[1]:.2f}" + f"; coeffs={x1x2_pooled_res[0]}" * print_coeffs)


def set_seed(seed=404):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_error(algorithm, x, y, error=ERROR):
    with torch.no_grad():
        if len(y.unique()) == 2:    # hack to detect classification
            return algorithm.predict(x).gt(0).ne(y).float().mean().item()
        else:                       # regression
            mse = (algorithm.predict(x) - y).pow(2).mean().item()
            if error == "MSE":
                return mse
            var_y = torch.var(y).item()
            fvu = mse / var_y
            if error == "FVU": # lower is better (like MSE), but still normalised by var(y)
                return fvu
            elif error == "R2":
                return 1. - fvu
            else:
                raise ValueError(f"Invalid error {error}. Choose one of: {ERRORS}.")


def compute_errors(model, envs):
    for split in envs.keys():
        if not bool(model.callbacks["errors"][split]):
            model.callbacks["errors"][split] = {
                key: [] for key in envs[split]["keys"]}

        for k, env in zip(envs[split]["keys"], envs[split]["envs"]):
            model.callbacks["errors"][split][k].append(
                compute_error(model, *env))


def get_alpha_qrm_name_latex(qrm_model_name, alpha_value_only=False, tight=False):
    """
    Convert EQRM model name to a latex-friendly version.
    :param qrm_model_name: EQRM model name (str), e.g. EQRM_a=0.75 or EQRM_a~=1.
    :param alpha_value_only: return only the alpha setting, not entire name.
    :param tight: squeeze space using latex tight spacing '\!' (bool).
    :return: desired latex name.
    """
    if "~" in qrm_model_name:
        latex_str = qrm_model_name.replace(r"_a~=", r"_\alpha\approx")
    else:
        latex_str = qrm_model_name.replace(r"_a=", r"_\alpha=")

    if tight:
        latex_str = latex_str.replace(r"\alpha", r"\alpha\!")
        latex_str = latex_str.replace(r"=", r"=\!")
        latex_str = latex_str.replace(r"\approx", r"\approx\!")

    name_str, alpha_str = latex_str.split("_")
    if alpha_value_only:
        # alpha_str = latex_str[latex_str.index(r"\alpha"):-2]   # _a=0.9
        return "$" + alpha_str + r"$"
    else:
        return f"$\\text{{{name_str}}}_{{{alpha_str}}}$"


def sort_models_alpha(model_name):
    """
    Sort models by model name (e.g. EQRM, ERM, Oracle) and then by alpha value.
    :param model_name: string model name, e.g. "EQRM_a=0.75".
    :return: tuple of keys to sort by.
    """
    if "=" in model_name:                                           # "EQRM_a=0.75" / "EQRM_a~=1"
        m_name, alpha_str = model_name.split("_")                   # "EQRM", "a=0.75" / "a~=1"
        alpha_value = float(alpha_str[alpha_str.index('=') + 1:])
    else:
        m_name = model_name
        alpha_value = 0

    return m_name, alpha_value


def sort_models_alpha_latex(model_name):
    if '=' in model_name:
        start_idx = model_name.index('=')
        end_idx = model_name.index('}$')
        m_name = model_name[:start_idx]
        alpha_value = float(model_name[start_idx + 1:end_idx])
    elif '\\approx' in model_name:
        m_name = model_name[:model_name.index('\\approx')]
        alpha_value = 1
    else:
        m_name = model_name
        alpha_value = 0

    return m_name, alpha_value


def sort_models_alpha_only_latex(model_name):
    if '=' in model_name:
        start_idx = model_name.index('=')
        # print(model_name)
        m_name = model_name[:start_idx]
        alpha_value = float(model_name[start_idx + 1:-1])
    elif '\\approx' in model_name:
        m_name = model_name[:model_name.index('\\approx')]
        alpha_value = 1
    else:
        m_name = model_name
        alpha_value = 0

    return m_name, alpha_value


def continuous_bisect_fun_left(f, v, lo, hi, n_steps=32):
    val_range = [lo, hi]
    k = 0.5 * sum(val_range)
    for _ in range(n_steps):
        val_range[int(f(k) > v)] = k
        next_k = 0.5 * sum(val_range)
        if next_k == k:
            break
        k = next_k
    return k


############################################################
# PyTorch Implementation of KDE. Builds on:
# https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py
############################################################

class Kernel(torch.nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bw=None):
        super().__init__()
        self.bw = 0.05 if bw is None else bw

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    def forward(self, test_Xs, train_Xs):
        """Computes p(x) for each x in test_Xs given train_Xs."""

    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        diffs = self._diffs(test_Xs, train_Xs)
        dims = tuple(range(len(diffs.shape))[2:])
        if dims == ():
            x_sq = diffs ** 2
        else:
            x_sq = torch.norm(diffs, p=2, dim=dims) ** 2

        var = self.bw ** 2
        exp = torch.exp(-x_sq / (2 * var))
        coef = 1. / torch.sqrt(2 * np.pi * var)

        return (coef * exp).mean(dim=1)

    def sample(self, train_Xs):
        # device = train_Xs.device
        noise = torch.randn(train_Xs.shape) * self.bw
        return train_Xs + noise

    def cdf(self, test_Xs, train_Xs):
        mus = train_Xs                                                      # kernel centred on each observation
        sigmas = torch.ones(len(mus), device=test_Xs.device) * self.bw      # bandwidth = stddev
        x_ = test_Xs.repeat(len(mus), 1).T                                  # repeat to allow broadcasting below
        return torch.mean(torch.distributions.Normal(mus, sigmas).cdf(x_))


def estimate_bandwidth(x, method="silverman"):
    x_, _ = torch.sort(x)
    n = len(x_)
    sample_std = torch.std(x_, unbiased=True)

    if method == 'silverman':
        # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        iqr = torch.quantile(x_, 0.75) - torch.quantile(x_, 0.25)
        bandwidth = 0.9 * torch.min(sample_std, iqr / 1.34) * n ** (-0.2)

    elif method.lower() == 'gauss-optimal':
        bandwidth = 1.06 * sample_std * (n ** -0.2)

    else:
        raise ValueError(f"Invalid method selected: {method}.")

    return bandwidth


class KernelDensityEstimator(torch.nn.Module):
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel='gaussian', bw_select='Gauss-optimal'):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.train_Xs = train_Xs
        self._n_kernels = len(self.train_Xs)

        if bw_select is not None:
            self.bw = estimate_bandwidth(self.train_Xs, bw_select)
        else:
            self.bw = None

        if kernel.lower() == 'gaussian':
            self.kernel = GaussianKernel(self.bw)
        else:
            raise NotImplementedError(f"'{kernel}' kernel not implemented.")

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(self._n_kernels), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])

    def cdf(self, x):
        return self.kernel.cdf(x, self.train_Xs)

