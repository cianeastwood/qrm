import torch
import torch.nn.functional as F
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to
from wilds.common.utils import split_into_groups

EPS = 1e-6

class QRMBase(SingleModelAlgorithm):

    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, dist):
        model = initialize_model(config, d_out)
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps)
        
        self.q = config.var_quantile
        self.dist = dist

    def objective(self, results):

        def risk(group_ids):
            return self.loss.compute(
                results['y_pred'][group_ids], 
                results['y_true'][group_ids], 
                return_dict=False).reshape(1)

        _, group_indices, _ = split_into_groups(results['g'])
        env_risks = torch.cat([risk(ids) for ids in group_indices])

        self.dist.estimate_parameters(env_risks)
        obj = self.dist.icdf(self.q)

        return obj


class QRM_Normal(QRMBase):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        super(QRM_Normal, self).__init__(config, d_out, grouper, loss, metric, n_train_steps, dist=Normal())


class QRM_KDE(QRMBase):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        super(QRM_KDE, self).__init__(config, d_out, grouper, loss, metric, n_train_steps, dist=Nonparametric())


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
    sample_std = torch.std(x_, unbiased=True) + EPS

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


############################################################
# PyTorch implementation of 1D distributions.
############################################################


class Distribution1D:
    def __init__(self, dist_function=None):
        """
        :param dist_function: function to instantiate the distribution (self.dist).
        :param parameters: list of parameters in the correct order for dist_function.
        """
        self.dist = None
        self.dist_function = dist_function

    @property
    def parameters(self):
        raise NotImplementedError

    def create_dist(self):
        if self.dist_function is not None:
            return self.dist_function(*self.parameters)
        else:
            raise NotImplementedError("No distribution function was specified during intialization.")

    def estimate_parameters(self, x):
        raise NotImplementedError

    def log_prob(self, x):
        return self.create_dist().log_prob(x)

    def cdf(self, x):
        return self.create_dist().cdf(x)

    def icdf(self, q):
        return self.create_dist().icdf(q)

    def sample(self, n=1):
        if self.dist is None:
            self.dist = self.create_dist()
        n_ = torch.Size([]) if n == 1 else (n,)
        return self.dist.sample(n_)

    def sample_n(self, n=10):
        return self.sample(n)


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


class Nonparametric(Distribution1D):
    def __init__(self, use_kde=True, bw_select='Gauss-optimal'):
        self.use_kde = use_kde
        self.bw_select = bw_select
        self.bw, self.data, self.kde = None, None, None
        super().__init__()

    @property
    def parameters(self):
        return []

    def estimate_parameters(self, x):
        self.data, _ = torch.sort(x)

        if self.use_kde:
            self.kde = KernelDensityEstimator(self.data, bw_select=self.bw_select)
            self.bw = torch.ones(1, device=self.data.device) * self.kde.bw

    def icdf(self, q):
        if not self.use_kde:
            # Empirical or step CDF. Differentiable as torch.quantile uses (linear) interpolation.
            return torch.quantile(self.data, float(q))

        if q >= 0:
            # Find quantile via binary search on the KDE CDF
            lo = torch.distributions.Normal(self.data[0], self.bw[0]).icdf(q)
            hi = torch.distributions.Normal(self.data[-1], self.bw[-1]).icdf(q)
            return continuous_bisect_fun_left(self.kde.cdf, q, lo, hi)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            v = torch.mean(self.data + self.bw * math.sqrt(-2 * log_y))
            return v


class Normal(Distribution1D):
    def __init__(self, location=0, scale=1):
        self.location = location
        self.scale = scale
        super().__init__(torch.distributions.Normal)

    @property
    def parameters(self):
        return [self.location, self.scale]

    def estimate_parameters(self, x):
        mean = sum(x) / len(x)
        var = sum([(x_i - mean) ** 2 for x_i in x]) / (len(x) - 1)
        self.location = mean
        self.scale = torch.sqrt(var + EPS)

    def icdf(self, q):
        if q >= 0:
            return super().icdf(q)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            return self.location + self.scale * math.sqrt(-2 * log_y)
