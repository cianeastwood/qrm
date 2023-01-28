"""
PyTorch implementation of 1D distributions.
"""

import math
import torch
from torch.distributions import TransformedDistribution, Uniform, SigmoidTransform, AffineTransform
from scipy.optimize import fsolve
from scipy.special import gamma
from utils import *

EPS = 1e-16


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


class LogNormal(Distribution1D):
    def __init__(self, location=0, scale=1, shape=None, mean=None, stddev=None):
        self.location = location
        self.scale = scale
        self.shape = shape
        self.standard_normal_icdf = None

        if mean is not None and stddev is not None:
            # alternative parameterization via the mean and standard deviation
            self.location, self.scale = self.parameters_from_moments(mean, stddev)

            # # Test calculations
            # mean_, stddev_ = self.moments_from_parameters()
            # print(f"Mean: {mean:.3f}={mean_:.3f}?")
            # print(f"Standard dev: {stddev:.3f}={stddev_:.3f}?")
            # print(f"Location={self.location:.3f}; Scale={self.scale:.3f}.")

        super().__init__(torch.distributions.LogNormal)

    @property
    def parameters(self):
        return [self.location, self.scale]

    def parameters_from_moments(self, mean, stddev):
        loc = math.log(mean**2 / math.sqrt(stddev**2 + mean**2))
        scale = math.sqrt(math.log(stddev**2/mean**2 + 1))
        return loc, scale

    def moments_from_parameters(self):
        mean = math.exp(self.location + 0.5 * self.scale**2)
        var = (math.exp(self.scale**2) - 1) * math.exp(2 * self.location + self.scale**2)
        stddev = math.sqrt(var + EPS)
        return mean, stddev

    def estimate_parameters(self, x):
        log_x = [torch.log(x_i + EPS) for x_i in x]
        mean_log = sum(log_x) / len(log_x)
        var_log = sum([(log_x_i - mean_log) ** 2 for log_x_i in log_x]) / (len(log_x) - 1)

        self.location = sum(x) / len(x)
        self.scale = torch.exp(mean_log)
        self.shape = torch.sqrt(var_log + EPS)

    def icdf(self, q):
        """
        To get q *very* close to 1, we will pass log(1-q) and use the approximation here:
        https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
        """
        if self.shape is None:          # hack for now when shape unknown (estimate_parameters() is never called)
            return super().icdf(q)

        if q < 0:  # q actually represents log(y), where q = 1 - y is the quantile (avoids numerical issues)
            log_y = q
            value_standard_normal = torch.sqrt(-2 * log_y)
        else:
            if self.standard_normal_icdf is None:
                self.standard_normal_icdf = torch.distributions.Normal(0, 1).icdf
            value_standard_normal = self.standard_normal_icdf(q)

        value = self.location + self.scale * torch.exp(self.shape * value_standard_normal)

        if torch.isnan(value):
            raise ValueError("Quantile q is too close to 1, causing nans in the inverse CDF.")

        return value


class HalfNormal(Distribution1D):
    def __init__(self, scale=1, stddev=None):
        self.scale = scale
        if stddev is not None:
            self.scale = stddev
        super().__init__(torch.distributions.HalfNormal)

    @property
    def parameters(self):
        return [self.scale]

    def estimate_parameters(self, x):
        self.scale = torch.var(x, unbiased=True)

    def icdf(self, q):
        """
        To get q *very* close to 1, we will pass log(1-q) and use the approximation here:
        https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
        """
        if q < 0:
            # q actually represents log(y), where q = 1 - y is the quantile (avoids numerical issues)
            log_y = q
            return self.scale * math.sqrt(-2 * log_y)
        else:
            # q represents the quantile
            return super().icdf(q)


class Logistic(Distribution1D):
    def __init__(self, location=0, scale=1, mean=None, stddev=None):
        self.location = location
        self.scale = scale

        if stddev is not None:
            self.location = mean
            self.scale = stddev * math.sqrt(3) / math.pi

        def Logistic():
            return TransformedDistribution(
                Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(self.location, self.scale)]
            )

        super().__init__(Logistic)

    @property
    def parameters(self):
        return [self.location, self.scale]

    def estimate_parameters(self, x):
        self.location = torch.mean(x)
        self.scale = torch.std(x, unbiased=True) * math.sqrt(3) / math.pi

    def icdf(self, q):
        if q < 0:
            # q actually represents -log(q / (1 - q)), where q is the quantile (avoids numerical issues)
            log_q_over_1_minus_q = -q
        else:
            # q represents the quantile
            log_q_over_1_minus_q = torch.log(q / (1. - q))

        return self.location + self.scale * log_q_over_1_minus_q


class Gumbel(Distribution1D):
    """
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda366g.htm
    """
    def __init__(self, location=0, scale=1, mean=None, stddev=None):
        self.location = location
        self.scale = scale

        if mean is not None and stddev is not None:
            # alternative parameterization via the mean and standard deviation
            self.parameters_from_moments(mean, stddev)
            # print(self.location, self.scale)

        super().__init__(torch.distributions.Gumbel)

    @property
    def parameters(self):
        return [self.location, self.scale]

    def parameters_from_moments(self, mean, stddev):
        self.location = mean + 0.54006 * stddev
        self.scale = stddev * math.sqrt(6) / math.pi

    def estimate_parameters(self, x):
        mean = sum(x) / len(x)
        var = sum([(x_i - mean) ** 2 for x_i in x]) / (len(x) - 1)
        std = torch.sqrt(var + EPS)

        self.location = mean + 0.54006 * std
        self.scale = std * math.sqrt(6) / math.pi

    def icdf(self, q):
        if q < 0:  # q actually represents log(log(1/q)), where q is the quantile (avoids numerical issues)
            neg_log_log_1_q = -q
        else:
            neg_log_log_1_q = -torch.log(torch.log(1./q) + EPS)

        return self.location + self.scale * neg_log_log_1_q


class Exponential(Distribution1D):
    def __init__(self, rate=1, mean=None):
        self.rate = rate
        if mean is not None:
            # alternative parameterization via the mean
            self.rate = 1. / (mean + EPS)
        super().__init__(torch.distributions.Exponential)

    @property
    def parameters(self):
        return [self.rate]

    def estimate_parameters(self, x):
        mean = sum(x) / len(x)
        self.rate = 1. / (mean + EPS)


class Weibull(Distribution1D):
    """
    2-parameter Weibull distribution (location=0).
    """
    def __init__(self, scale=1, shape=1, mean=None, stddev=None):
        self.scale = scale
        self.shape = shape
        
        if mean is not None and stddev is not None:
            # alternative parameterization via the mean and standard deviation
            self.parameters_from_moments(mean, stddev)

            # # Test calculations
            # mean_, stddev_ = self.moments_from_parameters()
            # print(f"Mean: {mean:.3f}={mean_:.3f}?")
            # print(f"Standard dev: {stddev:.3f}={stddev_:.3f}?")
            # print(f"Scale={self.scale:.3f}; Shape={self.shape:.3f}.")

        super().__init__(torch.distributions.Weibull)



    @property
    def parameters(self):
        return [self.scale, self.shape]

    def parameters_from_moments(self, mean, stddev):
        """
        See https://math.stackexchange.com/questions/1769765/weibull-distribution-from-mean-and-variance-to-shape-and-scale-factor
        """
        def f(k):
            return stddev**2 / mean**2 - (gamma(1 + 2./k) / (gamma(1 + 1./k))**2) + 1
        k0 = 1.
        self.shape = fsolve(f, k0)[0]
        self.scale = mean / gamma(1 + 1./self.shape)

    def moments_from_parameters(self):
        lambda_ = self.scale
        k = self.shape

        mean = lambda_ * gamma(1 + 1./k)
        var = lambda_**2 * (gamma(1 + 2./k) - (gamma(1 + 1./k))**2)
        stddev = math.sqrt(var + EPS)

        return mean, stddev
    
    def estimate_parameters(self, x, iters=10, eps=1e-6):
        """
        MLE using Newton-Raphson optimization. Adapted from https://github.com/mlosch/python-weibullfit.
        """
        k = 1.0
        k_t_1 = k
        ln_x = torch.log(x)

        for t in range(iters):
            # Partial derivative df/dk
            x_k = x ** k
            x_k_ln_x = x_k * ln_x
            ff = torch.sum(x_k_ln_x)
            fg = torch.sum(x_k)
            f1 = torch.mean(ln_x)
            f = ff / fg - f1 - (1.0 / k)

            ff_prime = torch.sum(x_k_ln_x * ln_x)
            fg_prime = ff
            f_prime = (ff_prime / fg - (ff / fg * fg_prime / fg)) + (1. / (k * k))

            # Newton-Raphson method k = k - f(k;x)/f'(k;x)
            k = k - f / f_prime

            if torch.isnan(f):
                print("NAN")
                return torch.nan, torch.nan

            if abs(k - k_t_1) < eps:
                break

            k_t_1 = k

        self.scale = torch.mean(x ** k) ** (1.0 / k)
        self.shape = k

    def icdf(self, q):
        """
        See https://www.itl.nist.gov/div898/handbook/eda/section3/eda3668.htm
        """
        if q < 0:  # q actually represents log(log(1/q.)), where q is the quantile (avoids numerical issues)
            neg_log_1_minus_q = -q
        else:
            neg_log_1_minus_q = -math.log(1. - q)

        return self.scale * neg_log_1_minus_q**(1. / self.shape)


class Pareto(Distribution1D):
    def __init__(self, scale=1, shape=1):
        self.scale = scale
        self.shape = shape
        super().__init__(torch.distributions.Pareto)

    @property
    def parameters(self):
        return [self.scale, self.shape]

    def estimate_parameters(self, x):
        self.scale = min(x)
        self.shape = len(x) / sum(torch.log((x_i / self.scale) + EPS) for x_i in x)

    def icdf(self, q):
        """See https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/parppf.htm"""
        return self.scale * (1. - q) ** (-1. / self.shape)


class Cauchy(Distribution1D):
    def __init__(self, location=0, scale=1):
        self.location = location
        self.scale = scale
        super().__init__(torch.distributions.Cauchy)

    @property
    def parameters(self):
        return [self.location, self.scale]

    def estimate_parameters(self, x):
        """
        https://en.wikipedia.org/wiki/Cauchy_distribution#Estimation_of_parameters
        """
        # Extract quantile indices
        n = len(x)
        n_12 = 12 * n // 10
        n_25 = n // 4
        n_50 = n // 2
        n_75 = n_25 * 3

        # Sort data
        sorted_x, _ = torch.sort(x)

        # Robust estimation of the location (i.e. median) via the (truncated) mean of middle 24%
        middle_24_percent = sorted_x[n_50 - n_12:n_50 + n_12]
        self.location = torch.mean(middle_24_percent)

        # Estimation of the scale: half the IQR
        iqr = sorted_x[n_75] - sorted_x[n_25]
        self.scale = 0.5 * iqr


class GMM(Distribution1D):
    def __init__(self, weights=None, means=None, stddevs=None, skip_m_step=True):
        self.weights = weights
        self.means = means
        self.stddevs = stddevs
        self.skip_m_step = skip_m_step

        super().__init__(torch.distributions.MixtureSameFamily)

    @property
    def parameters(self):
        return [torch.distributions.Categorical(self.weights), self.component_distribution]

    @property
    def component_distribution(self):
        return torch.distributions.Normal(self.means, self.stddevs)

    def estimate_parameters(self, x):
        # M step: estimate means and stddevs
        if not self.skip_m_step:
            if self.weights is None:
                raise NotImplementedError
            raise NotImplementedError

        # E step: estimate weights / mixing coefficients
        posteriors = torch.exp(self.component_distribution.log_prob(x))
        self.weights = posteriors / posteriors.sum()

    def icdf(self, q):
        """
        To get q *very* close to 1, we will pass log(1-q) and use the approximation here:
        https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
        """
        if q < 0:
            # looking for extreme values in right tail --> only need to consider the Gaussian with largest variance
            # q actually represents log(y), where q = 1 - y is the quantile (avoids numerical issues)
            log_y = q
            idx = torch.argmax(self.stddevs)
            return self.means[idx] + self.stddevs[idx] * math.sqrt(-2 * log_y)
        else:
            # q represents the quantile which we must calculate for the mixture distribution
            # find limits (can probably be a bit smarter here...)
            component_dist_quantiles = self.component_distribution.icdf(q)          # (K)
            lo = torch.min(component_dist_quantiles)
            hi = torch.max(component_dist_quantiles)

            return continuous_bisect_fun_left(self.cdf, q, lo, hi)




