import torch
import math
import utils
from distributions import *


############################################################################################################
# Base dataset from https://github.com/facebookresearch/InvarianceUnitTests/blob/main/scripts/datasets.py.
############################################################################################################

class Example1:
    """
    Cause and effect of a target with heteroskedastic noise.
    """
    def __init__(self, dim_inv, dim_spu, n_envs, verbose):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "regression"
        self.envs = {}

        if n_envs >= 2:
            self.envs = {'E0': 0.1, 'E1': 1.5}
        if n_envs >= 3:
            self.envs["E2"] = 2
        if n_envs > 3:
            for env in range(3, n_envs):
                var = 10 ** torch.zeros(1).uniform_(-2, 1).item()
                self.envs["E" + str(env)] = var

        self.wxy = torch.randn(self.dim_inv, self.dim_inv) / self.dim_inv
        self.wyz = torch.randn(self.dim_inv, self.dim_spu) / self.dim_spu

    def sample(self, n=1000, env="E0", split="train"):
        sdv = self.envs[env]
        x = torch.randn(n, self.dim_inv) * sdv
        y = x @ self.wxy + torch.randn(n, self.dim_inv) * sdv
        z = y @ self.wyz + torch.randn(n, self.dim_spu)

        if split == "test":
            # test "environment" corresponds to shuffling the effect features
            z = z[torch.randperm(len(z))]

        inputs = torch.cat((x, z), -1) @ self.scramble
        outputs = y.sum(1, keepdim=True)

        return inputs, outputs


class Example1s(Example1):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.linalg.qr(torch.randn(self.dim, self.dim))


############################################################
# Dataset from our paper:
############################################################

class Example2:
    """
    Cause and effect of a target with homoskedastic noise and **a distribution over shifts/environments**.

    Similar to Example 1, but has:
        1) homoskedastic noise (simplest case where only the effect-feature noise distribution changes).
        2) a distribution over shifts/environments, i.e. the stddev of the effect-feature's noise. This means that
        for test environments, a new environment is sampled from the distribution over environments, rather than just
        shuffling the effect features. This corresponds to an analysis of probable or probabilistic domain
        generalization, rather than only worst-case.
        3) 1D features (simplest case).
        4) analytical calculation of errors (to sanity check empirical scores and help choose env distributions).

    Criteria for choosing sdv_1, sdv_2, sdv_Y:
        - X_1 is stable
        - X_2 is unstable: sdv_2 varies across environments, i.e. var(sdv_2) is "large enough".
        - X_2 is more informative about Y than X_1, so ERM uses it, i.e. lower MSE when using X_2 than using X_1.

    TODO:
        1) Update naming of "test" split -- shuffled X_2 for the test split is now only used to train an Oracle model
        which ignores X_2. Note: requires significant changes to main.py and collect_results.py. Would also remove
        compatibility with Example1 above.
    """

    def __init__(self, dim_inv=1, dim_spu=1, n_envs=20, p_env=LogNormal(0, math.sqrt(0.5)), verbose=False):
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = self.dim_inv + self.dim_spu
        self.scramble = torch.eye(self.dim_inv + self.dim_spu)
        self.p_env = p_env

        self.task = "regression"
        self.envs = {}

        # Fixed stddev of x1 and y (controls the oracle risk/mse/r^2).
        self.x1_sdv = 1.
        self.y_sdv = math.sqrt(2.)

        for env in range(n_envs):
            sdv = self.p_env.sample()
            self.envs["E" + str(env)] = sdv

        # Print OLS analytical errors
        x2_var_pooled = sum([v**2 for k, v in self.envs.items()]) / n_envs
        utils.analytical_results(self.x1_sdv ** 2, self.y_sdv ** 2, x2_var_pooled)

        if verbose:
            # Print OLS empirical errors
            env_inputs, env_outputs = [], []
            for env in self.envs.keys():
                inputs, outputs = self.sample(env=env, n=10000)
                env_inputs.append(inputs)
                env_outputs.append(outputs)
            utils.empirical_results(env_inputs, env_outputs)

    def sample(self, n=1000, env="E0", split="train"):
        if "test" in env:
            # Sample a new test environment for evaluation
            if "q=" in env:
                # Specified quantile under p_env, i.e. P(S <= sdv) = q
                q = torch.tensor(float(env[env.index("=") + 1:]), dtype=torch.float64)
                sdv = self.p_env.icdf(q)
            else:
                # Random sample
                sdv = self.p_env.sample()
        else:
            # Use existing environment
            sdv = self.envs[env]

        x1 = torch.randn(n, self.dim_inv) * self.x1_sdv
        y = x1 + torch.randn(n, self.dim_inv) * self.y_sdv
        x2 = y + torch.randn(n, self.dim_spu) * sdv

        if "E" in env and split == "test":
            # Shuffled "test" split only used to train Oracle model (to be updated...)
            x2 = x2[torch.randperm(len(x2))]

        inputs = torch.cat((x1, x2), -1)
        outputs = y.sum(1, keepdim=True)

        if "q=" in env:
            return inputs, outputs, sdv
        else:
            return inputs, outputs


#####################################################################
# Noise stddev could be sampled from other distributions for Example2:
#####################################################################

class Example2HN(Example2):
    def __init__(self, dim_inv=1, dim_spu=1, n_envs=20, p_env=HalfNormal(stddev=math.sqrt(0.5)), verbose=False):
        super().__init__(dim_inv, dim_spu, n_envs, p_env, verbose)


class Example2GB(Example2):
    def __init__(self, dim_inv=1, dim_spu=1, n_envs=20, p_env=Gumbel(mean=0.1, stddev=0.85), verbose=False):
        super().__init__(dim_inv, dim_spu, n_envs, p_env, verbose)


class Example2WB(Example2):
    def __init__(self, dim_inv=1, dim_spu=1, n_envs=20, p_env=Weibull(mean=0.4, stddev=2.5), verbose=False):
        super().__init__(dim_inv, dim_spu, n_envs, p_env, verbose)


class Example2EXP(Example2):
    def __init__(self, dim_inv=1, dim_spu=1, n_envs=20, p_env=Exponential(mean=0.5), verbose=False):
        super().__init__(dim_inv, dim_spu, n_envs, p_env, verbose)


DATASETS = {
    "Example1": Example1,
    "Example1s": Example1s,
    "Example2": Example2,
}


