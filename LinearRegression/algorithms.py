import torch
import json
import random as r
import utils
from distributions import *
from torch.autograd import grad


EPS = 1e-16


class Algorithm(torch.nn.Module):
    def __init__(self, in_features, out_features, task, hparams="default"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.task = task

        # network architecture
        self.network = torch.nn.Linear(in_features, out_features)

        # loss
        if self.task == "regression":
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.BCEWithLogitsLoss()

        # hyper-parameters
        if hparams == "default":
            self.hparams = {k: v[0] for k, v in self.HPARAMS.items()}
        elif hparams == "random":
            self.hparams = {k: v[1] for k, v in self.HPARAMS.items()}
        else:
            self.hparams = json.loads(hparams)

        # callbacks
        self.callbacks = {}
        for key in ["errors"]:
            self.callbacks[key] = {
                "train": [],
                "validation": [],
                "test": []
            }


class ERM(Algorithm):
    def __init__(self, in_features, out_features, task, hparams="default"):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-2, 10**r.uniform(-4, -2))
        self.HPARAMS['wd'] = (0, 10**r.uniform(-6, -2))

        super().__init__(in_features, out_features, task, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def fit(self, envs, n_iterations, callback=False):
        x = torch.cat([xe for xe, ye in envs["train"]["envs"]])
        y = torch.cat([ye for xe, ye in envs["train"]["envs"]])

        # x_val = torch.cat([xe for xe, ye in envs["validation"]["envs"]])
        # y_val = torch.cat([ye for xe, ye in envs["validation"]["envs"]])

        for epoch in range(n_iterations):
            self.optimizer.zero_grad()
            loss = self.loss(self.network(x), y)
            loss.backward()
            self.optimizer.step()

            # if epoch % 100 == 0:
            #     self.network.eval()
            #     print(self.loss(self.network(x_val), y_val).item())
            #     self.network.train()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


class IRM(Algorithm):
    """
    Abstract class for IRM
    """

    def __init__(
            self, in_features, out_features, task, hparams="default", version=1):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**r.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**r.uniform(-6, -2))
        self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**r.uniform(-3, -.3))

        super().__init__(in_features, out_features, task, hparams)
        self.version = version

        self.network = self.IRMLayer(self.network)
        self.net_parameters, self.net_dummies = self.find_parameters(
            self.network)

        self.optimizer = torch.optim.Adam(
            self.net_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def find_parameters(self, network):
        """
        Alternative to network.parameters() to separate real parameters
        from dummmies.
        """
        parameters = []
        dummies = []

        for name, param in network.named_parameters():
            if "dummy" in name:
                dummies.append(param)
            else:
                parameters.append(param)
        return parameters, dummies

    class IRMLayer(torch.nn.Module):
        """
        Add a "multiply by one and sum zero" dummy operation to
        any layer. Then you can take gradients with respect these
        dummies. Often applied to Linear and Conv2d layers.
        """

        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.dummy_mul = torch.nn.Parameter(torch.Tensor([1.0]))
            self.dummy_sum = torch.nn.Parameter(torch.Tensor([0.0]))

        def forward(self, x):
            return self.layer(x) * self.dummy_mul + self.dummy_sum

    def fit(self, envs, n_iterations, callback=False):
        for epoch in range(n_iterations):
            losses_env = []
            gradients_env = []
            for x, y in envs["train"]["envs"]:
                losses_env.append(self.loss(self.network(x), y))
                gradients_env.append(grad(
                    losses_env[-1], self.net_dummies, create_graph=True))

            # Average loss across envs
            losses_avg = sum(losses_env) / len(losses_env)
            gradients_avg = grad(
                losses_avg, self.net_dummies, create_graph=True)

            penalty = 0
            for gradients_this_env in gradients_env:
                for g_env, g_avg in zip(gradients_this_env, gradients_avg):
                    if self.version == 1:
                        penalty += g_env.pow(2).sum()
                    else:
                        raise NotImplementedError

            obj = (1 - self.hparams["irm_lambda"]) * losses_avg
            obj += self.hparams["irm_lambda"] * penalty

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


class IRMv1(IRM):
    """
    IRMv1 with penalty \sum_e \| \nabla_{w|w=1} \mR_e (\Phi \circ \vec{w}) \|_2^2
    From https://arxiv.org/abs/1907.02893v1 
    """

    def __init__(self, in_features, out_features, task, hparams="default"):
        super().__init__(in_features, out_features, task, hparams, version=1)


class VREX(Algorithm):
    """
    Directly penalize error variance across environments.
    https://arxiv.org/abs/2003.00688
    """
    def __init__(self, in_features, out_features, task, hparams="default"):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**r.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**r.uniform(-6, -2))
        self.HPARAMS['vrex_lambda'] = (0.99, 1 - 10 ** r.uniform(-3, -.3))

        super().__init__(in_features, out_features, task, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def fit(self, envs, n_iterations, callback=False):
        for epoch in range(n_iterations):
            losses = [self.loss(self.network(x), y) for x, y in envs["train"]["envs"]]
            losses_avg = sum(losses) / len(losses)                                   # mean
            penalty = sum([(l - losses_avg) ** 2 for l in losses]) / len(losses)     # var

            obj = (1 - self.hparams["vrex_lambda"]) * losses_avg
            obj += self.hparams["vrex_lambda"] * penalty

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


class EQRM(Algorithm):
    def __init__(self, in_features, out_features, task, hparams="default", dist=Nonparametric()):
        self.dist = dist
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-2, 10**r.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**r.uniform(-6, -2))
        self.HPARAMS['alpha'] = (-10**5, r.choices((r.uniform(0.5, 1.), -100**r.uniform(0.5, 3)), weights=(0.75, 0.25)))
        super().__init__(in_features, out_features, task, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def fit(self, envs, n_iterations, callback=False):
        q = torch.tensor(self.hparams["alpha"], dtype=torch.float64)

        for epoch in range(n_iterations):
            losses = [self.loss(self.network(x), y) for x, y in envs["train"]["envs"]]      # list of 10 size-0 tensors
            losses = torch.concat([l.reshape(1) for l in losses])                           # tensor of size 10

            self.dist.estimate_parameters(losses)
            obj = self.dist.icdf(q)

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


ALGORITHMS = {
    "ERM": ERM,
    "EQRM": EQRM,
    "Oracle": ERM,
    # "IRMv1": IRMv1,
    # "VREX": VREX,
}
