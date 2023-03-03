import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


MIN_LIKELIHOOD = 1e-9
MAX_LIKELIHOOD = 1e4

class LowerBoundToward(torch.autograd.Function):
    """
    Assumes output shape is identical to input shape.
    """
    @staticmethod
    def forward(ctx, tensor, lower_bound):
        # lower_bound:  Scalar float.
        ctx.mask = tensor.ge(lower_bound)
        return torch.clamp(tensor, lower_bound)

    @staticmethod
    def backward(ctx, grad_output):
        gate = torch.logical_or(
            ctx.mask, grad_output.lt(0.)).type(grad_output.dtype)
        return grad_output * gate, None


lower_bound_toward = LowerBoundToward.apply


class HyperpriorDensity(nn.Module):
    """
    Probability model for hyper-latents z. Based on Sec. 6.1. of [1].
    Returns convolution of non-parametric hyperlatent density with uniform
    distribution
    U(-1/2, 1/2).

    Assumes that the input tensor is at least 2D, with a batch dimension
    at the beginning and a channel dimension as specified by `data_format`. The
    layer trains an independent probability density model for each channel, but
    assumes that across all other dimensions, the inputs are i.i.d.

    [1] Ballé et. al., "Variational image compression with a scale hyperprior",
        arXiv:1802.01436 (2018).
    """

    def __init__(self, n_channels, init_scale=10., filters=(3, 3, 3),
                 min_s=None,
                 max_s=None,
                 min_likelihood=MIN_LIKELIHOOD,
                 max_likelihood=MAX_LIKELIHOOD, **kwargs):
        """
        init_scale: Scaling factor determining the initial width of the
                    probability densities.
        filters:    Number of filters at each layer < K
                    of the density model. Default K=4 layers.
        """
        super(HyperpriorDensity, self).__init__(**kwargs)

        self.min_symbol = min_s
        self.max_symbol = max_s
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in filters)
        self.min_likelihood = float(min_likelihood)
        self.max_likelihood = float(max_likelihood)
        self.n_channels = n_channels
        self.dtype = torch.float32

        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        # Define univariate density model
        for k in range(len(self.filters) + 1):  # 1408 params for 32 channels
            # Weights
            H_init = np.log(np.expm1(1 / scale / filters[k + 1]))
            H_k = nn.Parameter(torch.ones((n_channels, filters[k + 1], filters[
                k])))  # apply softmax for non-negativity
            torch.nn.init.constant_(H_k, H_init)
            self.register_parameter('H_{}'.format(k), H_k)

            # Scale factors
            a_k = nn.Parameter(torch.zeros((n_channels, filters[k + 1], 1)))
            self.register_parameter('a_{}'.format(k), a_k)

            # Biases
            b_k = nn.Parameter(torch.zeros((n_channels, filters[k + 1], 1)))
            torch.nn.init.uniform_(b_k, -0.5, 0.5)
            self.register_parameter('b_{}'.format(k), b_k)

    def cdf_logits(self, x, update_parameters=True):
        """
        Evaluate logits of the cumulative densities.
        Independent density model for each channel.

        x:  The values at which to evaluate the cumulative densities.
            torch.Tensor - shape `(C, 1, *)`.
        """
        logits = x

        for k in range(len(self.filters) + 1):
            H_k = getattr(self, 'H_{}'.format(str(k)))  # Weight
            a_k = getattr(self, 'a_{}'.format(str(k)))  # Scale
            b_k = getattr(self, 'b_{}'.format(str(k)))  # Bias

            if update_parameters is False:
                H_k, a_k, b_k = H_k.detach(), a_k.detach(), b_k.detach()
            logits = torch.bmm(F.softplus(H_k), logits)  # [C,filters[k+1],*]
            logits = logits + b_k
            logits = logits + torch.tanh(a_k) * torch.tanh(logits)

        return logits

    def quantization_offset(self, **kwargs):
        return 0.

    def lower_tail(self, tail_mass):
        cdf_logits_func = lambda x: self.cdf_logits(x, update_parameters=False)
        lt = estimate_tails(
            cdf_logits_func,
            target=-np.log(2. / tail_mass - 1.),
            shape=torch.Size((self.n_channels,1, 1))
        ).detach()
        return lt.reshape(self.n_channels)

    def upper_tail(self, tail_mass):
        cdf_logits_func = lambda x: self.cdf_logits(x, update_parameters=False)
        ut = estimate_tails(
            cdf_logits_func,
            target=np.log(2. / tail_mass - 1.),
            shape=torch.Size((self.n_channels,1, 1))
        ).detach()
        return ut.reshape(self.n_channels)

    def median(self):
        cdf_logits_func = lambda x: self.cdf_logits(x, update_parameters=False)
        _median = estimate_tails(
            cdf_logits_func,
            target=0.,
            shape=torch.Size((self.n_channels,1,1))
        ).detach()
        return _median.reshape(self.n_channels)

    def pdf(self, x):
        """
        Expected input: (N,C,H,W)
        """
        batch_size, channels, height, width = x.shape
        symbols = torch.arange(self.min_symbol, self.max_symbol + 1, device=x.device)
        num_symbols = len(symbols)
        dummy_input = torch.ones(batch_size, channels, height, width, num_symbols, device=x.device)
        dummy_input *= symbols[None, None, None, None, :]
        dummy_input = dummy_input.view(batch_size, channels, height, width * num_symbols)
        distribution = self.likelihood(dummy_input)
        distribution = distribution.view(batch_size, channels, height, width, num_symbols)
        return distribution

    def likelihood(self, x, collapsed_format=False, **kwargs):
        """
        Expected input: (N,C,H,W)
        """
        latents = x

        # Converts latents to (C,1,*) format

        if collapsed_format is False:
            N, C, H, W = latents.size()
            latents = latents.permute(1, 0, 2, 3)
            shape = latents.shape
            latents = torch.reshape(latents, (shape[0], 1, -1))

        cdf_upper = self.cdf_logits(latents + 0.5)
        cdf_lower = self.cdf_logits(latents - 0.5)

        # Numerical stability using some sigmoid identities
        # to avoid subtraction of two numbers close to 1
        sign = -torch.sign(cdf_upper + cdf_lower)
        sign = sign.detach()
        likelihood_ = torch.abs(torch.sigmoid(sign * cdf_upper) - torch.sigmoid(sign * cdf_lower))
        # Naive
        # likelihood_ = torch.sigmoid(cdf_upper) - torch.sigmoid(cdf_lower)

        likelihood_ = lower_bound_toward(likelihood_, self.min_likelihood)

        if collapsed_format is True:
            return likelihood_

        # Reshape to (N,C,H,W)
        likelihood_ = torch.reshape(likelihood_, shape)
        likelihood_ = likelihood_.permute(1, 0, 2, 3)

        return likelihood_

    def forward(self, x, **kwargs):
        return self.likelihood(x)

def estimate_tails(cdf, target, shape, dtype=torch.float32, extra_counts=24):
    """
    Estimates approximate tail quantiles.
    This runs a simple Adam iteration to determine tail quantiles. The
    objective is to find an `x` such that: [[[ cdf(x) == target ]]]

    Note that `cdf` is assumed to be monotonic. When each tail estimate has passed the
    optimal value of `x`, the algorithm does `extra_counts` (default 10) additional
    iterations and then stops.

    This operation is vectorized. The tensor shape of `x` is given by `shape`, and
    `target` must have a shape that is broadcastable to the output of `func(x)`.

    Arguments:
    cdf: A callable that computes cumulative distribution function, survival
         function, or similar.
    target: The desired target value.
    shape: The shape of the tensor representing `x`.
    Returns:
    A `torch.Tensor` representing the solution (`x`).
    """
    # A bit hacky
    lr, eps = 1e-2, 1e-8
    beta_1, beta_2 = 0.9, 0.99

    # Tails should be monotonically increasing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tails = torch.zeros(shape, dtype=dtype, requires_grad=True, device=device)

    m = torch.zeros(shape, dtype=dtype)
    v = torch.ones(shape, dtype=dtype)
    counts = torch.zeros(shape, dtype=torch.int32)

    while torch.min(counts) < extra_counts:
        loss = abs(cdf(tails) - target)
        loss.backward(torch.ones_like(tails))

        tgrad = tails.grad.cpu()

        with torch.no_grad():
            m = beta_1 * m + (1. - beta_1) * tgrad
            v = beta_2 * v + (1. - beta_2) * torch.square(tgrad)
            tails -= (lr * m / (torch.sqrt(v) + eps)).to(device)

        # Condition assumes tails init'd at zero
        counts = torch.where(
            torch.logical_or(counts > 0, tgrad.cpu() * tails.cpu() > 0),
            counts+1,
            counts
        )

        tails.grad.zero_()

    return tails