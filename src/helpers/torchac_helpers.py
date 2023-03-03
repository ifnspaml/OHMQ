import torch


def estimate_bitrate_from_pmf(pmf, sym):
    L = pmf.shape[-1]
    pmf = pmf.reshape(-1, L)
    sym = sym.reshape(-1, 1)
    assert pmf.shape[0] == sym.shape[0]

    relevant_probabilities = torch.gather(pmf, dim=1, index=sym.to(torch.int64))
    bitrate = torch.sum(-torch.log2(relevant_probabilities + 1e-8))

    return bitrate.item()


def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)

    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)

    return cdf_with_0
