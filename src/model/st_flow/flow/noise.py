import torch


class PriorSampler:
    def __init__(self, prior_sample_type, **kwargs):
        self.prior_sample_type = prior_sample_type

        if prior_sample_type == "gaussian":
            self.prior_sampler = gaussian_prior
        elif prior_sample_type == "zero":
            self.prior_sampler = all_zeros
        elif prior_sample_type == "zinb":
            # https://github.com/scverse/scvi-tools/blob/main/src/scvi/distributions/_negative_binomial.py#L433
            from scvi.distributions import ZeroInflatedNegativeBinomial

            prior_sampler = ZeroInflatedNegativeBinomial(
                                        total_count=kwargs.get("total_count", None),
                                        logits=kwargs.get("logits", None),
                                        zi_logits=kwargs.get("zi_logits", None),  # real number
                                    )
            self.prior_sampler = lambda shape: prior_sampler.sample(shape).squeeze(-1)
        else:
            raise ValueError("Invalid prior sample type")

    def sample(self, shape):
        return self.prior_sampler(shape)


def gaussian_prior(shape):
    return torch.randn(shape)


def all_zeros(shape):
    return torch.zeros(shape)