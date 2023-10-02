import math

import attr

from lib.lang_algorithm.pcfg import string_log_probability

def compute_lower_bound_perplexity(sampler, num_valid_lengths, samples):
    """
    Compute the lower bound perplexity of a specific set of strings generated
    by a PCFG subject to length constraints.
    """
    # Let p_G be the probability distribution over strings defined by the PCFG.
    # For a given string w with length l, we compute p(w) as
    # p(w) = p(l) p_G(w) / \sum_{w' where |w'| = l } p_G(w').
    parts = compute_lower_bound_parts(sampler, samples)
    return parts_to_perplexity(parts, num_valid_lengths)

@attr.s
class Parts:
    total_neg_log_prob = attr.ib()
    total_len = attr.ib()
    num_samples = attr.ib()

def compute_lower_bound_parts(sampler, samples):
    r"""
    Compute \sum_w log p(w) / \sum_{w' where |w'| = |w|} p_G(w') and
    \sum_w |w| for w in samples.
    """
    total_neg_log_prob = 0.0
    total_len = 0
    num_samples = 0
    for sample in samples:
        total_neg_log_prob -= sampler.log_probability_given_length(sample)
        # Remember to include EOS in the denominator for consistency with
        # model perplexity.
        total_len += len(sample) + 1
        num_samples += 1
    return Parts(total_neg_log_prob, total_len, num_samples)

def parts_to_perplexity(parts, num_valid_lengths):
    length_neg_log_prob = math.log(num_valid_lengths)
    neg_log_prob = parts.total_neg_log_prob + length_neg_log_prob * parts.num_samples
    return math.exp(neg_log_prob / parts.total_len)

def compute_cross_entropy_diff(perplexity, lower_bound_perplexity):
    return math.log(perplexity) - math.log(lower_bound_perplexity)
