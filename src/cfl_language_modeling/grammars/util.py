def mean_to_stop_prob(mean):
    # If X is a nonterminal with rules
    #   1. X -> a X / 1-p
    #   2. X -> \epsilon / p
    # Then |X| follows a negative binomial distibution with r = 1, so that
    # the mean of the distribution is r (1-p) / p = (1-p) / p.
    # See https://mathworld.wolfram.com/NegativeBinomialDistribution.html
    # and https://en.wikipedia.org/wiki/Negative_binomial_distribution
    # (Note that at the time of writing the mean given on Wikipedia is just
    # plain wrong.)
    # To give |X| a mean of m, we can set p = 1 / (m+1).
    return 1 / (mean + 1)

def mean_to_continue_prob(mean):
    return 1 - mean_to_stop_prob(mean)
