import numpy as np


def embed_gaussian(d, n=1):
    """
    returns n normal vectors with variance = 1/n, inline with Plate's caluclations

    :param d: (int), dimensions of the embedding
    :param n: (int, default=1), number of embeddings to return

    :return: d-length np.array
    """
    return np.random.normal(loc=0., scale=1./np.sqrt(d), size=(n, d))


def conv_circ(signal, kernal, n=None):
    '''
    Parameters
    ----------

    signal: array of length D

    ker: array of length D

    Returns
    -------

    array of length D

    '''
    if n == None:
        n = len(signal) + len(kernal) - 1

    return np.real(np.fft.ifft(np.fft.fft(signal, n) * np.fft.fft(kernal, n)))


def plate_formula(n, k, err):
    '''
    Determine the number of dimensions needed according to Plate's (2003)
    formula:
      D = 3.16(K-0.25)ln(N/err^3)
    where D is the number of dimensions, K is the maximum number of terms
    to be combined, N is the number of atomic values in the language, and
    err is the probability of error.

    USAGE: D = plate_formula(n, k, err)
    '''
    return int(round(3.16 * (k - 0.25) * (np.log(n) - 3 * np.log(err))))


def embed(n, d, distr='spikeslab_gaussian', param=None):
    # Embed symbols in a vector space.
    #
    # USAGE: X = embed(n, d, distr='spikeslab_gaussian', param=None)
    #
    # INPUTS:
    #   n - number of symbols
    #   d - number of dimensions
    #   distr - string specifying the distribution on the vector space:
    #           'spikeslab_gaussian' - mixture of Gaussian "slab" and Bernoulli "spike"
    #           'spikeslab_uniform' - mixture of uniform "slab" and Bernoulli "spike"
    #
    #   param (optional) - parameters of the distribution:
    #                      'spikeslab_gaussian' - param = [variance, spike probability] (default: [1 1])
    #                      'spikeslab_uniform' - param = [bound around 0, spike probability] (default: [1 1])
    # OUTPUTS;
    #   X - [N x D] matrix
    #
    # Sam Gershman, Jan 2013

    if param is None:
        param = [1, 1]
    spike = np.round(np.random.rand(n, d) < param[1])

    if distr == 'spikeslab_gaussian':
        slab = np.random.randn(n, d) * param[1]
    elif distr == 'spikeslab_uniform':
        slab = np.random.uniform(-param[1], param[1], (n, d))
    else:
        raise (Exception)

    return spike * slab


def encode(a, b):
    return conv_circ(a, b, np.size(a))


def embed_onehot(n, d):
    v = np.zeros((n, d))
    for ii in range(n):
        v[ii][np.random.randint(d)] = 1
    return v


def decode(a, b):
    c = np.real(np.fft.ifft(np.fft.fft(a, np.size(a)) * np.conj(np.fft.fft(b, np.size(a)))))
    return c / np.size(a)