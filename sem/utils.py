import os
import sys
import traceback
import numpy as np
from functools import wraps
from multiprocessing import Process, Queue


def unroll_data(x, t=1):
    """
    This function is used by recurrent neural nets to do back-prop through time.

    Unrolls a data_set for with time-steps, truncated for t time-steps
    appends t-1 D-dimensional zero vectors at the beginning.

    Parameters:
        x: array, shape (N, D) or shape (D,)

        t: int
            time-steps to truncate the unroll

    output
    ------

        X_unrolled: array, shape (N-1, t, D)

    """
    if np.ndim(x) == 2:
        n, d = np.shape(x)
    elif np.ndim(x):
        n, d = 1, np.shape(x)[0]
        x = np.reshape(x, (1, d))

    x_unrolled = np.zeros((n, t, d))

    # append a t-1 blank (zero) input patterns to the beginning
    data_set = np.concatenate([np.zeros((t - 1, d)), x])

    for ii in range(n):
        x_unrolled[ii, :, :] = data_set[ii: ii + t, :]

    return x_unrolled

# precompute for speed (doesn't really help but whatever)
log_2pi = np.log(2.0 * np.pi)

def fast_mvnorm_diagonal_logprob(x, variances):
    """
    Assumes a zero-mean mulitivariate normal with a diagonal covariance function

    Parameters:

        x: array, shape (D,)
            observations

        variances: array, shape (D,)
            Diagonal values of the covariance function

    output
    ------

        log-probability: float

    """
    return -0.5 * (log_2pi * np.shape(x)[0] + np.sum(np.log(variances) + (x**2) / variances ))


def get_prior_scale(df, target_variance):
    """
    This function solves for the scale parameter need for a scaled inverse chi-squard 
    distribution, give degrees of freedom (df) and the desired variance (i.e. the 
    mode of the distribution, is this function is intended to determine the prior over
    a Guassian variance).
      
    The mode of a scaled-inverse chi-squared is defined:
    (see Gelman, et al., Bayesian Data Analysis 2004)

    mode(theta) = df / (df + 2) * scale

    hense, if we set mode(theta) to our target, then the scale is

    scale = target_variance * (df + 2) / df

    """
    return target_variance * (df + 2) / df

def delete_object_attributes(myobj):
    # take advantage of mutability here
    while myobj.__dict__.items():
        attr = [k for k in myobj.__dict__.keys()][0]
        myobj.__delattr__(attr)
    
### UPDATE, 11/17/20: Code below does not work with python 3.7+ (at least)
# def processify(func):
#     '''Decorator to run a function as a process.
#     Be sure that every argument and the return value
#     is *pickable*.
#     The created process is joined, so the code does not
#     run in parallel.

#     Credit: I took this function from Marc Schlaich's github:
#     https://gist.github.com/schlamar/2311116
#     '''

#     def process_func(q, *args, **kwargs):
#         try:
#             ret = func(*args, **kwargs)
#         except Exception:
#             ex_type, ex_value, tb = sys.exc_info()
#             error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
#             ret = None
#         else:
#             error = None

#         q.put((ret, error))

#     # register original function with different name
#     # in sys.modules so it is pickable
#     # NTF 11/17/20: this hack is no longer working and the wrapper no longer works!
#     process_func.__name__ = func.__name__ + 'processify_func'
#     setattr(sys.modules[__name__], process_func.__name__, process_func)

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         q = Queue()
#         p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
#         p.start()
#         ret, error = q.get()
#         p.join()

#         if error:
#             ex_type, ex_value, tb_str = error
#             message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
#             raise ex_type(message)

#         return ret
#     return wrapper