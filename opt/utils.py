import pandas as pd
import numpy as np
import models
import cPickle as pickle
from sklearn.metrics import adjusted_rand_score


def generate_random_events(n_events, data_file=None):
    """

    Parameters
    ----------
    n_events: int

    data_file: str
        full file path of the Reynolds, Braver, & Zachs data.
        contains pandas dataframe (pickled) with 13 events of
        8-12 time-points and 54 dimensions

    :return:
    """

    if data_file is None:
        data_file = './datasets/motion_data.pkl'
    motion_data = pd.read_pickle(data_file)
    n_patterns = len(set(motion_data.EventNumber))

    z = np.mean(np.linalg.norm(motion_data.values[:, :-1], axis=1))

    X = []
    y = []
    p_prev = -1
    for _ in range(n_events):
        while True:
            p = np.random.randint(n_patterns)
            if p != p_prev:
                p_prev = p
                break
        e = motion_data.loc[motion_data.EventNumber == p, :].values[:, :-1]
        X.append(e / z)
        y.append([p] * e.shape[0])
    return np.concatenate(X), np.concatenate(y)


def evaluate(x, y, omega, k=None, number=0, save=False, list_event_boundaries=None):
    """

    Parameters
    ----------
    x: NxD array
        scene vectors

    y: array of length N
        true class labels

    omega: dict
        dictionary of kwargs for the SEM model

    k: int
        maximum number of clusters


    Return
    ------
        r: int, adjusted rand score
    """

    sem = models.SEM(**omega)

    if k is None:
        k = x.shape[0] / 2

    sem.run(x, k=k)

    y_hat = np.argmax(sem.results.post, axis=1)

    r = adjusted_rand_score(y, y_hat)

    if save:
        f = open('SEM_sample_%d.save' % number, 'wb')

        pickle.dump({'AdjRandScore': r, 'Omega': omega}, f)
        f.close()
        return

    return sem, r


# generate random string
#
def randstr(N=10):
    import string
    import random
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


if __name__ == '__main__':
    pass
