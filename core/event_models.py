import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, GRU, Dropout, LSTM, LeakyReLU, Lambda
from tensorflow.keras.initializers import glorot_uniform  # Or your initializer of choice
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import l2_normalize
from core.utils import fast_mvnorm_diagonal_logprob, unroll_data

print("TensorFlow Version: {}".format(tf.__version__))

### there are a ~ton~ of tf warnings from Keras, suppress them here
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# run a check that tensorflow works on import
def check_tf():
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

    with tf.compat.v1.Session() as sess:
        sess.run(c)
    print("TensorFlow Check Passed")
check_tf()



def reset_weights(session, model):
    for layer in model.layers:
        if hasattr(layer, "kernel_initializer"):
            layer.kernel.initializer.run(session=session)


def map_variance(samples, df0, scale0):
    """
    This estimator assumes an scaled inverse-chi squared prior over the
    variance and a Gaussian likelihood. The parameters d and scale
    of the internal function parameterize the posterior of the variance.
    Taken from Bayesian Data Analysis, ch2 (Gelman)

    samples: N length array or NxD array
    df0: prior degrees of freedom
    scale0: prior scale parameter
    mu: (optional) mean function

    returns: float or d-length array, mode of the posterior
    """
    if np.ndim(samples) > 1:
        n, d = np.shape(samples)
    else:
        n = np.shape(samples)[0]
        d = 1

    v = np.var(samples, axis=0)
    df = df0 + n
    scale = (df0 * scale0 + n * v) / df
    return df * scale / (df * 2)


class LinearEvent(object):
    """ this is the base clase of the event model """

    def __init__(self, d, var_df0, var_scale0, optimizer=None, n_epochs=10, init_model=False,
                 kernel_initializer='glorot_uniform', l2_regularization=0.00, batch_size=32, prior_log_prob=0.0,
                 reset_weights=False, batch_update=True, optimizer_kwargs=None):
        """

        :param d: dimensions of the input space
        """
        self.d = d
        self.f_is_trained = False
        self.f0_is_trained = False
        self.f0 = np.zeros(d)

        self.x_history = [np.zeros((0, self.d))]
        self.prior_probability = prior_log_prob

        if (optimizer is None) and (optimizer_kwargs is None):
            optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
        elif (optimizer is None) and not (optimizer_kwargs is None):
            optimizer = Adam(**optimizer_kwargs)
        elif (optimizer is not None) and (type(optimizer) != str):
            optimizer = optimizer()

        self.compile_opts = dict(optimizer=optimizer, loss='mean_squared_error')
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = regularizers.l2(l2_regularization)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.var_df0 = var_df0
        self.var_scale0 = var_scale0
        self.d = d
        self.reset_weights = reset_weights
        self.batch_update = batch_update
        self.training_pairs = []
        self.prediction_errors = np.zeros((0, self.d), dtype=np.float)
        self.model_weights = None

        # initialize the covariance with the mode of the prior distribution
        self.Sigma = np.ones(d) * var_df0 * var_scale0 / (var_df0 + 2)

        self.is_visited = False  # governs the special case of model's first prediction (i.e. with no experience)

        # switch for inheritance -- don't want to init the model for sub-classes
        if init_model:
            self.init_model()

    def init_model(self):
        self._compile_model()
        self.model_weights = self.model.get_weights()
        return self.model

    def _compile_model(self):
        self.model = Sequential([
            Dense(self.d, input_shape=(self.d,), use_bias=True, kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer),
            Activation('linear')
        ])
        self.model.compile(**self.compile_opts)

    def set_model(self, sess, model):
        self.sess = sess
        self.model = model
        self.do_reset_weights()

    def reestimate(self):
        self.do_reset_weights()
        self.estimate()

    def do_reset_weights(self):
        # self._compile_model()
        reset_weights(self.sess, self.model)
        self.model_weights = self.model.get_weights()

    def update(self, X, Xp, update_estimate=True):
        """
        Parameters
        ----------
        X: NxD array-like data of inputs

        y: NxD array-like data of outputs

        Returns
        -------
        None

        """
        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert X.ndim == 1
        assert X.shape[0] == self.d
        assert Xp.ndim == 1
        assert Xp.shape[0] == self.d

        x_example = X.reshape((1, self.d))
        xp_example = Xp.reshape((1, self.d))

        # concatenate the training example to the active event token
        self.x_history[-1] = np.concatenate([self.x_history[-1], x_example], axis=0)

        # also, create a list of training pairs (x, y) for efficient sampling
        #  picks  random time-point in the history
        self.training_pairs.append(tuple([x_example, xp_example]))

        if update_estimate:
            self.estimate()
            self.f_is_trained = True

    def update_f0(self, Xp, update_estimate=True):
        self.update(np.zeros(self.d), Xp, update_estimate=update_estimate)
        self.f0_is_trained = True

        # precompute f0 for speed
        self.f0 = self._predict_f0()

    def get_variance(self):
        # Sigma is stored as a vector corresponding to the entries of the diagonal covariance matrix
        return self.Sigma

    def predict_next(self, X):
        """
        wrapper for the prediction function that changes the prediction to the identity function
        for untrained models (this is an initialization technique)

        """
        if not self.f_is_trained:
            if np.ndim(X) > 1:
                return np.copy(X[-1, :]).reshape(1, -1)
            return np.copy(X).reshape(1, -1)

        return self._predict_next(X)

    def _predict_next(self, X):
        """
        Parameters
        ----------
        X: 1xD array-like data of inputs

        Returns
        -------
        y: 1xD array of prediction vectors

        """
        if X.ndim > 1:
            X0 = X[-1, :]
        else:
            X0 = X
 
        self.model.set_weights(self.model_weights)
        return self.model.predict(np.reshape(X0, newshape=(1, self.d)))

    def predict_f0(self):
        """
        wrapper for the prediction function that changes the prediction to the identity function
        for untrained models (this is an initialization technique)

        N.B. This answer is cached for speed

        """
        return self.f0

    def _predict_f0(self):
        return self._predict_next(np.zeros(self.d))

    def log_likelihood_f0(self, Xp):

        if not self.f0_is_trained:
            return self.prior_probability

        # predict the initial point
        Xp_hat = self.predict_f0()

        # return the probability
        return fast_mvnorm_diagonal_logprob(Xp.reshape(-1) - Xp_hat.reshape(-1), self.Sigma)

    def log_likelihood_next(self, X, Xp):
        if not self.f_is_trained:
            return self.prior_probability

        Xp_hat = self.predict_next(X)
        return fast_mvnorm_diagonal_logprob(Xp.reshape(-1) - Xp_hat.reshape(-1), self.Sigma)

    def log_likelihood_sequence(self, X, Xp):
        if not self.f_is_trained:
            return self.prior_probability

        Xp_hat = self.predict_next_generative(X)
        return fast_mvnorm_diagonal_logprob(Xp.reshape(-1) - Xp_hat.reshape(-1), self.Sigma)

    # create a new cluster of scenes
    def new_token(self):
        if len(self.x_history) == 1 and self.x_history[0].shape[0] == 0:
            # special case for the first cluster which is already created
            return
        self.x_history.append(np.zeros((0, self.d)))

    def predict_next_generative(self, X):
        self.model.set_weights(self.model_weights)
        # the LDS is a markov model, so these functions are the same
        return self.predict_next(X)

    def run_generative(self, n_steps, initial_point=None):
        self.model.set_weights(self.model_weights)
        if initial_point is None:
            x_gen = self._predict_f0()
        else:
            x_gen = np.reshape(initial_point, (1, self.d))
        for ii in range(1, n_steps):
            x_gen = np.concatenate([x_gen, self.predict_next_generative(x_gen[:ii, :])])
        return x_gen

    def estimate(self):
        if self.reset_weights:
            self.do_reset_weights()
        else:
            self.model.set_weights(self.model_weights)

        n_pairs = len(self.training_pairs)

        if self.batch_update:
            def draw_sample_pair():
                # draw a random cluster for the history
                idx = np.random.randint(n_pairs)
                return self.training_pairs[idx]
        else:
            # for online sampling, just use the last training sample
            def draw_sample_pair():
                return self.training_pairs[-1]

        # run batch gradient descent on all of the past events!
        for _ in range(self.n_epochs):

            # draw a set of training examples from the history
            x_batch = []
            xp_batch = []
            for _ in range(self.batch_size):

                x_sample, xp_sample = draw_sample_pair()

                # these data aren't
                x_batch.append(x_sample)
                xp_batch.append(xp_sample)

            x_batch = np.reshape(x_batch, (self.batch_size, self.d))
            xp_batch = np.reshape(xp_batch, (self.batch_size, self.d))
            self.model.train_on_batch(x_batch, xp_batch)

        # cache the model weights
        self.model_weights = self.model.get_weights()

        # Update Sigma
        x_train_0, xp_train_0 = self.training_pairs[-1]
        xp_hat = self.model.predict(x_train_0)
        self.prediction_errors = np.concatenate([self.prediction_errors, xp_train_0 - xp_hat], axis=0)
        if np.shape(self.prediction_errors)[0] > 1:
            self.Sigma = map_variance(self.prediction_errors, self.var_df0, self.var_scale0)


class NonLinearEvent(LinearEvent):

    def __init__(self, d, var_df0, var_scale0, n_hidden=None, hidden_act='tanh', batch_size=32,
                 optimizer=None, n_epochs=10, init_model=False, kernel_initializer='glorot_uniform',
                 l2_regularization=0.00, dropout=0.50, prior_log_prob=0.0, reset_weights=False,
                 batch_update=True,
                 optimizer_kwargs=None):
        LinearEvent.__init__(self, d, var_df0, var_scale0, optimizer=optimizer, n_epochs=n_epochs,
                             init_model=False, kernel_initializer=kernel_initializer, batch_size=batch_size,
                             l2_regularization=l2_regularization, prior_log_prob=prior_log_prob,
                             reset_weights=reset_weights, batch_update=batch_update,
                             optimizer_kwargs=optimizer_kwargs)

        if n_hidden is None:
            n_hidden = d
        self.n_hidden = n_hidden
        self.hidden_act = hidden_act
        self.dropout = dropout

        if init_model:
            self.init_model()

    def _compile_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.n_hidden, input_shape=(self.d,), activation=self.hidden_act,
                             kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.add(Dropout(rate=1-self.dropout))
        self.model.add(Dense(self.d, activation='linear',
                             kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class NonLinearEvent_normed(NonLinearEvent):

    def __init__(self, d, var_df0, var_scale0, n_hidden=None, hidden_act='tanh',
                 optimizer=None, n_epochs=10, init_model=False, kernel_initializer='glorot_uniform',
                 l2_regularization=0.00, dropout=0.50, prior_log_prob=0.0, reset_weights=False, batch_size=32,
                 batch_update=True, optimizer_kwargs=None):

        NonLinearEvent.__init__(self, d, var_df0, var_scale0, optimizer=optimizer, n_epochs=n_epochs,
                                     l2_regularization=l2_regularization,batch_size=batch_size,
                                     kernel_initializer=kernel_initializer, init_model=False,
                                     prior_log_prob=prior_log_prob, reset_weights=reset_weights,
                                     batch_update=batch_update, optimizer_kwargs=optimizer_kwargs)

        if n_hidden is None:
            n_hidden = d
        self.n_hidden = n_hidden
        self.hidden_act = hidden_act
        self.dropout = dropout

        if init_model:
            self.init_model()

    def _compile_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.n_hidden, input_shape=(self.d,), activation=self.hidden_act,
                             kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.add(Dropout(rate=1-self.dropout))
        self.model.add(Dense(self.d, activation='linear',
                             kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.add(Lambda(lambda x: l2_normalize(x, axis=-1)))  
        self.model.compile(**self.compile_opts)


class StationaryEvent(LinearEvent):

    def _predict_next(self, X):
        """
        Parameters
        ----------
        X: 1xD array-like data of inputs

        Returns
        -------
        y: 1xD array of prediction vectors

        """

        return self.model.predict(np.zeros((1, self.d)))



class RecurentLinearEvent(LinearEvent):

    # RNN which is initialized once and then trained using stochastic gradient descent
    # i.e. each new scene is a single example batch of size 1

    def __init__(self, d, var_df0, var_scale0, t=3,
                 optimizer=None, n_epochs=10, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=False, prior_log_prob=0.0, reset_weights=False,
                 batch_update=True, optimizer_kwargs=None):
        #
        # D = dimension of single input / output example
        # t = number of time steps to unroll back in time for the recurrent layer
        # n_hidden1 = # of nodes in first hidden layer
        # n_hidden2 = # of nodes in second hidden layer
        # hidden_act1 = activation f'n of first hidden layer
        # hidden_act2 = activation f'n of second hidden layer
        # sgd_kwargs = arguments for the stochastic gradient descent algorithm
        # n_epochs = how many gradient descent steps to perform for each training batch
        # dropout = what fraction of nodes to drop out during training (to prevent overfitting)

        LinearEvent.__init__(self, d, var_df0, var_scale0, optimizer=optimizer, n_epochs=n_epochs,
                             init_model=False, kernel_initializer=kernel_initializer,
                             l2_regularization=l2_regularization, prior_log_prob=prior_log_prob,
                             reset_weights=reset_weights, batch_update=batch_update, optimizer_kwargs=optimizer_kwargs)

        self.t = t
        self.n_epochs = n_epochs

        # list of clusters of scenes:
        # each element of list = history of scenes for given cluster
        # history = N x D tensor, N = # of scenes in cluster, D = dimension of single scene
        #
        self.x_history = [np.zeros((0, self.d))]
        self.batch_size = batch_size

        if init_model:
            self.init_model()

        # cache the initial weights for retraining speed
        self.init_weights = None

    def do_reset_weights(self):
        # # self._compile_model()
        if self.init_weights is None:
            for layer in self.model.layers:
                new_weights = [glorot_uniform()(w.shape).eval(session=self.sess) for w in layer.get_weights()]
                layer.set_weights(new_weights)
            self.model_weights = self.model.get_weights()
            self.init_weights = self.model.get_weights()
        else:
            self.model.set_weights(self.init_weights)

    # initialize model once so we can then update it online
    def _compile_model(self):
        self.model = Sequential()
        self.model.add(SimpleRNN(self.d, input_shape=(self.t, self.d),
                                 activation=None, kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)

    # concatenate current example with the history of the last t-1 examples
    # this is for the recurrent layer
    #
    def _unroll(self, x_example):
        x_train = np.concatenate([self.x_history[-1][-(self.t - 1):, :], x_example], axis=0)
        x_train = np.concatenate([np.zeros((self.t - x_train.shape[0], self.d)), x_train], axis=0)
        x_train = x_train.reshape((1, self.t, self.d))
        return x_train

    # predict a single example
    def _predict_next(self, X):
        self.model.set_weights(self.model_weights)
        # Note: this function predicts the next conditioned on the training data the model has seen

        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert np.ndim(X) == 1
        assert X.shape[0] == self.d

        x_test = X.reshape((1, self.d))

        # concatenate current example with history of last t-1 examples
        # this is for the recurrent part of the network
        x_test = self._unroll(x_test)
        return self.model.predict(x_test)

    def _predict_f0(self):
        return self.predict_next_generative(np.zeros(self.d))

    def _update_variance(self):
        if np.shape(self.prediction_errors)[0] > 1:
            self.Sigma = map_variance(self.prediction_errors, self.var_df0, self.var_scale0)

    def update(self, X, Xp, update_estimate=True):
        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert X.ndim == 1
        assert X.shape[0] == self.d
        assert Xp.ndim == 1
        assert Xp.shape[0] == self.d

        x_example = X.reshape((1, self.d))
        xp_example = Xp.reshape((1, self.d))

        # concatenate the training example to the active event token
        self.x_history[-1] = np.concatenate([self.x_history[-1], x_example], axis=0)

        # also, create a list of training pairs (x, y) for efficient sampling
        #  picks  random time-point in the history
        _n = np.shape(self.x_history[-1])[0]
        x_train_example = np.reshape(
                    unroll_data(self.x_history[-1][max(_n - self.t, 0):, :], self.t)[-1, :, :], (1, self.t, self.d)
                )
        self.training_pairs.append(tuple([x_train_example, xp_example]))

        if update_estimate:
            self.estimate()
            self.f_is_trained = True

    def predict_next_generative(self, X):
        self.model.set_weights(self.model_weights)
        X0 = np.reshape(unroll_data(X, self.t)[-1, :, :], (1, self.t, self.d))
        return self.model.predict(X0)

    # optional: run batch gradient descent on all past event clusters
    def estimate(self):
        if self.reset_weights:
            self.do_reset_weights()
        else:
            self.model.set_weights(self.model_weights)

        n_pairs = len(self.training_pairs)

        if self.batch_update:
            def draw_sample_pair():
                # draw a random cluster for the history
                idx = np.random.randint(n_pairs)
                return self.training_pairs[idx]
        else:
            # for online sampling, just use the last training sample
            def draw_sample_pair():
                return self.training_pairs[-1]

        # run batch gradient descent on all of the past events!
        for _ in range(self.n_epochs):

            # draw a set of training examples from the history
            x_batch = np.zeros((0, self.t, self.d))
            xp_batch = np.zeros((0, self.d))
            for _ in range(self.batch_size):

                x_sample, xp_sample = draw_sample_pair()

                x_batch = np.concatenate([x_batch, x_sample], axis=0)
                xp_batch = np.concatenate([xp_batch, xp_sample], axis=0)

            self.model.train_on_batch(x_batch, xp_batch)
        self.model_weights = self.model.get_weights()

        # Update Sigma
        x_train_0, xp_train_0 = self.training_pairs[-1]
        xp_hat = self.model.predict(x_train_0)
        self.prediction_errors = np.concatenate([self.prediction_errors, xp_train_0 - xp_hat], axis=0)
        self._update_variance()
 

class RecurrentEvent(RecurentLinearEvent):

    def __init__(self, d, var_df0, var_scale0, t=3, n_hidden=None, optimizer=None,
                 n_epochs=10, dropout=0.50, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=False, prior_log_prob=0.0, reset_weights=False, 
                 batch_update=True, optimizer_kwargs=None):

        RecurentLinearEvent.__init__(self, d, var_df0, var_scale0, t=t, optimizer=optimizer, n_epochs=n_epochs,
                                     l2_regularization=l2_regularization, batch_size=batch_size,
                                     kernel_initializer=kernel_initializer, init_model=False, prior_log_prob=prior_log_prob,
                                     reset_weights=reset_weights, batch_update=batch_update, optimizer_kwargs=optimizer_kwargs)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.dropout = dropout

        if init_model:
            self.init_model()

    def _compile_model(self):
        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(SimpleRNN(self.n_hidden, input_shape=(self.t, self.d),
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(rate=1-self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                  kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class GRUEvent(RecurentLinearEvent):

    def __init__(self, d, var_df0, var_scale0, t=3, n_hidden=None, optimizer=None,
                 n_epochs=10, dropout=0.50, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=False, prior_log_prob=0.0, reset_weights=False,
                 batch_update=True, optimizer_kwargs=None):

        RecurentLinearEvent.__init__(self, d, var_df0, var_scale0, t=t, optimizer=optimizer, n_epochs=n_epochs,
                                     l2_regularization=l2_regularization, batch_size=batch_size,
                                     kernel_initializer=kernel_initializer, init_model=False,
                                     prior_log_prob=prior_log_prob, reset_weights=reset_weights,
                                     batch_update=batch_update, optimizer_kwargs=optimizer_kwargs)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.dropout = dropout

        if init_model:
            self.init_model()

    def _compile_model(self):
        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(GRU(self.n_hidden, input_shape=(self.t, self.d),
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(rate=1-self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                  kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class GRUEvent_normed(RecurentLinearEvent):

    def __init__(self, d, var_df0, var_scale0, t=3, n_hidden=None, optimizer=None,
                 n_epochs=10, dropout=0.50, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=False, prior_log_prob=0.0, reset_weights=False,
                 batch_update=True, optimizer_kwargs=None):

        RecurentLinearEvent.__init__(self, d, var_df0, var_scale0, t=t, optimizer=optimizer, n_epochs=n_epochs,
                                     l2_regularization=l2_regularization, batch_size=batch_size,
                                     kernel_initializer=kernel_initializer, init_model=False,
                                     prior_log_prob=prior_log_prob, reset_weights=reset_weights,
                                     batch_update=batch_update, optimizer_kwargs=optimizer_kwargs)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.dropout = dropout

        if init_model:
            self.init_model()

    def _compile_model(self):
        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(GRU(self.n_hidden, input_shape=(self.t, self.d),
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(rate = 1-self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                  kernel_initializer=self.kernel_initializer))
        self.model.add(Lambda(lambda x: l2_normalize(x, axis=-1)))  
        self.model.compile(**self.compile_opts)



class GRUEvent_spherical_noise(GRUEvent):

    def _update_variance(self):
        if np.shape(self.prediction_errors)[0] > 1:
            var = map_variance(self.prediction_errors.reshape(-1), self.var_df0, self.var_scale0)
            self.Sigma = var * np.ones(self.d)



class LSTMEvent(RecurentLinearEvent):

    def __init__(self, d, var_df0, var_scale0, t=3, n_hidden=None, optimizer=None,
                 n_epochs=10, dropout=0.50, l2_regularization=0.00,
                 batch_size=32, kernel_initializer='glorot_uniform', init_model=False, prior_log_prob=0.0,
                 reset_weights=False, batch_update=True, optimizer_kwargs=None):

        RecurentLinearEvent.__init__(self, d, var_df0, var_scale0, t=t, optimizer=optimizer, n_epochs=n_epochs,
                                     l2_regularization=l2_regularization, batch_size=batch_size,
                                     kernel_initializer=kernel_initializer, init_model=False,
                                     prior_log_prob=prior_log_prob, reset_weights=reset_weights,
                                     batch_update=batch_update, optimizer_kwargs=optimizer_kwargs)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.dropout = dropout

        if init_model:
            self.init_model()

    def _compile_model(self):
        self.model = Sequential()
        # input_shape[0] = time-steps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(LSTM(self.n_hidden, input_shape=(self.t, self.d),
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(rate = 1-self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)
