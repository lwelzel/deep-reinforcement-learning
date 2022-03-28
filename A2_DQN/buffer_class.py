import numpy as np
from sys import version_info


class MetaBuffer(object):
    """
    Meta CLass for buffers
    """
    n_args = 5  # s(t), a(t), r(t), s(t+1), done(t+1)
    # TODO: find better name
    n_actions_depth = 4
    dtype = np.float64  # might want to change this for discretization

    def __init__(self, depth=int(1e5), sample_batch_length=int(1e4), **kwargs):
        super(MetaBuffer).__init__()

        self._rng = np.random.default_rng()

        # meta selection and self-knowledge
        self._step = 0
        self.buffer_selecter = lambda **buffer_kwargs: None

        # buffer
        # s(t), a(t), r(t), s(t+1), done(t+1)
        self._sample_batch_length = sample_batch_length  # TODO: random value, decide on reasonable

        self._buffer = np.zeros(shape=(depth,
                                       self.__class__.n_args,
                                       self.__class__.n_actions_depth), dtype=object)
        self._transition = np.zeros(shape=(self.__class__.n_args,
                                           self.__class__.n_actions_depth), dtype=object)
        self._transition_length = np.prod(self._transition.shape)

    def __repr__(self):
        return '<%s.%s MetaBuffer object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

    def update_buffer(self, transition: np.ndarray):
        """
        Updates the replay buffer with the transition.
        The transition is a tuple or np.array of (s(t), a(t), r(t), s(t+1), done(t+1))
        If the transition is a tuple it will be implicitly be cast to a np.array with type self.__class__.dtype

        :param transition: a tuple of s(t), a(t), r(t), s(t+1), done(t+1)
        """

        # TODO: avoid packing in the function calling the update, *args tuple or implicit to-tuple cast?
        # TODO: should instead cast to right shape, write @staticmethod?
        for i, column in enumerate(transition):
            self._transition[i, :] = column

        # TODO: remove is proper reshaping certain
        # assert _transition.shape == (self.__class__.n_args, self.__class__.n_actions_depth), \
        #     "transition must be consistent np.ndarray - current required shape: (5, 4)"

        np.put(a=self._buffer,
               ind=np.arange(self._step * self._transition_length,
                             self._step * self._transition_length + self._transition_length),
               v=self._transition, mode="wrap")
        self._step += 1


    @classmethod
    def __reset_all__(cls):
        pass

    @property
    def sample(self):
        # TODO: when implemented long batches:
        # return self._rng.choice(self._buffer, self._sample_batch_length, replace=False, )
        # for sequential buffer samples:
        # return np.take(self._buffer, indices, axis=None, out=None, mode='raise')

        return self._rng.choice(self._buffer, 1, replace=False, )


if __name__ == "__main__":
    meta_buffer = MetaBuffer(depth=1000)

    # for i, transition in enumerate(np.arange(0., 2000. * 5, 1).reshape((2000, 5))):
    #     meta_buffer.update_buffer(transition=transition)
    #
    # for i in np.arange(2000 * 5, (2000 + 10) * 5, 1).reshape((10, 5)):
    #     print(meta_buffer.sample)
