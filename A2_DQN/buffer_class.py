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

    def __init__(self, depth=1000, **kwargs):
        super(MetaBuffer).__init__()

        self._rng = np.random.default_rng()

        # meta selection and self-knowledge
        self._step = 0
        self.buffer_selecter = lambda **buffer_kwargs: None

        # buffer
        # s(t), a(t), r(t), s(t+1), done(t+1)
        self._buffer = np.zeros(shape=(depth,
                                       self.__class__.n_args,
                                       self.__class__.n_actions_depth), dtype=object)

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

        # TODO: should instead cast to right shape, write @staticmethod?
        _transition_depth = np.max([len(column) for column in transition])
        _transition = np.empty(self.__class__.n_args, _transition_depth)
        for i, column in enumerate(transition):
            _transition[i, :] = column

        _len_data_set = np.product(_transition.shape)

        # TODO: remove is proper reshaping certain
        # assert _transition.shape == (self.__class__.n_args, self.__class__.n_actions_depth), \
        #     "transition must be consistent np.ndarray - current required shape: (5, 4)"

        np.put(a=self._buffer,
               ind=np.arange(self._step * _len_data_set,
                             self._step * _len_data_set + _len_data_set),
                             v=_transition, mode="wrap")
        self._step += 1

    @classmethod
    def __reset_all__(cls):
        pass

    @property
    def sample(self):
        return self._rng.choice(self._buffer, 1, replace=False, )


if __name__ == "__main__":
    meta_buffer = MetaBuffer(depth=1000)

    for i, transition in enumerate(np.arange(0., 2000. * 5, 1).reshape((2000, 5))):
        meta_buffer.update_buffer(transition=transition)

    for i in np.arange(2000 * 5, (2000 + 10) * 5, 1).reshape((10, 5)):
        print(meta_buffer.sample)

