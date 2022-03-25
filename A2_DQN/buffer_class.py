import numpy as np
from sys import version_info

class MetaBuffer(np.ndarray):
    """
    Meta CLass for buffers
    """
    n_args = 5  # s(t), a(t), r(t), s(t+1), done(t+1)
    dtype = np.float64  # might want to change this for discretization

    # from np documentation for subclassing np.ndarrays
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None,):
        obj = super().__new__(subtype, shape, dtype,
                              buffer, offset, strides, order)

        obj.rng = None
        obj._step = None
        obj.buffer_selecter = None

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.rng = getattr(obj, 'rng', None)
        self._step = getattr(obj, '_step', None)
        self.buffer_selecter = getattr(obj, 'buffer_selecter', None)


    def __init__(self, depth, *kwargs):
        super(MetaBuffer, self).__init__(shape=(depth, self.__class__.n_args),
                                         dtype=self.__class__.dtype)
        self.rng = np.random.default_rng()

        # meta selection and self-knowledge
        self._step = 0
        self.buffer_selecter = lambda **buffer_kwargs: None

        # buffer
        # s(t), a(t), r(t), s(t+1), done(t+1)
        self._buffer = np.empty(shape=(depth, 5), dtype=self.__class__.dtype)

    def __repr__(self):
        return '<%s.%s MetaBuffer object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

    def update_buffer(self, transition=np.empty(shape=(1, 5))):
        """
        Updates the replay buffer with the transition.
        The transition is a tuple or np.array of (s(t), a(t), r(t), s(t+1), done(t+1))
        If the transition is a tuple it will be implicitly be cast to a np.array with type self.__class__.dtype

        :param transition: a tuple of s(t), a(t), r(t), s(t+1), done(t+1)
        """
        np.put(self._buffer, ind=self._step, v=transition)


    @classmethod
    def __reset_all__(cls):
        pass

    @property
    def get_buffered(self):
        return self.buffer_selecter()