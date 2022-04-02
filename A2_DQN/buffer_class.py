import numpy as np
from sys import version_info


class MetaBuffer(object):
    """
    Meta CLass for buffers
    """
    n_args = 5  # s(t), a(t), r(t), s(t+1), done(t+1)
    # TODO: find better name
    n_actions_depth = 4
    dtype = "float32"  # might want to change this for discretization

    def __init__(self, depth=int(1e5), sample_batch_length=int(1e4), **kwargs):
        super(MetaBuffer).__init__()

        self._rng = np.random.default_rng()

        # meta selection and self-knowledge
        self._step = 0
        self.depth = depth
        self._buffer_idx = np.arange(depth)
        # we want the batches we draw to be sufficiently random
        _max_batch_fill = 0.2

        # buffer
        # s(t), a(t), r(t), s(t+1), done(t+1)
        try:
            assert sample_batch_length < _max_batch_fill * depth, f"Batch length must be smaller than buffer depth."
        except AssertionError:
            sample_batch_length = int(_max_batch_fill * depth)
            print(f"Downsizing batch length to {sample_batch_length} (10% of depth)")
        self._sample_batch_length = sample_batch_length  # TODO: random value right now, decide on reasonable

        self._buffer = np.zeros(shape=(depth,
                                       self.__class__.n_args,
                                       self.__class__.n_actions_depth), dtype=self.__class__.dtype)
        self._transition = np.zeros(shape=(self.__class__.n_args,
                                           self.__class__.n_actions_depth), dtype=self.__class__.dtype)
        self._transition_length = np.prod(self._transition.shape)

    def __repr__(self):
        return '<%s.%s MetaBuffer object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

    def update_buffer(self, transition):
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

        np.put(a=self._buffer,
               ind=np.arange(self._step * self._transition_length,
                             self._step * self._transition_length + self._transition_length),
               v=self._transition, mode="wrap")
        self._step += 1

    def __reset_all__(self):
        self._step = 0
        self._buffer = np.zeros(shape=self._buffer.shape,
                                dtype=self.__class__.dtype)

    @property
    def sample(self):
        # this is literally killing me inside
        idxs = self._rng.choice(self._buffer_idx, self._sample_batch_length, replace=False, )
        return self._buffer[idxs, 0, :],\
               self._buffer[idxs, 1, ::self.__class__.n_args],\
               self._buffer[idxs, 2, ::self.__class__.n_args],\
               self._buffer[idxs, 3, :],\
               self._buffer[idxs, 4, ::self.__class__.n_args],


class PrioBuffer(MetaBuffer):
    """
    CLass for prioritization buffer.
    Adds prioritization for each buffer sample that are used to draw greedy/exploratory samples.
    """
    n_args = 6  # s(t), a(t), r(t), s(t+1), done(t+1), priority
    # TODO: find better name
    n_actions_depth = 4
    dtype = "float32"  # might want to change this for discretization

    def __init__(self, depth=int(1e5), sample_batch_length=int(1e4), prio_temp=0.75, min_prio=0.01, **kwargs):
        super(PrioBuffer).__init__(depth, sample_batch_length, **kwargs)

        self.prio_temp = prio_temp  # annealing (together with epsilon?)
        self.min_prio = min_prio

        self._buffer[:5] = np.clip(self._buffer[:5], a_min=self.min_prio, a_max=1.)


    def update_buffer(self, transition):
        """
        Updates the replay buffer with the transition.
        The transition is a tuple or np.array of (s(t), a(t), r(t), s(t+1), done(t+1))
        If the transition is a tuple it will be implicitly be cast to a np.array with type self.__class__.dtype

        :param transition: a tuple of s(t), a(t), r(t), s(t+1), done(t+1), expected_reward
        """

        # TODO: verify initial function with all the same values
        transition[5] = np.clip(np.power(transition[5],
                                         self.prio_temp) / np.sum(np.power(self._buffer[:, 5],
                                                                           self.prio_temp)),
                                a_min=self.min_prio, a_max=1.)

        # TODO: avoid packing in the function calling the update, *args tuple or implicit to-tuple cast?
        # TODO: should instead cast to right shape, write @staticmethod?
        for i, column in enumerate(transition):
            self._transition[i, :] = column

        np.put(a=self._buffer,
               ind=np.arange(self._step * self._transition_length,
                             self._step * self._transition_length + self._transition_length),
               v=self._transition, mode="wrap")
        self._step += 1

    @property
    def sample(self):
        return self._rng.choice(self._buffer[:, :5],
                                self._sample_batch_length,
                                replace=False,
                                p=self._buffer[5])


if __name__ == "__main__":
    sample_batch_length = 10
    meta_buffer = MetaBuffer(depth=1000, sample_batch_length=sample_batch_length)

    for i, transition in enumerate(np.arange(0., 2000. * 5 * 4, 1).reshape((2000, 5, 4))):
        meta_buffer.update_buffer(transition=transition)

    n_batches = 2
    batches = np.zeros(shape=(n_batches, sample_batch_length, 5, 4))

    for i, new_batch in enumerate(np.arange(2000 * 5 * 4, (2000 + 2) * 5 * 4, 1).reshape((2, 5, 4))):
        sample = meta_buffer.sample
        print("Drawing sample of shape: ",  sample.shape)
        batches[i] = sample

    assert ~np.all(batches[0] == batches[1]), "Batches should not be the same at all positions."
    print("Passed.")
