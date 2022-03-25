import numpy as np
from sys import version_info

class MetaBuffer(object):
    """
    Meta CLass for buffers
    """

    if version_info < (3, 9):
        raise EnvironmentError("Please update to Python 3.9 or later (code written for Python 3.9.x).\n"
                               "This is required for the proper function of MetaClasses\n"
                               "which use the double  @classmethod @property decorator.\n"
                               "We use this, pythonic approach, to keep track of our particles.")

    def __init__(self, *kwargs):
        super(MetaBuffer, self).__init__()
        self.rng = np.random.default_rng()

        # meta selection and self-knowledge
        self._now_ = 0
        self.buffer_selecter = lambda **buffer_kwargs: None

    def __repr__(self):
        return '<%s.%s MetaBuffer object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

    def get_current_buffer(self):
        return self.buffer_selecter()

    @classmethod
    def __reset_all__(cls):
        pass

    @property
    def buffer(self):
        return self.get_current_buffer