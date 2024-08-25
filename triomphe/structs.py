from dsl import *

class Feature(object):
    def __init__(self, feature_str):
        self.feature_str = feature_str
        self.feature = None

    def __call__(self, *args, **kwargs):
        if self.feature is None:
            self.feature = eval('lambda x : ' + self.feature_str)
        return self.feature(*args, **kwargs)

    def __repr__(self):
        return self.feature_str

    def __str__(self):
        return self.feature_str

    def __getstate__(self):
        return self.feature_str

    def __setstate__(self, feature_str):
        self.feature_str = feature_str
        self.feature = None

    def __add__(self, s):
        if isinstance(s, str):
            return Feature(self.feature_str + s)
        elif isinstance(s, Feature):
            return Feature(self.feature_str + s.feature_str)
        raise Exception()

    def __radd__(self, s):
        if isinstance(s, str):
            return Feature(s + self.feature_str)
        elif isinstance(s, Feature):
            return Feature(s.feature_str + self.feature_str)
        raise Exception()
