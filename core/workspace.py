import importlib
import os
import sys
from absl import logging


class Register:
    """
    Module register
    """
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""
        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            return add(None, target)
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()


class Registers:
    """
    All module registers.
    """
    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    task = Register('task')
    model = Register('model')
    solver = Register('solver')
    loss = Register('loss')
    metric = Register('metric')
    preparer = Register('preparer')
    preprocessor = Register('preprocessor')
    postprocess = Register('postprocess')
    serving = Register('serving')
    dataset = Register('dataset')
    transform = Register('transform')