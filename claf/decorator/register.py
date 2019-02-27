
from claf.config.registry import Registry


class register:
    """
        Decorator Class
        register subclass with decorator.
        (eg. @register("model:bidaf"), @register("reader:squad") )
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, obj):
        registry = Registry()
        registry.add(self.name, obj)
        return obj
