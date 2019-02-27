class Singleton(type):
    """
    Design Pattern Base

    Singleton Meta Class
    the singleton pattern is a software design pattern that restricts the
    instantiation of a class to one object.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
