
import argparse


class NestedNamespace(argparse.Namespace):
    """
    Nested Namespace
    (Simple class used by default by parse_args() to create
     an object holding attributes and return it.)
    """

    def __setattr__(self, name, value):
        if "." in name:
            group, name = name.split(".", 1)
            namespace = getattr(self, group, NestedNamespace())
            setattr(namespace, name, value)
            self.__dict__[group] = namespace
        else:
            self.__dict__[name] = value

    def delete_unselected(self, namespace, excepts=[]):
        delete_keys = []
        for key in namespace.__dict__:
            if key not in excepts:
                delete_keys.append(key)

        for key in delete_keys:
            delattr(namespace, key)

    def overwrite(self, config):
        def _overwrite(namespace, d):
            for k, v in d.items():
                if type(v) == dict:
                    nested_namespace = getattr(namespace, k, None)
                    if nested_namespace is None:
                        nested_namespace = NestedNamespace()
                        nested_namespace.load_from_json(v)

                        setattr(namespace, k, nested_namespace)
                    else:
                        _overwrite(nested_namespace, v)
                else:
                    setattr(namespace, k, v)
            return namespace

        return _overwrite(self, config)

    def load_from_json(self, dict_data):

        name_value_pairs = []

        def make_key_value_pairs(d, prefix=""):
            for k, v in d.items():
                if type(v) == dict:
                    next_prefix = k
                    if prefix != "":
                        next_prefix = f"{prefix}.{k}"
                    make_key_value_pairs(v, prefix=next_prefix)
                else:
                    key_with_prefix = k
                    if prefix != "":
                        key_with_prefix = f"{prefix}.{k}"
                    name_value_pairs.append((key_with_prefix, v))

        make_key_value_pairs(dict_data)
        for (name, value) in name_value_pairs:
            self.__setattr__(name, value)
