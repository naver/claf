class arguments_required:
    """
        Decorator Class
        check required arguments for predict function
        (eg. @arguments_required(["db_path", "table_id"]))
    """

    def __init__(self, required_fields):
        self.required_fields = required_fields

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            arguments = args[2]
            for item in self.required_fields:
                if arguments.get(item, None) is None:
                    raise ValueError(f"--{item} is required argument.")
            return fn(*args, **kwargs)

        return wrapper
