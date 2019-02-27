class PassText:
    """
    Pass text without tokenize
    """

    def __init__(self):
        self.name = "pass"
        self.cache_name = "pass"

    def tokenize(self, text):
        return text
