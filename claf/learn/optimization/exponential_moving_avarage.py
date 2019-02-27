class EMA:
    """
    Exponential Moving Average

    the moving averages of all weights of the model are maintained
        with the exponential decay rate of {ema}.

    * Args:
        model: for model's parameters
        mu: decay rate
    """

    def __init__(self, model, mu):
        self.mu = mu
        self.shadow = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.register(name, param.data)

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
