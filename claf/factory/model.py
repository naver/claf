
from overrides import overrides

from claf.config.registry import Registry
from claf.model.base import ModelWithTokenEmbedder, ModelWithoutTokenEmbedder
from claf.model.reading_comprehension.mixin import ReadingComprehension
from claf.tokens import token_embedder

from .base import Factory


class ModelFactory(Factory):
    """
    Model Factory Class

    Create Concrete model according to config.model_name
    Get model from model registries (eg. @register("model:{model_name}"))

    * Args:
        config: model config from argument (config.model)
    """

    def __init__(self):
        self.registry = Registry()

    @overrides
    def create(self, config, token_makers, **params):
        name = config.name
        model_config = {}
        if getattr(config, config.name, None):
            model_config = vars(getattr(config, config.name))

        model = self.registry.get(f"model:{name}")

        if issubclass(model, ModelWithTokenEmbedder):
            token_embedder = self.create_token_embedder(model, token_makers)
            model_config["token_embedder"] = token_embedder
        elif issubclass(model, ModelWithoutTokenEmbedder):
            model_config["token_makers"] = token_makers
        else:
            raise ValueError(
                "Model must have inheritance. (ModelWithTokenEmbedder or ModelWithoutTokenEmbedder)"
            )
        return model(**model_config, **params)

    def create_token_embedder(self, model, token_makers):
        # 1. Specific case
        # ...

        # 2. Base case
        if issubclass(model, ReadingComprehension):
            return token_embedder.RCTokenEmbedder(token_makers)
        else:
            return token_embedder.BasicTokenEmbedder(token_makers)
