"""Model selection based on TraT allowed_models."""

from __future__ import annotations

from ..errors import ModelNotAuthorizedError, ModelNotAvailableError
from .model_category import ModelCategory
from .model_ref import ModelRef


class ModelSelector:
    """Selects models based on TraT allowed_models configuration.

    The TraT's tctx.allowed_models contains lists of ModelRef strings per category.
    This selector returns the first (preferred) model for a given category.
    """

    def __init__(self, allowed_models: dict[str, list[str]]) -> None:
        self._allowed: dict[str, list[ModelRef]] = {}
        for category, refs in allowed_models.items():
            self._allowed[category] = [ModelRef.parse(r) for r in refs]

    def select(self, category: ModelCategory) -> ModelRef:
        """Select the preferred model for the given category.

        Returns the first model in the allowed list for the category.

        Raises:
            ModelNotAvailableError: If no models are configured for the category.
        """
        models = self._allowed.get(category.value, [])
        if not models:
            raise ModelNotAvailableError(
                f"No models configured for category {category.value!r}"
            )
        return models[0]

    def select_all(self, category: ModelCategory) -> list[ModelRef]:
        """Return all allowed models for the given category."""
        return list(self._allowed.get(category.value, []))

    def is_allowed(self, model: ModelRef, category: ModelCategory) -> bool:
        """Check if a specific model is allowed for a category."""
        return model in self._allowed.get(category.value, [])

    def validate_model(self, model: ModelRef, category: ModelCategory) -> None:
        """Validate that a model is allowed for the category.

        Raises:
            ModelNotAuthorizedError: If the model is not in allowed_models.
        """
        if not self.is_allowed(model, category):
            raise ModelNotAuthorizedError(
                f"Model {model} not authorized for category {category.value!r}. "
                f"Allowed: {self.select_all(category)}"
            )


class StaticModelSelector(ModelSelector):
    """Simple selector that returns the same model for all categories.

    Useful for local development where a single model is used.
    """

    def __init__(self, model: str | ModelRef) -> None:
        if isinstance(model, str):
            model = ModelRef.parse(model)
        self._model = model
        # Build allowed_models with this model for all categories
        ref_str = str(model)
        super().__init__({
            cat.value: [ref_str] for cat in ModelCategory
        })

    def select(self, category: ModelCategory) -> ModelRef:
        return self._model
