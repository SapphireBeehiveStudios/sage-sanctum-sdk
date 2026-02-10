"""Tests for ModelSelector and StaticModelSelector."""

import pytest

from sage_sanctum.errors import ModelNotAuthorizedError, ModelNotAvailableError
from sage_sanctum.llm.model_category import ModelCategory
from sage_sanctum.llm.model_ref import ModelRef
from sage_sanctum.llm.model_selector import ModelSelector, StaticModelSelector


class TestModelSelector:
    @pytest.fixture
    def allowed_models(self):
        return {
            "triage": ["openai:gpt-4o-mini"],
            "analysis": ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-latest"],
            "reasoning": ["openai:o1"],
            "embeddings": ["openai:text-embedding-3-small"],
        }

    @pytest.fixture
    def selector(self, allowed_models):
        return ModelSelector(allowed_models)

    def test_select_triage(self, selector):
        ref = selector.select(ModelCategory.TRIAGE)
        assert ref == ModelRef(provider="openai", model="gpt-4o-mini")

    def test_select_analysis(self, selector):
        ref = selector.select(ModelCategory.ANALYSIS)
        assert ref == ModelRef(provider="openai", model="gpt-4o")

    def test_select_reasoning(self, selector):
        ref = selector.select(ModelCategory.REASONING)
        assert ref == ModelRef(provider="openai", model="o1")

    def test_select_all_analysis(self, selector):
        models = selector.select_all(ModelCategory.ANALYSIS)
        assert len(models) == 2
        assert models[0] == ModelRef(provider="openai", model="gpt-4o")
        assert models[1] == ModelRef(provider="anthropic", model="claude-3-5-sonnet-latest")

    def test_select_empty_category_raises(self):
        selector = ModelSelector({"triage": ["openai:gpt-4o-mini"]})
        with pytest.raises(ModelNotAvailableError):
            selector.select(ModelCategory.REASONING)

    def test_is_allowed(self, selector):
        model = ModelRef(provider="openai", model="gpt-4o")
        assert selector.is_allowed(model, ModelCategory.ANALYSIS)
        assert not selector.is_allowed(model, ModelCategory.TRIAGE)

    def test_validate_model_passes(self, selector):
        model = ModelRef(provider="openai", model="gpt-4o")
        selector.validate_model(model, ModelCategory.ANALYSIS)

    def test_validate_model_raises(self, selector):
        model = ModelRef(provider="openai", model="gpt-4o")
        with pytest.raises(ModelNotAuthorizedError):
            selector.validate_model(model, ModelCategory.TRIAGE)


class TestStaticModelSelector:
    def test_returns_same_model_for_all_categories(self):
        selector = StaticModelSelector("gpt-4o")
        for category in ModelCategory:
            ref = selector.select(category)
            assert ref == ModelRef(provider="openai", model="gpt-4o")

    def test_accepts_model_ref(self):
        model = ModelRef(provider="anthropic", model="claude-3-5-sonnet")
        selector = StaticModelSelector(model)
        ref = selector.select(ModelCategory.ANALYSIS)
        assert ref == model

    def test_is_allowed_for_any_category(self):
        selector = StaticModelSelector("gpt-4o")
        model = ModelRef(provider="openai", model="gpt-4o")
        for category in ModelCategory:
            assert selector.is_allowed(model, category)
