"""Tests for ModelRef parsing and formatting."""

import pytest

from sage_sanctum.errors import ModelRefParseError
from sage_sanctum.llm.model_ref import ModelRef


class TestModelRefParse:
    def test_explicit_provider(self):
        ref = ModelRef.parse("openai:gpt-4o")
        assert ref.provider == "openai"
        assert ref.model == "gpt-4o"

    def test_anthropic_explicit(self):
        ref = ModelRef.parse("anthropic:claude-3-5-sonnet-latest")
        assert ref.provider == "anthropic"
        assert ref.model == "claude-3-5-sonnet-latest"

    def test_google_explicit(self):
        ref = ModelRef.parse("google:gemini-2.0-flash")
        assert ref.provider == "google"
        assert ref.model == "gemini-2.0-flash"

    def test_auto_detect_openai(self):
        ref = ModelRef.parse("gpt-4o")
        assert ref.provider == "openai"
        assert ref.model == "gpt-4o"

    def test_auto_detect_anthropic(self):
        ref = ModelRef.parse("claude-3-5-sonnet-latest")
        assert ref.provider == "anthropic"
        assert ref.model == "claude-3-5-sonnet-latest"

    def test_auto_detect_google(self):
        ref = ModelRef.parse("gemini-2.0-flash")
        assert ref.provider == "google"
        assert ref.model == "gemini-2.0-flash"

    def test_auto_detect_o1(self):
        ref = ModelRef.parse("o1")
        assert ref.provider == "openai"
        assert ref.model == "o1"

    def test_empty_string_raises(self):
        with pytest.raises(ModelRefParseError):
            ModelRef.parse("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ModelRefParseError):
            ModelRef.parse("   ")

    def test_empty_provider_raises(self):
        with pytest.raises(ModelRefParseError):
            ModelRef.parse(":gpt-4o")

    def test_empty_model_raises(self):
        with pytest.raises(ModelRefParseError):
            ModelRef.parse("openai:")

    def test_strips_whitespace(self):
        ref = ModelRef.parse("  openai:gpt-4o  ")
        assert ref.provider == "openai"
        assert ref.model == "gpt-4o"


class TestModelRefForLitellm:
    def test_openai_no_prefix(self):
        ref = ModelRef(provider="openai", model="gpt-4o")
        assert ref.for_litellm == "gpt-4o"

    def test_anthropic_prefix(self):
        ref = ModelRef(provider="anthropic", model="claude-3-5-sonnet-latest")
        assert ref.for_litellm == "anthropic/claude-3-5-sonnet-latest"

    def test_anthropic_already_prefixed(self):
        ref = ModelRef(provider="anthropic", model="anthropic/claude-3-5-sonnet")
        assert ref.for_litellm == "anthropic/claude-3-5-sonnet"

    def test_google_prefix(self):
        ref = ModelRef(provider="google", model="gemini-2.0-flash")
        assert ref.for_litellm == "gemini/gemini-2.0-flash"

    def test_google_already_prefixed(self):
        ref = ModelRef(provider="google", model="gemini/gemini-2.0-flash")
        assert ref.for_litellm == "gemini/gemini-2.0-flash"


class TestModelRefStr:
    def test_str(self):
        ref = ModelRef(provider="openai", model="gpt-4o")
        assert str(ref) == "openai:gpt-4o"

    def test_repr(self):
        ref = ModelRef(provider="openai", model="gpt-4o")
        assert "openai" in repr(ref)
        assert "gpt-4o" in repr(ref)


class TestModelRefFrozen:
    def test_immutable(self):
        ref = ModelRef(provider="openai", model="gpt-4o")
        with pytest.raises(AttributeError):
            ref.provider = "anthropic"

    def test_hashable(self):
        ref1 = ModelRef(provider="openai", model="gpt-4o")
        ref2 = ModelRef(provider="openai", model="gpt-4o")
        assert ref1 == ref2
        assert hash(ref1) == hash(ref2)
        assert len({ref1, ref2}) == 1
