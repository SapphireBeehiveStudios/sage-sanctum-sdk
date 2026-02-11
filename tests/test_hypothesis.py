"""Property-based tests using Hypothesis."""

from hypothesis import given, settings
from hypothesis import strategies as st

from sage_sanctum.auth.trat import AllowedModels
from sage_sanctum.io.outputs import Finding, Location, SarifOutput
from sage_sanctum.llm.model_category import ModelCategory
from sage_sanctum.llm.model_ref import ModelRef
from sage_sanctum.llm.model_selector import ModelSelector


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# provider:model ref strings — provider is lowercase, neither part is empty or
# contains colons/whitespace (matching real-world model reference strings).
_provider_st = st.from_regex(r"[a-z][a-z0-9\-]{0,19}", fullmatch=True)
_model_name_st = st.from_regex(r"[a-zA-Z0-9][a-zA-Z0-9\-\.]{0,39}", fullmatch=True)

_model_ref_st = st.builds(ModelRef, provider=_provider_st, model=_model_name_st)

_severity_st = st.sampled_from(["critical", "high", "medium", "low", "note"])

_location_st = st.builds(
    Location,
    file=st.from_regex(r"[a-z][a-z0-9_/]{0,30}\.[a-z]{1,4}", fullmatch=True),
    start_line=st.integers(min_value=1, max_value=100_000),
    end_line=st.integers(min_value=0, max_value=100_000),
    start_column=st.integers(min_value=0, max_value=500),
    end_column=st.integers(min_value=0, max_value=500),
)

_finding_st = st.builds(
    Finding,
    id=st.from_regex(r"[A-Z]{2,5}-[0-9]{1,4}", fullmatch=True),
    title=st.text(min_size=1, max_size=80),
    description=st.text(min_size=1, max_size=200),
    severity=_severity_st,
    location=_location_st,
    cwe=st.from_regex(r"(CWE-)?[0-9]{1,4}", fullmatch=True) | st.just(""),
    remediation=st.text(max_size=100),
    confidence=st.sampled_from(["high", "medium", "low"]),
)

_model_ref_str_st = st.tuples(_provider_st, _model_name_st).map(
    lambda t: f"{t[0]}:{t[1]}"
)

_allowed_models_st = st.builds(
    AllowedModels,
    triage=st.lists(_model_ref_str_st, max_size=5),
    analysis=st.lists(_model_ref_str_st, max_size=5),
    reasoning=st.lists(_model_ref_str_st, max_size=5),
    embeddings=st.lists(_model_ref_str_st, max_size=5),
)


# ---------------------------------------------------------------------------
# Property 1: ModelRef.parse ↔ str round-trip
# ---------------------------------------------------------------------------


class TestModelRefRoundTrip:
    @given(ref=_model_ref_st)
    def test_str_then_parse_is_identity(self, ref: ModelRef):
        """str(ref) -> parse -> should yield the same ModelRef."""
        assert ModelRef.parse(str(ref)) == ref

    @given(ref=_model_ref_st)
    def test_parse_is_idempotent(self, ref: ModelRef):
        """Parsing the string form twice yields the same result."""
        s = str(ref)
        assert ModelRef.parse(s) == ModelRef.parse(str(ModelRef.parse(s)))

    @given(ref=_model_ref_st)
    def test_str_format(self, ref: ModelRef):
        """str(ref) is always 'provider:model'."""
        s = str(ref)
        assert ":" in s
        provider, model = s.split(":", 1)
        assert provider == ref.provider
        assert model == ref.model


# ---------------------------------------------------------------------------
# Property 2: SARIF output structural invariants
# ---------------------------------------------------------------------------


class TestSarifStructuralInvariants:
    @given(findings=st.lists(_finding_st, max_size=20))
    @settings(max_examples=50)
    def test_result_count_equals_finding_count(self, findings: list[Finding]):
        """SARIF results list has exactly as many entries as findings."""
        output = SarifOutput(
            tool_name="test", tool_version="1.0", findings=findings
        )
        sarif = output.to_dict()
        results = sarif["runs"][0]["results"]
        assert len(results) == len(findings)

    @given(findings=st.lists(_finding_st, min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_every_unique_id_appears_once_in_rules(self, findings: list[Finding]):
        """Each unique finding ID maps to exactly one rule."""
        output = SarifOutput(
            tool_name="test", tool_version="1.0", findings=findings
        )
        sarif = output.to_dict()
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = [r["id"] for r in rules]

        unique_finding_ids = list(dict.fromkeys(f.id for f in findings))
        assert rule_ids == unique_finding_ids

    @given(findings=st.lists(_finding_st, min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_all_results_reference_existing_rules(self, findings: list[Finding]):
        """Every result's ruleId is present in the rules list."""
        output = SarifOutput(
            tool_name="test", tool_version="1.0", findings=findings
        )
        sarif = output.to_dict()
        rule_ids = {r["id"] for r in sarif["runs"][0]["tool"]["driver"]["rules"]}
        for result in sarif["runs"][0]["results"]:
            assert result["ruleId"] in rule_ids

    @given(findings=st.lists(_finding_st, max_size=20))
    @settings(max_examples=50)
    def test_sarif_level_is_always_valid(self, findings: list[Finding]):
        """Every SARIF result level is one of the valid SARIF values."""
        valid_levels = {"error", "warning", "note", "none"}
        output = SarifOutput(
            tool_name="test", tool_version="1.0", findings=findings
        )
        sarif = output.to_dict()
        for result in sarif["runs"][0]["results"]:
            assert result["level"] in valid_levels

    @given(findings=st.lists(_finding_st, max_size=20))
    @settings(max_examples=50)
    def test_start_line_is_at_least_one(self, findings: list[Finding]):
        """SARIF startLine is always >= 1 even if finding has start_line=0."""
        output = SarifOutput(
            tool_name="test", tool_version="1.0", findings=findings
        )
        sarif = output.to_dict()
        for result in sarif["runs"][0]["results"]:
            region = result["locations"][0]["physicalLocation"]["region"]
            assert region["startLine"] >= 1


# ---------------------------------------------------------------------------
# Property 3: ModelSelector select/is_allowed/select_all consistency
# ---------------------------------------------------------------------------


class TestModelSelectorConsistency:
    @given(
        model_refs=st.lists(_model_ref_str_st, min_size=1, max_size=5),
        category=st.sampled_from(list(ModelCategory)),
    )
    def test_select_is_allowed(self, model_refs: list[str], category: ModelCategory):
        """The selected model is always allowed for its category."""
        selector = ModelSelector({category.value: model_refs})
        selected = selector.select(category)
        assert selector.is_allowed(selected, category)

    @given(
        model_refs=st.lists(_model_ref_str_st, min_size=1, max_size=5),
        category=st.sampled_from(list(ModelCategory)),
    )
    def test_select_is_first_of_select_all(
        self, model_refs: list[str], category: ModelCategory
    ):
        """select() returns the first element of select_all()."""
        selector = ModelSelector({category.value: model_refs})
        assert selector.select(category) == selector.select_all(category)[0]

    @given(
        model_refs=st.lists(_model_ref_str_st, min_size=1, max_size=5),
        category=st.sampled_from(list(ModelCategory)),
    )
    def test_select_all_returns_defensive_copy(
        self, model_refs: list[str], category: ModelCategory
    ):
        """select_all() returns a new list each time (defensive copy)."""
        selector = ModelSelector({category.value: model_refs})
        list1 = selector.select_all(category)
        list2 = selector.select_all(category)
        assert list1 == list2
        assert list1 is not list2

    @given(
        model_refs=st.lists(_model_ref_str_st, min_size=1, max_size=5),
        category=st.sampled_from(list(ModelCategory)),
    )
    def test_validate_model_does_not_raise_for_selected(
        self, model_refs: list[str], category: ModelCategory
    ):
        """validate_model() never raises for a model returned by select()."""
        selector = ModelSelector({category.value: model_refs})
        selected = selector.select(category)
        selector.validate_model(selected, category)  # should not raise


# ---------------------------------------------------------------------------
# Property 4: AllowedModels from_dict ↔ to_dict round-trip
# ---------------------------------------------------------------------------


class TestAllowedModelsRoundTrip:
    @given(am=_allowed_models_st)
    def test_from_dict_to_dict_round_trip(self, am: AllowedModels):
        """from_dict(am.to_dict()) reconstructs an equal AllowedModels."""
        assert AllowedModels.from_dict(am.to_dict()) == am

    @given(am=_allowed_models_st)
    def test_to_dict_has_all_categories(self, am: AllowedModels):
        """to_dict() always includes all four category keys."""
        d = am.to_dict()
        assert set(d.keys()) == {"triage", "analysis", "reasoning", "embeddings"}
