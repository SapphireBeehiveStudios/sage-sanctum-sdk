"""Property-based tests using Hypothesis."""

import base64
import inspect
import json

from hypothesis import given, settings
from hypothesis import strategies as st

from sage_sanctum.auth.spiffe import _decode_jwt_payload
from sage_sanctum.auth.trat import AllowedModels
from sage_sanctum.errors import SageSanctumError
from sage_sanctum.gateway.http import GatewayHttpClient
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


# ---------------------------------------------------------------------------
# Property 5: _parse_response round-trip
# ---------------------------------------------------------------------------

# Strategy: generate valid HTTP response components and build raw bytes,
# then verify _parse_response reconstructs the status, headers, and body.

_http_status_st = st.sampled_from([200, 201, 204, 301, 400, 401, 403, 404, 429, 500, 502, 503])
_http_reason_st = st.sampled_from(["OK", "Created", "Bad Request", "Not Found", "Error"])

# Header names: token chars (no colons/whitespace); values: printable ASCII without \r\n.
_header_name_st = st.from_regex(r"[A-Za-z][A-Za-z0-9\-]{0,20}", fullmatch=True)
_header_value_st = st.from_regex(r"[a-zA-Z0-9 /;=\-\.,_]{1,60}", fullmatch=True)
_header_pair_st = st.tuples(_header_name_st, _header_value_st)

# Body: arbitrary text that doesn't include the header/body separator sequence.
_body_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    max_size=200,
).filter(lambda s: "\r\n\r\n" not in s)


def _build_raw_http(status: int, reason: str, headers: list[tuple[str, str]], body: str) -> bytes:
    """Build a valid raw HTTP/1.1 response with Content-Length."""
    body_bytes = body.encode("utf-8")
    lines = [f"HTTP/1.1 {status} {reason}"]
    lines.extend(f"{k}: {v}" for k, v in headers)
    lines.append(f"Content-Length: {len(body_bytes)}")
    header_block = "\r\n".join(lines) + "\r\n\r\n"
    return header_block.encode("utf-8") + body_bytes


class TestParseResponseRoundTrip:
    """Construct raw HTTP response bytes and verify _parse_response preserves components."""

    @given(
        status=_http_status_st,
        reason=_http_reason_st,
        extra_headers=st.lists(_header_pair_st, max_size=5),
        body=_body_st,
    )
    @settings(max_examples=100)
    def test_status_preserved(self, status, reason, extra_headers, body):
        """Parsed status code matches the one in the raw response."""
        raw = _build_raw_http(status, reason, extra_headers, body)
        client = GatewayHttpClient(host="localhost", port=1)
        resp = client._parse_response(raw)
        assert resp.status == status

    @given(
        status=_http_status_st,
        reason=_http_reason_st,
        extra_headers=st.lists(_header_pair_st, min_size=1, max_size=5, unique_by=lambda h: h[0].lower()),
        body=_body_st,
    )
    @settings(max_examples=100)
    def test_headers_preserved(self, status, reason, extra_headers, body):
        """All extra headers appear in the parsed response (lower-cased keys).

        Uses unique header names because duplicate headers overwrite earlier values.
        """
        raw = _build_raw_http(status, reason, extra_headers, body)
        client = GatewayHttpClient(host="localhost", port=1)
        resp = client._parse_response(raw)
        for name, value in extra_headers:
            assert resp.headers[name.strip().lower()] == value.strip()

    @given(
        status=_http_status_st,
        reason=_http_reason_st,
        body=_body_st,
    )
    @settings(max_examples=100)
    def test_body_preserved(self, status, reason, body):
        """Parsed body data matches the original body string."""
        raw = _build_raw_http(status, reason, [], body)
        client = GatewayHttpClient(host="localhost", port=1)
        resp = client._parse_response(raw)
        assert resp.data == body

    @given(
        status=_http_status_st,
        reason=_http_reason_st,
        extra_headers=st.lists(_header_pair_st, max_size=5),
        body=_body_st,
    )
    @settings(max_examples=50)
    def test_content_length_in_parsed_headers(self, status, reason, extra_headers, body):
        """Content-Length header is always present in parsed output."""
        raw = _build_raw_http(status, reason, extra_headers, body)
        client = GatewayHttpClient(host="localhost", port=1)
        resp = client._parse_response(raw)
        assert "content-length" in resp.headers


# ---------------------------------------------------------------------------
# Property 6: _decode_jwt_payload round-trip
# ---------------------------------------------------------------------------

# Strategy: generate a JSON-serializable dict, encode it as a fake JWT,
# and verify _decode_jwt_payload recovers the exact dict.

_jwt_payload_st = st.fixed_dictionaries({
    "sub": st.text(min_size=1, max_size=30),
    "exp": st.integers(min_value=0, max_value=2**31),
    "iss": st.text(min_size=1, max_size=30),
}).flatmap(
    lambda base: st.fixed_dictionaries({
        **{k: st.just(v) for k, v in base.items()},
        "extra": st.text(max_size=20),
    })
)


def _encode_jwt_segment(data: dict) -> str:
    """Encode a dict as a base64url JWT segment (no padding)."""
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def _make_fake_jwt(payload: dict) -> str:
    """Build a fake 3-segment JWT from a payload dict."""
    header = {"alg": "EdDSA", "typ": "JWT"}
    h = _encode_jwt_segment(header)
    p = _encode_jwt_segment(payload)
    s = base64.urlsafe_b64encode(b"signature").rstrip(b"=").decode()
    return f"{h}.{p}.{s}"


class TestDecodeJwtPayloadRoundTrip:
    """Encode a dict as JWT payload, decode it, verify round-trip."""

    @given(payload=_jwt_payload_st)
    def test_encode_decode_round_trip(self, payload):
        """_decode_jwt_payload recovers the original dict."""
        jwt_str = _make_fake_jwt(payload)
        decoded = _decode_jwt_payload(jwt_str)
        assert decoded == payload

    @given(payload=_jwt_payload_st)
    def test_decode_is_idempotent(self, payload):
        """Decoding the same JWT twice yields identical results."""
        jwt_str = _make_fake_jwt(payload)
        first = _decode_jwt_payload(jwt_str)
        second = _decode_jwt_payload(jwt_str)
        assert first == second

    @given(
        payload=st.fixed_dictionaries({
            "sub": st.text(min_size=1, max_size=10),
            "exp": st.integers(min_value=0, max_value=2**31),
        })
    )
    def test_padding_lengths_all_handled(self, payload):
        """Base64url payloads of various lengths (padding 0-3) decode correctly.

        The base64url encoding may need 0, 1, 2, or 3 padding chars. This test
        verifies the padding restoration logic handles all cases.
        """
        jwt_str = _make_fake_jwt(payload)
        # Verify we stripped padding in the JWT
        assert "=" not in jwt_str.split(".")[1]
        decoded = _decode_jwt_payload(jwt_str)
        assert decoded["sub"] == payload["sub"]
        assert decoded["exp"] == payload["exp"]


# ---------------------------------------------------------------------------
# Property 7: ModelRef.for_litellm prefix invariant
# ---------------------------------------------------------------------------


class TestForLitellmPrefix:
    """for_litellm applies the correct routing prefix for each provider."""

    @given(model=_model_name_st)
    def test_anthropic_gets_anthropic_prefix(self, model):
        """Anthropic models always get 'anthropic/' prefix in for_litellm."""
        ref = ModelRef(provider="anthropic", model=model)
        litellm = ref.for_litellm
        assert litellm.startswith("anthropic/")
        # The model name is preserved within the litellm string
        assert model in litellm

    @given(model=_model_name_st)
    def test_google_gets_gemini_prefix(self, model):
        """Google models always get 'gemini/' prefix in for_litellm."""
        ref = ModelRef(provider="google", model=model)
        litellm = ref.for_litellm
        assert litellm.startswith("gemini/")
        assert model in litellm

    @given(
        provider=st.from_regex(r"[a-z][a-z0-9\-]{0,19}", fullmatch=True).filter(
            lambda p: p not in ("anthropic", "google")
        ),
        model=_model_name_st,
    )
    def test_other_providers_no_prefix(self, provider, model):
        """Non-anthropic, non-google providers return model name unchanged."""
        ref = ModelRef(provider=provider, model=model)
        assert ref.for_litellm == model

    @given(model=_model_name_st)
    def test_anthropic_prefixed_model_not_double_prefixed(self, model):
        """If model already starts with 'anthropic/', for_litellm doesn't double-prefix."""
        ref = ModelRef(provider="anthropic", model=f"anthropic/{model}")
        litellm = ref.for_litellm
        assert not litellm.startswith("anthropic/anthropic/")
        assert litellm == f"anthropic/{model}"

    @given(model=_model_name_st)
    def test_google_prefixed_model_not_double_prefixed(self, model):
        """If model already starts with 'gemini/', for_litellm doesn't double-prefix."""
        ref = ModelRef(provider="google", model=f"gemini/{model}")
        litellm = ref.for_litellm
        assert not litellm.startswith("gemini/gemini/")
        assert litellm == f"gemini/{model}"


# ---------------------------------------------------------------------------
# Property 8: Error exit code uniqueness
# ---------------------------------------------------------------------------


def _collect_error_classes() -> list[type]:
    """Collect all concrete SageSanctumError subclasses."""
    import sage_sanctum.errors as errors_mod

    result = []
    for name in dir(errors_mod):
        obj = getattr(errors_mod, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, SageSanctumError)
            and obj is not SageSanctumError
        ):
            result.append(obj)
    return result


class TestErrorExitCodeUniqueness:
    """Every concrete error subclass has a unique exit_code."""

    def test_all_exit_codes_are_unique(self):
        """No two error classes share the same exit_code."""
        classes = _collect_error_classes()
        # Only check leaf classes (those not subclassed by another in the list)
        leaf_classes = [
            cls for cls in classes
            if not any(issubclass(other, cls) and other is not cls for other in classes)
        ]
        codes = {}
        for cls in leaf_classes:
            code = cls.exit_code
            assert code not in codes, (
                f"{cls.__name__} and {codes[code].__name__} share exit_code={code}"
            )
            codes[code] = cls

    def test_exit_codes_in_documented_ranges(self):
        """Each error class's exit_code falls within its documented range."""
        range_map = {
            "AuthError": range(10, 20),
            "ForbiddenError": range(20, 30),
            "GatewayError": range(30, 40),
            "ValidationError": range(40, 50),
            "OutputError": range(50, 60),
            "ModelError": range(60, 70),
        }
        classes = _collect_error_classes()
        for cls in classes:
            for base_name, valid_range in range_map.items():
                import sage_sanctum.errors as errors_mod
                base_cls = getattr(errors_mod, base_name, None)
                if base_cls and issubclass(cls, base_cls) and cls is not base_cls:
                    assert cls.exit_code in valid_range, (
                        f"{cls.__name__}.exit_code={cls.exit_code} "
                        f"not in {base_name} range {valid_range.start}-{valid_range.stop - 1}"
                    )

    @given(msg=st.text(min_size=1, max_size=50))
    def test_exit_code_preserved_through_instantiation(self, msg):
        """Instantiating any error preserves the class-level exit_code."""
        classes = _collect_error_classes()
        for cls in classes:
            err = cls(msg)
            assert err.exit_code == cls.exit_code
