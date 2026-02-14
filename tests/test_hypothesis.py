"""Property-based tests using Hypothesis."""

import base64
import copy
import inspect
import json

from hypothesis import given, settings
from hypothesis import strategies as st

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sage_sanctum.auth.spiffe import _decode_jwt_payload
from sage_sanctum.auth.trat import AllowedModels, TransactionToken
import pytest

from sage_sanctum.errors import ModelNotAuthorizedError, SageSanctumError
from sage_sanctum.gateway.http import GatewayHttpClient
from sage_sanctum.io.outputs import Finding, Location, SarifOutput
from sage_sanctum.llm.gateway_chat import _messages_to_dicts, _strip_schema_extras
from sage_sanctum.llm.model_category import ModelCategory
from sage_sanctum.llm.model_ref import ModelRef
from sage_sanctum.llm.model_selector import ModelSelector, StaticModelSelector
from sage_sanctum.testing.mocks import MockEmbeddings

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


# ---------------------------------------------------------------------------
# Property 9: TransactionToken.from_jwt full claim chain preservation
# ---------------------------------------------------------------------------

# Strategies for generating valid TraT JWT payloads with nested structures.

_trat_allowed_models_dict_st = st.fixed_dictionaries({
    "triage": st.lists(_model_ref_str_st, max_size=3),
    "analysis": st.lists(_model_ref_str_st, max_size=3),
    "reasoning": st.lists(_model_ref_str_st, max_size=3),
    "embeddings": st.lists(_model_ref_str_st, max_size=3),
})

_trat_tctx_st = st.fixed_dictionaries({
    "run_id": st.text(min_size=1, max_size=30),
    "org_id": st.text(min_size=1, max_size=20),
    "repo_url": st.text(max_size=60),
    "agent_type": st.text(max_size=20),
    "agent_mode": st.text(max_size=20),
    "allowed_models": _trat_allowed_models_dict_st,
    "allowed_providers": st.lists(st.text(min_size=1, max_size=10), max_size=3),
})

_trat_rctx_st = st.fixed_dictionaries({
    "trigger": st.text(max_size=20),
    "pr_number": st.integers(min_value=1, max_value=100_000) | st.none(),
    "actor": st.text(max_size=30),
    "source_ip": st.text(max_size=15),
})

_trat_payload_st = st.fixed_dictionaries({
    "txn": st.text(min_size=1, max_size=30),
    "sub": st.text(max_size=30),
    "scope": st.text(max_size=50),
    "req_wl": st.text(max_size=60),
    "iat": st.floats(min_value=0, max_value=2**31, allow_nan=False, allow_infinity=False),
    "exp": st.floats(min_value=0, max_value=2**31, allow_nan=False, allow_infinity=False),
    "aud": st.text(max_size=30),
    "iss": st.text(max_size=30),
    "tctx": _trat_tctx_st,
    "rctx": _trat_rctx_st,
})


class TestTransactionTokenFromJwtPreservation:
    """TransactionToken.from_jwt preserves all claims through the nested deserialization chain."""

    @given(payload=_trat_payload_st)
    @settings(max_examples=50)
    def test_top_level_claims_preserved(self, payload):
        """Top-level JWT claims (txn, sub, scope, etc.) are preserved."""
        jwt_str = _make_fake_jwt(payload)
        token = TransactionToken.from_jwt(jwt_str)

        assert token.txn == payload["txn"]
        assert token.sub == payload["sub"]
        assert token.scope == payload["scope"]
        assert token.req_wl == payload["req_wl"]
        assert token.aud == payload["aud"]
        assert token.iss == payload["iss"]
        assert token.iat == float(payload["iat"])
        assert token.exp == float(payload["exp"])

    @given(payload=_trat_payload_st)
    @settings(max_examples=50)
    def test_tctx_claims_preserved(self, payload):
        """Nested tctx claims are preserved through TransactionContext.from_dict."""
        jwt_str = _make_fake_jwt(payload)
        token = TransactionToken.from_jwt(jwt_str)

        tctx = payload["tctx"]
        assert token.tctx.run_id == tctx["run_id"]
        assert token.tctx.org_id == tctx["org_id"]
        assert token.tctx.repo_url == tctx["repo_url"]
        assert token.tctx.agent_type == tctx["agent_type"]
        assert token.tctx.agent_mode == tctx["agent_mode"]
        assert token.tctx.allowed_providers == tctx["allowed_providers"]

    @given(payload=_trat_payload_st)
    @settings(max_examples=50)
    def test_allowed_models_preserved(self, payload):
        """Deeply nested allowed_models are preserved through AllowedModels.from_dict."""
        jwt_str = _make_fake_jwt(payload)
        token = TransactionToken.from_jwt(jwt_str)

        am = payload["tctx"]["allowed_models"]
        assert token.tctx.allowed_models.triage == am["triage"]
        assert token.tctx.allowed_models.analysis == am["analysis"]
        assert token.tctx.allowed_models.reasoning == am["reasoning"]
        assert token.tctx.allowed_models.embeddings == am["embeddings"]

    @given(payload=_trat_payload_st)
    @settings(max_examples=50)
    def test_rctx_claims_preserved(self, payload):
        """Nested rctx claims are preserved through RequesterContext.from_dict."""
        jwt_str = _make_fake_jwt(payload)
        token = TransactionToken.from_jwt(jwt_str)

        rctx = payload["rctx"]
        assert token.rctx.trigger == rctx["trigger"]
        assert token.rctx.pr_number == rctx["pr_number"]
        assert token.rctx.actor == rctx["actor"]
        assert token.rctx.source_ip == rctx["source_ip"]

    @given(payload=_trat_payload_st)
    @settings(max_examples=50)
    def test_raw_jwt_preserved(self, payload):
        """The raw JWT string is stored verbatim for gateway forwarding."""
        jwt_str = _make_fake_jwt(payload)
        token = TransactionToken.from_jwt(jwt_str)
        assert token.raw == jwt_str


# ---------------------------------------------------------------------------
# Property 10: SARIF conditional field presence
# ---------------------------------------------------------------------------


class TestSarifConditionalFields:
    """Optional SARIF fields are present iff the corresponding source values are set."""

    @given(finding=_finding_st)
    @settings(max_examples=100)
    def test_end_line_presence(self, finding: Finding):
        """endLine appears in SARIF iff location.end_line > 0."""
        output = SarifOutput(tool_name="t", tool_version="1", findings=[finding])
        region = output.to_dict()["runs"][0]["results"][0]["locations"][0][
            "physicalLocation"
        ]["region"]
        if finding.location.end_line > 0:
            assert "endLine" in region
            assert region["endLine"] == finding.location.end_line
        else:
            assert "endLine" not in region

    @given(finding=_finding_st)
    @settings(max_examples=100)
    def test_start_column_presence(self, finding: Finding):
        """startColumn appears in SARIF iff location.start_column > 0."""
        output = SarifOutput(tool_name="t", tool_version="1", findings=[finding])
        region = output.to_dict()["runs"][0]["results"][0]["locations"][0][
            "physicalLocation"
        ]["region"]
        if finding.location.start_column > 0:
            assert "startColumn" in region
            assert region["startColumn"] == finding.location.start_column
        else:
            assert "startColumn" not in region

    @given(finding=_finding_st)
    @settings(max_examples=100)
    def test_end_column_presence(self, finding: Finding):
        """endColumn appears in SARIF iff location.end_column > 0."""
        output = SarifOutput(tool_name="t", tool_version="1", findings=[finding])
        region = output.to_dict()["runs"][0]["results"][0]["locations"][0][
            "physicalLocation"
        ]["region"]
        if finding.location.end_column > 0:
            assert "endColumn" in region
            assert region["endColumn"] == finding.location.end_column
        else:
            assert "endColumn" not in region

    @given(finding=_finding_st)
    @settings(max_examples=100)
    def test_fixes_presence(self, finding: Finding):
        """fixes appears in SARIF result iff remediation is non-empty."""
        output = SarifOutput(tool_name="t", tool_version="1", findings=[finding])
        result = output.to_dict()["runs"][0]["results"][0]
        if finding.remediation:
            assert "fixes" in result
            assert result["fixes"][0]["description"]["text"] == finding.remediation
        else:
            assert "fixes" not in result


# ---------------------------------------------------------------------------
# Property 11: SARIF CWE tag normalization
# ---------------------------------------------------------------------------


class TestSarifCweNormalization:
    """CWE tags in SARIF are always normalized to 'CWE-' prefix."""

    @given(
        finding=_finding_st.filter(lambda f: f.cwe != ""),
    )
    @settings(max_examples=100)
    def test_cwe_tag_always_prefixed(self, finding: Finding):
        """Non-empty CWE always produces a tag starting with 'CWE-'."""
        output = SarifOutput(tool_name="t", tool_version="1", findings=[finding])
        rules = output.to_dict()["runs"][0]["tool"]["driver"]["rules"]
        # Find the rule matching this finding
        rule = next(r for r in rules if r["id"] == finding.id)
        tags = rule["properties"]["tags"]
        cwe_tags = [t for t in tags if t.startswith("CWE-")]
        assert len(cwe_tags) == 1
        assert "security" in tags

    @given(
        cwe_num=st.from_regex(r"[0-9]{1,4}", fullmatch=True),
    )
    @settings(max_examples=50)
    def test_bare_number_gets_prefix(self, cwe_num: str):
        """A bare CWE number like '89' becomes 'CWE-89' in SARIF tags."""
        finding = Finding(
            id="TEST-1", title="t", description="d", severity="high",
            location=Location(file="f.py", start_line=1), cwe=cwe_num,
        )
        output = SarifOutput(tool_name="t", tool_version="1", findings=[finding])
        rule = output.to_dict()["runs"][0]["tool"]["driver"]["rules"][0]
        assert f"CWE-{cwe_num}" in rule["properties"]["tags"]

    @given(
        cwe_num=st.from_regex(r"[0-9]{1,4}", fullmatch=True),
    )
    @settings(max_examples=50)
    def test_prefixed_cwe_not_double_prefixed(self, cwe_num: str):
        """A CWE already prefixed like 'CWE-89' is not double-prefixed to 'CWE-CWE-89'."""
        finding = Finding(
            id="TEST-1", title="t", description="d", severity="high",
            location=Location(file="f.py", start_line=1), cwe=f"CWE-{cwe_num}",
        )
        output = SarifOutput(tool_name="t", tool_version="1", findings=[finding])
        rule = output.to_dict()["runs"][0]["tool"]["driver"]["rules"][0]
        tags = rule["properties"]["tags"]
        assert f"CWE-{cwe_num}" in tags
        assert f"CWE-CWE-{cwe_num}" not in tags

    @given(
        finding=_finding_st.filter(lambda f: f.cwe == ""),
    )
    @settings(max_examples=50)
    def test_empty_cwe_no_tags(self, finding: Finding):
        """Empty CWE produces no tags property on the rule."""
        output = SarifOutput(tool_name="t", tool_version="1", findings=[finding])
        rule = output.to_dict()["runs"][0]["tool"]["driver"]["rules"][0]
        assert "tags" not in rule["properties"]


# ---------------------------------------------------------------------------
# Property 12: _strip_schema_extras idempotence
# ---------------------------------------------------------------------------

# Strategy: generate Pydantic-like JSON schemas with title, default, nested
# objects, arrays, anyOf, and $defs/$ref — then verify _strip_schema_extras
# is idempotent (applying it twice gives the same result as once).

_prop_name_st = st.from_regex(r"[a-z][a-z_]{0,9}", fullmatch=True)

_leaf_schema_st = st.fixed_dictionaries(
    {"type": st.sampled_from(["string", "integer", "number", "boolean"])},
    optional={
        "title": st.text(min_size=1, max_size=15),
        "default": st.sampled_from(["", 0, True, None, "default_val"]),
    },
)

_json_schema_st = st.recursive(
    _leaf_schema_st,
    lambda inner: st.one_of(
        # Object with properties
        st.builds(
            lambda props, title: {
                "type": "object",
                "properties": props,
                **({"title": title} if title else {}),
            },
            props=st.dictionaries(
                keys=_prop_name_st,
                values=inner,
                min_size=1,
                max_size=3,
            ),
            title=st.text(min_size=1, max_size=15) | st.none(),
        ),
        # Array with items
        st.builds(
            lambda items, title: {
                "type": "array",
                "items": items,
                **({"title": title} if title else {}),
            },
            items=inner,
            title=st.text(min_size=1, max_size=15) | st.none(),
        ),
    ),
    max_leaves=8,
)


@st.composite
def _schema_with_defs(draw):
    """Generate a schema that may include $defs and $ref pointers."""
    schema = draw(_json_schema_st)

    # Optionally add $defs + $ref if this is an object schema with properties
    if (
        schema.get("type") == "object"
        and schema.get("properties")
        and draw(st.booleans())
    ):
        props = schema["properties"]
        key = draw(st.sampled_from(sorted(props.keys())))
        ref_name = f"Ref{key.capitalize()}"
        schema.setdefault("$defs", {})[ref_name] = props[key]
        props[key] = {"$ref": f"#/$defs/{ref_name}"}

    # Optionally wrap a property in anyOf
    if (
        schema.get("type") == "object"
        and schema.get("properties")
        and draw(st.booleans())
    ):
        props = schema["properties"]
        key = draw(st.sampled_from(sorted(props.keys())))
        original = props[key]
        if "$ref" not in original:
            props[key] = {"anyOf": [original, {"type": "null"}]}

    return schema


def _walk_schema(schema: dict, visitor) -> None:
    """Walk all nested schema dicts, calling visitor(sub_schema) on each."""
    visitor(schema)
    if "properties" in schema:
        for prop_schema in schema["properties"].values():
            if isinstance(prop_schema, dict):
                _walk_schema(prop_schema, visitor)
    if "items" in schema and isinstance(schema["items"], dict):
        _walk_schema(schema["items"], visitor)
    for keyword in ("anyOf", "allOf", "oneOf"):
        if keyword in schema and isinstance(schema[keyword], list):
            for item in schema[keyword]:
                if isinstance(item, dict):
                    _walk_schema(item, visitor)


class TestStripSchemaExtrasIdempotence:
    """_strip_schema_extras is a normalization function: applying it twice
    yields the same result as applying it once."""

    @given(schema=_schema_with_defs())
    @settings(max_examples=100)
    def test_idempotent(self, schema: dict):
        """f(f(x)) == f(x) for _strip_schema_extras."""
        schema1 = copy.deepcopy(schema)
        _strip_schema_extras(schema1)

        schema2 = copy.deepcopy(schema1)
        _strip_schema_extras(schema2)

        assert schema1 == schema2


# ---------------------------------------------------------------------------
# Property 13: _strip_schema_extras post-conditions
# ---------------------------------------------------------------------------


class TestStripSchemaExtrasPostConditions:
    """After _strip_schema_extras, the schema satisfies OpenAI strict mode requirements."""

    @given(schema=_schema_with_defs())
    @settings(max_examples=100)
    def test_no_title_or_default_remain(self, schema: dict):
        """No 'title' or 'default' keys remain at any nesting level."""
        _strip_schema_extras(schema)
        forbidden = []

        def check(s):
            for key in ("title", "default"):
                if key in s:
                    forbidden.append(key)

        _walk_schema(schema, check)
        assert forbidden == [], f"Found forbidden keys: {forbidden}"

    @given(schema=_schema_with_defs())
    @settings(max_examples=100)
    def test_no_defs_or_refs_remain(self, schema: dict):
        """No '$defs', 'definitions', or '$ref' keys remain."""
        _strip_schema_extras(schema)
        assert "$defs" not in schema
        assert "definitions" not in schema

        found_refs = []

        def check(s):
            if "$ref" in s:
                found_refs.append(s["$ref"])

        _walk_schema(schema, check)
        assert found_refs == [], f"Found unresolved $ref: {found_refs}"

    @given(schema=_schema_with_defs())
    @settings(max_examples=100)
    def test_objects_have_additional_properties_false(self, schema: dict):
        """Every object-typed schema has additionalProperties: false."""
        _strip_schema_extras(schema)
        violations = []

        def check(s):
            if s.get("type") == "object" or "properties" in s:
                if s.get("additionalProperties") is not False:
                    violations.append(s)

        _walk_schema(schema, check)
        assert violations == [], f"Objects missing additionalProperties: false: {violations}"

    @given(schema=_schema_with_defs())
    @settings(max_examples=100)
    def test_objects_have_required_matching_properties(self, schema: dict):
        """Every object schema has 'required' listing all property keys."""
        _strip_schema_extras(schema)
        violations = []

        def check(s):
            if "properties" in s:
                expected = list(s["properties"].keys())
                actual = s.get("required", [])
                if actual != expected:
                    violations.append({"expected": expected, "actual": actual})

        _walk_schema(schema, check)
        assert violations == [], f"Objects with wrong 'required': {violations}"


# ---------------------------------------------------------------------------
# Property 14: StaticModelSelector category-independence
# ---------------------------------------------------------------------------


class TestStaticModelSelectorCategoryIndependence:
    """StaticModelSelector returns the same model for every category."""

    @given(ref=_model_ref_st)
    def test_select_returns_same_model_for_all_categories(self, ref: ModelRef):
        """select() returns the exact same ModelRef for every ModelCategory."""
        selector = StaticModelSelector(ref)
        for category in ModelCategory:
            assert selector.select(category) == ref

    @given(ref=_model_ref_st)
    def test_is_allowed_for_all_categories(self, ref: ModelRef):
        """is_allowed() returns True for the static model in every category."""
        selector = StaticModelSelector(ref)
        for category in ModelCategory:
            assert selector.is_allowed(ref, category)

    @given(ref=_model_ref_st)
    def test_validate_model_never_raises(self, ref: ModelRef):
        """validate_model() never raises for the static model in any category."""
        selector = StaticModelSelector(ref)
        for category in ModelCategory:
            selector.validate_model(ref, category)  # should not raise

    @given(
        ref=_model_ref_st,
        category=st.sampled_from(list(ModelCategory)),
    )
    def test_select_all_contains_static_model(self, ref: ModelRef, category: ModelCategory):
        """select_all() includes the static model for every category."""
        selector = StaticModelSelector(ref)
        all_models = selector.select_all(category)
        assert ref in all_models

    @given(ref_str=_model_ref_str_st)
    def test_string_initialization_equivalent(self, ref_str: str):
        """Initializing from a string produces the same behavior as from a ModelRef."""
        selector_str = StaticModelSelector(ref_str)
        selector_ref = StaticModelSelector(ModelRef.parse(ref_str))
        for category in ModelCategory:
            assert selector_str.select(category) == selector_ref.select(category)


# ---------------------------------------------------------------------------
# Property 15: _is_response_complete Content-Length boundary correctness
# ---------------------------------------------------------------------------

# Uses _build_raw_http (from Property 5) which always includes Content-Length.

_nonempty_body_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=200,
).filter(lambda s: "\r\n\r\n" not in s)


class TestIsResponseCompleteBoundary:
    """_is_response_complete returns True iff enough bytes for Content-Length."""

    @given(
        status=_http_status_st,
        reason=_http_reason_st,
        body=_body_st,
    )
    @settings(max_examples=100)
    def test_complete_response_detected(self, status, reason, body):
        """A complete response with matching Content-Length is detected."""
        raw = _build_raw_http(status, reason, [], body)
        client = GatewayHttpClient(host="localhost", port=1)
        assert client._is_response_complete(raw)

    @given(
        status=_http_status_st,
        reason=_http_reason_st,
        body=_nonempty_body_st,
    )
    @settings(max_examples=100)
    def test_truncated_body_not_complete(self, status, reason, body):
        """Removing 1 byte from a non-empty body makes it incomplete."""
        raw = _build_raw_http(status, reason, [], body)
        truncated = raw[:-1]
        client = GatewayHttpClient(host="localhost", port=1)
        assert not client._is_response_complete(truncated)

    @given(
        status=_http_status_st,
        reason=_http_reason_st,
        body=_body_st,
        extra=st.binary(min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_appending_data_keeps_complete(self, status, reason, body, extra):
        """Once complete, appending data keeps it complete (monotonicity)."""
        raw = _build_raw_http(status, reason, [], body)
        client = GatewayHttpClient(host="localhost", port=1)
        assert client._is_response_complete(raw + extra)

    def test_headers_only_not_complete(self):
        """Response with only headers (no body separator) is incomplete."""
        raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5"
        client = GatewayHttpClient(host="localhost", port=1)
        assert not client._is_response_complete(raw)


# ---------------------------------------------------------------------------
# Property 16: _messages_to_dicts content and length preservation
# ---------------------------------------------------------------------------

_message_st = st.one_of(
    st.builds(SystemMessage, content=st.text(max_size=100)),
    st.builds(HumanMessage, content=st.text(max_size=100)),
    st.builds(AIMessage, content=st.text(max_size=100)),
)


class TestMessagesToDictsPreservation:
    """_messages_to_dicts preserves length, content, and produces valid roles."""

    @given(messages=st.lists(_message_st, max_size=20))
    def test_length_preserved(self, messages):
        """Output list has the same length as input list."""
        result = _messages_to_dicts(messages)
        assert len(result) == len(messages)

    @given(messages=st.lists(_message_st, max_size=20))
    def test_content_preserved(self, messages):
        """Each message's content string is preserved verbatim."""
        result = _messages_to_dicts(messages)
        for msg, d in zip(messages, result):
            assert d["content"] == msg.content

    @given(messages=st.lists(_message_st, max_size=20))
    def test_all_roles_valid(self, messages):
        """Every role is a valid OpenAI chat role."""
        valid_roles = {"system", "user", "assistant"}
        result = _messages_to_dicts(messages)
        for d in result:
            assert d["role"] in valid_roles

    @given(messages=st.lists(_message_st, max_size=20))
    def test_each_dict_has_role_and_content_keys(self, messages):
        """Every dict contains exactly 'role' and 'content' keys."""
        result = _messages_to_dicts(messages)
        for d in result:
            assert set(d.keys()) == {"role", "content"}

    @given(messages=st.lists(_message_st, max_size=20))
    def test_role_mapping_correct(self, messages):
        """SystemMessage → system, HumanMessage → user, AIMessage → assistant."""
        result = _messages_to_dicts(messages)
        for msg, d in zip(messages, result):
            if isinstance(msg, SystemMessage):
                assert d["role"] == "system"
            elif isinstance(msg, HumanMessage):
                assert d["role"] == "user"
            elif isinstance(msg, AIMessage):
                assert d["role"] == "assistant"


# ---------------------------------------------------------------------------
# Property 17: SARIF to_dict() JSON round-trip serializability
# ---------------------------------------------------------------------------


class TestSarifJsonRoundTrip:
    """SARIF to_dict() always produces a JSON-serializable dict that round-trips."""

    @given(findings=st.lists(_finding_st, max_size=20))
    @settings(max_examples=50)
    def test_to_dict_is_json_serializable(self, findings):
        """json.dumps(to_dict()) never raises for any valid findings."""
        output = SarifOutput(tool_name="test", tool_version="1.0", findings=findings)
        d = output.to_dict()
        json_str = json.dumps(d)
        assert json.loads(json_str) == d

    @given(
        findings=st.lists(_finding_st, max_size=20),
        tool_name=st.text(min_size=1, max_size=30),
        tool_version=st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True),
    )
    @settings(max_examples=50)
    def test_tool_metadata_preserved_through_serialization(
        self, findings, tool_name, tool_version
    ):
        """Tool name and version survive JSON round-trip."""
        output = SarifOutput(
            tool_name=tool_name, tool_version=tool_version, findings=findings
        )
        d = output.to_dict()
        roundtripped = json.loads(json.dumps(d))
        driver = roundtripped["runs"][0]["tool"]["driver"]
        assert driver["name"] == tool_name
        assert driver["version"] == tool_version


# ---------------------------------------------------------------------------
# Property 18: MockEmbeddings embed_query ↔ embed_documents consistency
# ---------------------------------------------------------------------------


class TestMockEmbeddingsConsistency:
    """MockEmbeddings is a public testing utility. Its embed_query and
    embed_documents methods must be consistent, deterministic, and
    produce correctly-dimensioned vectors."""

    @given(
        text=st.text(max_size=200),
        dimension=st.integers(min_value=1, max_value=64),
    )
    def test_query_equals_single_document(self, text: str, dimension: int):
        """embed_query(text) == embed_documents([text])[0] for any text."""
        emb = MockEmbeddings(dimension=dimension)
        query_vec = emb.embed_query(text)
        doc_vecs = emb.embed_documents([text])
        assert query_vec == doc_vecs[0]

    @given(
        text=st.text(max_size=200),
        dimension=st.integers(min_value=1, max_value=64),
    )
    def test_deterministic_across_instances(self, text: str, dimension: int):
        """Two separate MockEmbeddings instances produce identical vectors."""
        emb1 = MockEmbeddings(dimension=dimension)
        emb2 = MockEmbeddings(dimension=dimension)
        assert emb1.embed_query(text) == emb2.embed_query(text)

    @given(
        texts=st.lists(st.text(max_size=100), max_size=20),
        dimension=st.integers(min_value=1, max_value=64),
    )
    def test_dimension_correct(self, texts: list[str], dimension: int):
        """Every vector has exactly `dimension` components."""
        emb = MockEmbeddings(dimension=dimension)
        vecs = emb.embed_documents(texts)
        for vec in vecs:
            assert len(vec) == dimension

    @given(
        text=st.text(min_size=1, max_size=200),
        dimension=st.integers(min_value=1, max_value=64),
    )
    def test_components_are_finite(self, text: str, dimension: int):
        """All vector components are finite floats (no NaN or inf)."""
        import math

        emb = MockEmbeddings(dimension=dimension)
        vec = emb.embed_query(text)
        for x in vec:
            assert isinstance(x, float)
            assert math.isfinite(x)

    @given(
        texts=st.lists(st.text(max_size=100), max_size=10),
        dimension=st.integers(min_value=1, max_value=64),
    )
    def test_embed_documents_length_matches_input(self, texts: list[str], dimension: int):
        """embed_documents returns exactly one vector per input text."""
        emb = MockEmbeddings(dimension=dimension)
        vecs = emb.embed_documents(texts)
        assert len(vecs) == len(texts)


# ---------------------------------------------------------------------------
# Property 19: ModelSelector validate_model ↔ is_allowed consistency
# ---------------------------------------------------------------------------


class TestModelSelectorValidateIsAllowedConsistency:
    """validate_model raises ModelNotAuthorizedError iff is_allowed returns False.
    Complements Property 3 by testing the negative (rejection) path."""

    @given(
        allowed_refs=st.lists(_model_ref_str_st, min_size=1, max_size=5),
        query_ref=_model_ref_st,
        category=st.sampled_from(list(ModelCategory)),
    )
    def test_validate_raises_iff_not_allowed(
        self, allowed_refs: list[str], query_ref: ModelRef, category: ModelCategory
    ):
        """validate_model and is_allowed agree on authorization decisions."""
        selector = ModelSelector({category.value: allowed_refs})
        if selector.is_allowed(query_ref, category):
            selector.validate_model(query_ref, category)  # should not raise
        else:
            with pytest.raises(ModelNotAuthorizedError):
                selector.validate_model(query_ref, category)

    @given(
        allowed_refs=st.lists(_model_ref_str_st, min_size=1, max_size=5),
        category=st.sampled_from(list(ModelCategory)),
        other_category=st.sampled_from(list(ModelCategory)),
    )
    def test_allowed_in_one_category_not_implies_other(
        self, allowed_refs: list[str], category: ModelCategory, other_category: ModelCategory
    ):
        """A model allowed in one category is not necessarily allowed in another."""
        selector = ModelSelector({category.value: allowed_refs})
        selected = selector.select(category)
        if category != other_category:
            # Model is only configured for `category`, not `other_category`
            assert not selector.is_allowed(selected, other_category)

    @given(
        allowed_refs=st.lists(_model_ref_str_st, min_size=1, max_size=5),
        category=st.sampled_from(list(ModelCategory)),
    )
    def test_select_all_exhausts_allowed(
        self, allowed_refs: list[str], category: ModelCategory
    ):
        """Every model returned by select_all is allowed, and no others are."""
        selector = ModelSelector({category.value: allowed_refs})
        all_models = selector.select_all(category)
        # All models in select_all are allowed
        for model in all_models:
            assert selector.is_allowed(model, category)
        # A random model not in select_all is NOT allowed
        random_model = ModelRef(provider="zzz-nonexistent", model="zzz-fake-model")
        if random_model not in all_models:
            assert not selector.is_allowed(random_model, category)
