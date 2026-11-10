# Module 13 — Harness Engineering

## 00 Why This Module Exists

A model API is not an application contract. The harness is the typed boundary that converts application intent, context, tools, and output requirements into an effective model request, then converts provider events into a validated application outcome. A weak harness makes incompatible targets look interchangeable, treats valid JSON as truth, loses streaming state, retries permanent errors, and cannot explain which request actually ran.

```text
application request + policy + context + schemas
                         |
                  canonical manifest
                         |
               capability negotiation
                  /              \
          unsupported         provider adapter
                                  |
                    effective serialized request
                                  |
                    raw response/event stream
                                  |
        assembly -> normalization -> validation ladder
                                  |
              accepted | repair/retry | fallback | fail
```

This module owns model-I/O request and response contracts, prompt/template serialization, target capability negotiation, provider adapters, structured outputs, constrained decoding, streaming assembly, error normalization, bounded retry/fallback, attempt lineage, and replay/conformance evidence. Module 12 owns the agent loop and tool-action policy; Module 14 owns durable checkpointing and side-effect execution; Module 15 owns the full evaluation program; Module 17 owns compatibility and rollout as the harness evolves; Module 23 owns platform-wide observability.

**Research cutoff:** 2026-09-26 for the original claim set. Entries marked *opened 2026-10-01* in Section 05 were read on that date for registry revision 1.1.0; where a publication or revision date is shown, it is on or before 2026-09-30. Claims not marked that way were not re-verified.

**Module Orientation**
- **Engineering Problem**: Preserve required semantics across changing model endpoints while producing typed, observable, bounded outcomes.
- **What You Will Do**: Build canonical and effective request models, probe backend capabilities, validate structured output, assemble adversarial streams, bound retries/fallbacks, trace XGrammar, and defend a portable production harness.
- **Environment**: Python 3.10+, two mock or disposable model adapters, JSON Schema fixtures, a streaming-event replayer, and privacy-safe request/attempt storage.
- **Evidence Rule**: Distinguish source observations (**O**), derivations (**D**), and telemetry-dependent hypotheses (**H**). Adapter convenience is not evidence of semantic equivalence.

## 01 Baseline Assumptions

- Module 00: experimental units, paired comparisons, uncertainty, and falsification.
- Modules 01 and 06: tokenization, sampling, stopping, and inference-time behavior.
- Module 04: latency distributions, deadlines, retries as load, and goodput.
- Module 07: behavioral uncertainty, abstention, and regression slicing.
- Modules 08–11: lineage, evidence, context accounting, and versioned state.
- Module 12: tool contracts, authority, effect evidence, and bounded agent loops.

## 02 Target Mastery

```yaml
depth_contract:
  conceptual: REQUIRED
  mechanistic: REQUIRED
  mathematical: REQUIRED
  quantitative: REQUIRED
  implementation: REQUIRED
  source_code: REQUIRED
  instrumentation: REQUIRED
  experimental: REQUIRED
  statistical: SELECTIVE
  production_reasoning: REQUIRED
  failure_analysis: REQUIRED
  falsification: REQUIRED
  security: SELECTIVE
  economics: SELECTIVE
  architecture_tradeoff: REQUIRED
  research_connection: SELECTIVE

estimated_effort:
  instruction: 4h       # sum of lesson instruction lines: 40+35+45+50+35+35 min
  guided_practice: 3h   # sum of lesson practice lines: 30+25+35+35+25+30 min
  labs: 12h             # LAB A-D, 3h each
  assessment: 3h        # Mastery transfer problem 2.5h + Incident 13.1 0.5h
  source_trace: 2h      # Section 09 Production Source Trace artifact
  total: 24h
```
Each category is counted once. The source trace is not also counted inside Lesson 13.3 or LAB B, and lab analysis is not counted again as lesson practice.

The learner must be able to define canonical and effective requests; reject silent capability loss; version prompts and serializers; separate parse, schema, semantic, authority, and effect checks; implement structured and streaming outputs; classify errors before retry/fallback; account for every attempt; pin and trace a structured-decoding implementation; and diagnose raw-versus-normalized protocol drift.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
                       application contract
                 task | content | tools | output
                              |
                    canonical request manifest
                              |
                  required capability set C_req
                              |
                 target capability set C_verified
                    /                       \
               reject/degrade          adapter compile
                                              |
                                  effective request bytes
                                              |
                                 provider/runtime/model
                                              |
                                    ordered raw events
                                              |
                         stream assembler + normalizer
                                              |
             parse -> schema -> invariants -> task/evidence
                         -> authority -> postcondition
```

Keep three representations:

1. **Canonical request:** application meaning before provider translation.
2. **Effective request:** exact transformed payload, resolved target, and adapter policy.
3. **Raw plus normalized response:** retain transport truth while exposing stable application types.

## 04 Lessons

### Lesson 13.1 — Harness Boundary and Effective Request

**Engineering Question:**
Which representations must be retained to explain what the application intended, what the adapter sent, and what the provider returned?

**Concepts & Definitions:**
- **Canonical request**: provider-neutral application intent and requirements.
- **Effective request**: exact resolved target, transformed payload, and adapter policy.
- **Normalized response**: stable application type with provenance to raw events.

**Mechanism Explanation:**

A canonical request manifest includes task/version, prompt template and rendered messages, typed content blocks, context and memory selections, tool and output schemas, model alias and resolved target, sampling/stopping parameters, timeout, metadata, and adapter policy. The adapter compiles it into an effective request; record both where policy permits (**D**, CLM-001, CLM-002).

Prompts and serializers are executable dependencies. A 2025 controlled study reports sensitivity to subtle phrasing and formatting changes across its evaluated tasks and models (**O**, CLM-003). Do not universalize its effect sizes; use it to justify versioning delimiters, ordering, examples, escaping, chat templates, and schema renderers and testing them on your workload.

The normalized response must not destroy the raw response. Stable application fields need provenance back to raw events and an explicit `missing`, `unsupported`, or `not reported` state rather than fabricated defaults.

**Quantitative Model / Derivation:**
Let $R_c$ be a canonical request and $A_p$ the adapter for provider $p$. The effective request is $R_e=A_p(R_c)$. A conformance check compares required semantic fields before and after this transformation; byte equality is neither required nor sufficient.

where:
- $R_c$ — the canonical request: what the application means, written without any provider's field names.
- $p$ — one provider/target; $A_p$ — the adapter (a function) that translates for that target.
- $R_e$ — the effective request: the output of $A_p$, i.e. what is actually sent.

In plain terms, the adapter is a translator. "Not required" means $R_e$ may use different names or shapes (row 6 below renames a field and still preserves meaning). "Not sufficient" means two payloads that look alike can still differ in meaning (rows 1–2 below). The check therefore asks, field by required field, "does the target still do what $R_c$ asked?", not "are the bytes the same?".

**Worked Example (synthetic fixture):**

*Input.* One canonical request for a funds-transfer tool call, and what a hypothetical adapter "B" sent and returned for it. Field names are illustrative and are not any provider's API.

| # | Required semantic | Canonical request $R_c$ | Effective request or normalized response via adapter B | Status |
|---|---|---|---|---|
| 1 | Forced tool choice | `tool_choice = {"type":"tool","name":"transfer_funds"}` | `tool_choice = "auto"` | **Weakened** |
| 2 | Strict output schema | `output_schema = {"id":"sch-transfer-v3","strict":true}` | schema body sent, `strict` key absent | **Dropped** |
| 3 | Stop sequence | `stop = ["\n\nEND"]` | not sent; the adapter cuts the returned text at the first `\n\nEND` | **Emulated** |
| 4 | Usage reporting | `usage_reporting = "required"` | raw response has no usage object; normalized response shows `{"input_tokens":0,"output_tokens":0}` | **Fabricated** |
| 5 | Sampling | `temperature = 0.2` | `temperature = 0.2` | Preserved |
| 6 | Output limit | `max_output_tokens = 512` | `max_tokens = 512` | Renamed |
| 7 | Message roles | `[system, user]` | system text prepended to the first user turn | Transformed |

*Steps.*
1. Compare by required semantic, not by bytes. Rows 5–7 differ in name or shape and may still preserve meaning. Rows 1–2 look similar and change meaning.
2. Classify each row.
   - *Preserved* and *Renamed* stay claims until a probe shows the target counts and applies the value the same way.
   - *Transformed* needs a serializer version and a fixture, because merged roles can change behavior.
   - *Emulated* must be declared. In row 3 the model keeps generating past the stop, those tokens are still produced and billed, and the finish reason comes from the adapter, not the provider.
   - *Weakened* and *Dropped* are capability losses.
   - *Fabricated* is the worst case: zero is a value. The correct normalized state is `usage: not_reported`.
3. Apply the dispatch rule from Lesson 13.2. $C_{req}$ = {forced tool choice, strict schema, stop sequence, usage reporting}. $C_{verified}(B)$ = {stop sequence, by declared emulation}. Three required capabilities are missing.

*Result.* Seven fields: one preserved, one renamed, one transformed, one emulated, one weakened, one dropped, one fabricated. Dispatch through B is rejected with `unsupported: forced_tool_choice, strict_schema, usage_reporting`, or returned as an explicitly degraded outcome if policy authorizes that.

*Interpretation and limits.* The classification is a derivation on a synthetic fixture (**D**, CLM-001, CLM-002). It does not describe any real adapter. A byte-level diff alone would have flagged rows 6–7 and could have missed row 4, which only appears when the raw response is compared with the normalized one.

**Knowledge Check:**
1. Why must raw provider events survive normalization?
2. Which version identifiers are needed to reproduce prompt serialization?

**Guided Practice:**
(a) Classify adapter "C" for the same canonical request (synthetic): it sends the forced tool choice unchanged; sends the schema with `strict: true` but removes the `pattern` keyword it does not support; sends the stop sequence; and returns usage as `{"total_tokens": 412}` only. (b) Then create canonical and effective manifests for one task of your own through two adapters and explain every semantic difference.

**Feedback Contract:**
- *Expected Evidence*: For (a): tool choice preserved; schema **dropped keyword** (`strict` survives, `pattern` does not, so the sent schema accepts more than the canonical one); stop preserved; usage **partially reported** (`total_tokens` known, the input/output split is `not_reported`, not zero). Decision: reject, or degrade explicitly, because the effective schema is weaker. For (b): exact target, prompt/schema/serializer versions, a per-field status table, the raw response, and explicit missing/unsupported states.
- *Common Failure*: Recording only the friendly SDK object, or marking adapter C "compatible" because `strict: true` is present.
- *Diagnostic Hint*: Can the exact outbound payload be reconstructed? Diff the schema that was sent, not the schema that was requested.
- *Concept to Revisit*: Canonical vs. Effective Request.

**Learning Outcome:**
Replay which logical request was intended, which bytes/events were exchanged, and which transformations produced the application result.

*(Effort: 40m instruction, 30m practice)*

---

### Lesson 13.2 — Capability Negotiation and Portability

**Engineering Question:**
When is a fallback semantically compatible rather than merely API-shaped like the preferred backend?

**Concepts & Definitions:**
- **Required capability set**: features whose absence changes the application contract.
- **Verified capability set**: behavior demonstrated for a pinned target and adapter.
- **Degraded outcome**: explicit result when policy authorizes relaxing a requirement.

**Mechanism Explanation:**

Define required capabilities `C_req(r)` and verified target capabilities `C_verified(p)`. Dispatch only when:

$$
C_{req}(r)\subseteq C_{verified}(p),
$$

after recording any deliberate emulation. This is an exact policy rule; discovering capabilities is empirical and must be refreshed (**D**, CLM-004).

Probe at least content roles/types, context and output limits, tool-call semantics, schema dialect/subset, streaming event types, log probabilities, seeds, stop behavior, usage, cancellation, refusals, and errors. The same field name or “compatible API” label does not prove identical behavior.

A fallback is compatible only if it satisfies required capabilities and passes task-specific prompt/schema/serializer fixtures. If emergency policy relaxes a requirement, return an explicit degraded outcome (**D**, CLM-011). Routing for price/quality optimization belongs to Module 22; here the question is whether a call remains contract-compatible.

**Quantitative Model / Derivation:**
Dispatch is allowed only when $C_{req}(r)\subseteq C_{verified}(p)$ after declared emulation. This set rule is exact for a manifest; the membership evidence is empirical and can expire.

where:
- $r$ — one request; $C_{req}(r)$ — the set of capabilities that request cannot do without.
- $p$ — one target; $C_{verified}(p)$ — the set of capabilities a probe has demonstrated on that pinned target and adapter (declared emulations included).
- $\subseteq$ — "is a subset of": every element on the left also appears on the right.

Step-by-step, the subset test is the same as checking that nothing is left over:

1. Compute the missing set $M = C_{req}(r)\setminus C_{verified}(p)$ (required, but not verified).
2. If $M=\emptyset$, then $C_{req}(r)\subseteq C_{verified}(p)$ and dispatch is allowed.
3. If $M\neq\emptyset$, reject and name the elements of $M$, or return an explicitly degraded outcome if policy allows.

Extra verified capabilities on the target do not matter; only missing required ones do. Computing $M$ rather than a yes/no answer is what makes the rejection message actionable.

**Worked Example (synthetic fixture):**
*Input.* $C_{req}$ = {JSON object output, recursive schema, parallel tool-call IDs, streaming}. Probes on the fallback target verified {JSON object output, streaming}.
*Steps.* $C_{req}\setminus C_{verified}$ = {recursive schema, parallel tool-call IDs}. The set is not empty, so the subset rule fails.
*Result.* The endpoint is reachable and returns parseable JSON, yet dispatch is rejected with two named missing capabilities, or returned as an explicitly degraded outcome if policy allows.
*Interpretation and limits.* The rule is exact for the manifest. The membership of $C_{verified}$ is only as current as its last probe; a provider or model revision invalidates it.

**Knowledge Check:**
1. Why does an identical parameter name not prove identical behavior?
2. Which capability probes must rerun after a provider/model revision?

**Independent Practice:**
(a) Synthetic check: $C_{req}$ = {tool calling, strict schema, seed, usage}. Target P verified {tool calling, strict schema, usage}; target Q verified {tool calling, seed, usage} and accepts a `strict` field without enforcing it. Decide dispatch for each. (b) Build a two-target capability matrix of your own and a fixture that catches one silent default and one unsupported feature.

**Feedback Contract:**
- *Expected Evidence*: For (a): P is missing {seed}; Q is missing {strict schema}, because accepting a field is not enforcing it. Neither satisfies the subset rule, so both are rejected or explicitly degraded. For (b): each required capability maps to a current probe, version, and failure policy.
- *Common Failure*: Treating successful HTTP transport, or an accepted parameter name, as portability.
- *Diagnostic Hint*: Which required semantic was actually exercised by a probe that would fail if the feature were ignored?
- *Concept to Revisit*: Evidence-Backed Capability Negotiation.

**Learning Outcome:**
Reject silent field loss and demonstrate portability with conformance evidence rather than adapter claims.

*(Effort: 35m instruction, 25m practice)*

---

### Lesson 13.3 — Structured Output and Constrained Decoding

**Engineering Question:**
What does constrained decoding guarantee, and which semantic, authority, and effect checks remain outside the grammar?

**Concepts & Definitions:**
- **Dialect/subset**: the exact schema vocabulary implemented by the validator or decoder.
- **Constrained decoding**: masking invalid next tokens for an implemented language.
- **Validation ladder**: parse, schema, invariants, task/evidence, authority, and effect checks.

**Mechanism Explanation:**

Pin the schema dialect and vocabulary. JSON Schema Draft 2020-12 separates Core and Validation vocabularies; `format` is annotation by default unless assertion behavior is enabled (**O**, CLM-005; Validation specification §7.2). Hosted APIs and decoding libraries may implement different subsets, strictness, recursion, ordering, and extensions.

Use a validation ladder:

```text
bytes/text parse?
  -> declared schema valid?
    -> application invariants valid?
      -> task/factual/evidence checks pass?
        -> action authorized?
          -> external postcondition observed?
```

Read the ladder top to bottom: each stage assumes the previous one passed and reads one more input. Parse reads only the text; schema adds the schema; invariants add application rules; task/evidence adds the user's request and sources; authority adds who is calling; the postcondition adds the outside world after execution. A failure is reported at the first stage that fails, so its name says which input was violated.

These predicates are not equivalent (**D**, CLM-007). Constrained decoding can keep generation inside an implemented formal language by masking invalid continuations (**O**, CLM-006); it cannot prove facts, business invariants absent from the grammar, permission, or side effects.

JSONSchemaBench evaluates schema-feature coverage, efficiency, and output quality separately across real-world schemas and the official test suite (**O**, CLM-008; abstract-level reading). Use the same dimensions for backend selection; valid-JSON rate alone hides unsupported keywords and semantic failure.

**Production source trace:** at XGrammar revision `4221346f3d26b8306b46de19cb40c8a525c4871c`, `GrammarCompiler.compile_json_schema` produces a tokenizer-aware compiled grammar, `GrammarMatcher.fill_next_token_bitmask` and `accept_token` expose stateful validity, and the Transformers `LogitsProcessor.__call__` accepts the last token, fills the next mask, and applies it to logits (**O**, CLM-015). Static inspection only, re-read at this commit on 2026-10-01; native code and benchmarks were not executed.

Three details from that reading matter for the ladder. They are one library's behavior at one commit, not a property of constrained decoding in general:

- `compile_json_schema` defaults to `strict_mode=True`, which its docstring describes as disallowing properties and items not named in the schema. The caller's schema is therefore not the only input that defines the accepted language.
- With `any_order=True`, the docstring states that required keys may be missing and keys may repeat. A grammar-valid output can then fail post-hoc schema validation, so stage 2 still has to run.
- `accept_token` returns `False` when a token does not match, and the Transformers processor asserts on it and is documented as single-use per `generate()` call.

XGrammar-2's 2026 maintainer material extends the family with composable structural tags and batching/speculative-decoding integration, and states that the library is for enforcing format rather than changing response semantics (**O**, CLM-016). Treat it as frontier, implementation-specific evidence; do not transfer its performance or adoption claims without reproduction.

**Quantitative Model / Trade-off Comparison:**
Report schema-feature coverage, compile/cache time, decode latency, structural validity, semantic validity, and task success separately. A structurally valid rate cannot be substituted for semantic accuracy.

**Worked Example (synthetic fixture):**

*Input.* Four separate artifacts. Keeping them separate is the point of the example.

1. Candidate document, 31 bytes of UTF-8 text:

```json
{"amount": -10, "account": "B"}
```

2. Permissive schema, JSON Schema Draft 2020-12:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "amount": {"type": "number"},
    "account": {"type": "string"}
  },
  "required": ["amount", "account"],
  "additionalProperties": false
}
```

3. Business invariant, enforced in application code and absent from the schema: a transfer amount is strictly positive, `amount > 0`.
4. Caller permissions, supplied by the authorization layer and unknown to the schema:

```json
{"caller": "u-17", "may_debit": ["A"]}
```

*Steps.*

| Stage | Check | Inputs used | Result |
|---|---|---|---|
| 1. Parse | standard JSON parser | document text | pass: an object with two members |
| 2. Schema | Draft 2020-12 validator | parsed object, schema | pass: `-10` is a number, `"B"` is a string, both required keys are present, no extra key |
| 3. Invariant | `amount > 0` | parsed object | **fail**: `-10 > 0` is false |
| 4. Task/evidence | does the amount match what the user asked for? | the user request | not evaluable: the fixture has no request |
| 5. Authority | `account` is in `may_debit` | parsed object, caller permissions | **fail**: `"B"` is not in `["A"]` |
| 6. Effect | postcondition after dispatch | external state | not reached: nothing is dispatched |

*Result.* The **first failing stage is stage 3**. A ladder stops there and returns a typed `invariant_violation`; it does not report "invalid JSON". Stage 5 is evaluated here only to show that it fails for an unrelated reason. Changing one input at a time separates the two:

| Document | Parse | Schema | Invariant | Authority |
|---|---|---|---|---|
| `{"amount": -10, "account": "B"}` | pass | pass | fail | fail |
| `{"amount": 10, "account": "B"}` | pass | pass | pass | fail |
| `{"amount": -10, "account": "A"}` | pass | pass | fail | pass |
| `{"amount": 10, "account": "A"}` | pass | pass | pass | pass |

*Encoding layer.* The text `{\"amount\": -10, \"account\": \"B\"}` is **not** a JSON document. A standard parser rejects it at offset 1, because a backslash cannot start a member name. That text is how the document appears *inside a JSON string literal*, for example when a tool call carries its arguments as a string:

```json
{"arguments": "{\"amount\": -10, \"account\": \"B\"}"}
```

Parsing this envelope yields a *string* for `arguments`. Validating that string against the schema fails at stage 2 with "is not of type 'object'", which is a harness error, not a model error. Decode one layer first: parse the envelope, take the string value, parse that string as its own JSON document, and only then start the ladder. Record which layer each failure belongs to.

*Interpretation and limits.*
- Where a constraint is declared decides which stage catches it. Adding `"exclusiveMinimum": 0` to `amount` makes the same document fail at stage 2 instead of stage 3.
- Authority cannot move into a static schema, because it depends on who is calling.
- A decoder constrained by the permissive schema may emit this document: it is inside the schema's language. That statement is a derivation (**D**, CLM-022), not a measurement of any decoder.
- The parse and schema results above are reproducible with Python `json.loads` and `jsonschema.Draft202012Validator`. No model was run.

**Knowledge Check:**
1. Why is `format` behavior dependent on dialect/configuration?
2. Can a grammar verify that a cited fact is true?

**Guided Practice:**
(a) With the same schema, invariant, and permissions, name the first failing stage for each input: (i) `{"amount": "10", "account": "A"}`; (ii) `{"amount": 10, "account": "A", "memo": "x"}`; (iii) `{"amount": 0, "account": "A"}`; (iv) the envelope `{"arguments": "{\"amount\": 5, \"account\": \"A\"}"}` handed to the ladder without decoding. (b) Test nested unions, references, optional fields, impossible domains, and unsupported keywords across post-hoc validation and one constrained backend.

**Feedback Contract:**
- *Expected Evidence*: For (a): (i) stage 2, a string is not a number; (ii) stage 2, `additionalProperties` is false; (iii) stage 3, zero is not strictly positive, and the schema passes; (iv) stage 2 for the wrong reason, the envelope has no `amount` or `account`; after decoding one layer the inner document passes every evaluated stage. For (b): pinned dialect/subset, feature tests, structural and semantic metrics, and unsupported cases.
- *Common Failure*: Reporting valid JSON as task correctness; reporting (iv) as a model failure; reporting (iii) as a schema failure.
- *Diagnostic Hint*: At which validation stage did the candidate first fail, and which of the four artifacts did that stage read?
- *Concept to Revisit*: Validation Ladder.

**Learning Outcome:**
Prove structural validity for supported constraints while keeping semantic validation and system effects separate.

*(Effort: 45m instruction, 35m practice)*

---

### Lesson 13.4 — Streaming Is a Protocol State Machine

**Engineering Question:**
How can arbitrary legal event boundaries yield one stable logical response without converting truncation or cancellation into success?

**Concepts & Definitions:**
- **Transport read**: a byte chunk delivered by the network stack; its boundaries carry no meaning.
- **Decoded provider event**: one framed protocol event with identity and ordering metadata, produced by decoding the byte stream.
- **Logical object**: content or tool call assembled according to terminal semantics.
- **Terminal state**: complete, truncated, cancelled, refused, or failed.

**Mechanism Explanation:**

Provider chunks are not complete JSON objects or messages (**D**, CLM-009). Two boundaries must not be confused: a *transport read* can end anywhere, including inside a multibyte character, while a *decoded provider event* ends where the wire format says it ends. Preserve sequence/event IDs, choice index, tool-call ID, event type, raw bytes/text, timestamps, and terminal/error/cancel signals. Assemble fragmented Unicode, content, reasoning channels where exposed, and tool arguments by stable identity.

Do not parse every partial JSON fragment as though it were complete. Track states such as `started`, `partial`, `complete`, `truncated`, `cancelled`, `refused`, and `failed`; validate a logical object when its protocol says it is complete. Client disconnect does not prove provider cancellation, and a finish reason must remain distinct from application acceptance.

Test boundary splits, duplicated/out-of-order events where transport allows them, parallel choices/calls, missing terminators, early disconnect, content after a terminal marker, and backpressure from a slow consumer.

**Quantitative Model / Derivation:**
Assembly is two folds. A decoder $D$ turns the byte stream $B$ into decoded events $E=D(B)$. An assembler folds events into state, $S_n=F(S_0,E)$.

- **Chunk-boundary invariance** is a property of $D$: every partition of $B$ into contiguous non-empty chunks must yield the same event sequence. A byte string of $n$ bytes has $2^{n-1}$ such partitions (**D**, CLM-021).
- **Delta-boundary invariance** is a property of $F$: a provider may cut the same arguments into different deltas, such as `{"un` then `it":"C"}`. The event sequences differ; the completed objects and terminal state must not.

where: $B$ is the raw byte stream; $D$ the decoder (framer); $E$ the resulting sequence of decoded events; $F$ the assembler; $S_0$ the empty initial state; $S_n$ the state after all events are applied. Note the symbol reuse: the $n$ in $S_n$ counts *events*, while the $n$ in $2^{n-1}$ counts *bytes*. They are different numbers.

*Why $2^{n-1}$.* A string of $n$ bytes has $n-1$ internal gaps (between bytes 1 and 2, 2 and 3, and so on). A partition into contiguous non-empty chunks is fully described by which gaps get a cut. Each gap is independently "cut" or "no cut", so there are $2\times2\times\cdots\times2=2^{n-1}$ choices. Example with $n=3$ bytes `abc`: 2 gaps, $2^2=4$ partitions: `abc`, `a|bc`, `ab|c`, `a|b|c`.

The count decides the test method:

| Fixture | Bytes $n$ | Partitions $2^{n-1}$ | Method |
|---|---:|---:|---|
| `data: "Đ"` plus a blank line | 12 | 2,048 | **exhaustive**: run all of them |
| Input B below, as serialized | 580 | $2^{579}$ | **generated**: all 579 single cuts, all $\binom{579}{2}=167{,}331$ double cuts, one byte per chunk, plus seeded random partitions |

In the 12-byte fixture, 1,024 of the 2,048 partitions cut between the two bytes of `Đ` (that one gap is forced to "cut" and the other 10 gaps are free: $2^{10}=1{,}024$, exactly half); a decoder that converts each chunk to text independently raises on exactly those. For large inputs a passing run means "no counterexample in the tested set", and the report must state the seed, the number of partitions, and which families were exhaustive. It is not a proof over all partitions.

*Sizing the generated families for $n=580$.* There are $n-1=579$ gaps. Single cuts choose 1 gap: 579 tests. Double cuts choose 2 distinct gaps: $\binom{579}{2}=\frac{579\times578}{2}=167{,}331$ tests. With the one-byte-per-chunk case that is under 170,000 deterministic runs, which is affordable, whereas $2^{579}$ is not. The "about 16 bytes" limit in LAB C corresponds to $2^{15}=32{,}768$ partitions.

**Worked Example (synthetic protocol fixture):**

Three layers are kept apart throughout:

| Layer | Unit | Boundary decided by | Identity |
|---|---|---|---|
| Transport read | byte chunk | network and buffers; arbitrary | byte offset |
| Decoded provider event | one framed event | the wire format's terminator | event ID, type |
| Logical object | one tool call or message | protocol completion events | call ID and index |

The fixture uses server-sent-event framing: UTF-8 text, one field per line, a blank line dispatches the event, and an event that is still incomplete when the stream ends is discarded rather than dispatched (**O**, CLM-018). The event names below are invented for the exercise. Hosted APIs define their own; one provider documents `response.output_item.added`, `response.function_call_arguments.delta`, and `response.function_call_arguments.done`, with `item_id`, `output_index`, and `call_id` identifying the call (**O**, CLM-019).

*Input A: transport fragmentation of one event.* Event 3 is 75 bytes on the wire:

```text
id: 3
event: tool_call.delta
data: {"index":0,"delta":"{\"city\":\"Đà"}
<blank line>
```

The `data` line is a JSON document whose `delta` value is a JSON *string* holding a fragment of another JSON document: the two encoding layers of Lesson 13.3. `Đ` is the two bytes `C4 90`. Suppose the transport delivers the 75 bytes in three reads:

| Read | Bytes | Read ends | Undecoded bytes held | Framer state after the read | Decoded events emitted |
|---|---:|---|---:|---|---|
| r1 | 68 | after `C4`, the first byte of `Đ` | 1 | two complete lines, one partial `data` line | none |
| r2 | 6 | before the final line feed | 0 | three complete lines, no blank line yet | none |
| r3 | 1 | at the final line feed | 0 | blank line seen: dispatch and reset | event 3 |

Three reads produce exactly one decoded event. Decoding r1 by itself as UTF-8 fails on the dangling lead byte, and parsing r1 as JSON fails. Neither is a protocol error: reads are not units of meaning.

*Input B: decoded events for two parallel tool calls.* `␠` marks a leading space in a fragment.

| Event ID | Type | Key (index, call ID) | Fragment | State after | Arguments buffer after |
|---|---|---|---|---|---|
| 1 | `tool_call.start` | 0, `call_a`, `lookup_city` | none | 0: started | 0: empty |
| 2 | `tool_call.start` | 1, `call_b`, `get_weather` | none | 0: started; 1: started | 0: empty; 1: empty |
| 3 | `tool_call.delta` | 0 | `{"city":"Đà` | 0: partial | 0: `{"city":"Đà` |
| 4 | `tool_call.delta` | 1 | `{"unit":` | 1: partial | 1: `{"unit":` |
| 5 | `tool_call.delta` | 0 | `␠Nẵng"}` | 0: partial | 0: `{"city":"Đà Nẵng"}` |
| 6 | `tool_call.delta` | 1 | `"C"}` | 1: partial | 1: `{"unit":"C"}` |
| 7 | `tool_call.done` | 0 | none | 0: complete | parse buffer 0 |
| 8 | `tool_call.done` | 1 | none | 1: complete | parse buffer 1 |
| 9 | `response.completed` | none | none | terminal: complete | hand both objects to the validation ladder |

*Result.* Terminal state `complete`, and two logical objects: `call_a = lookup_city({"city": "Đà Nẵng"})` and `call_b = get_weather({"unit": "C"})`. Each fragment went to the buffer named by its key. Arguments were parsed at events 7 and 8, not earlier, although both buffers already parse as JSON after event 6.

*Rejection cases on the same fixture.*

| Variant | What the assembler holds | Terminal state | Why it is not a success |
|---|---|---|---|
| Connection closes after event 4 | 0: `{"city":"Đà`; 1: `{"unit":` | `truncated` | no `done`, no terminal event; the buffers do not parse |
| Connection closes after event 6 | 0: `{"city":"Đà Nẵng"}`; 1: `{"unit":"C"}` | `truncated` | both buffers parse, but no protocol event declared them complete; parseable is not complete |
| Deltas appended in arrival order, key ignored | `{"city":"Đà{"unit": Nẵng"}"C"}` | `failed` | two tool calls merged into one buffer; the parse fails at offset 13 |
| Event 5 applied twice, no de-duplication by event ID | 0: `{"city":"Đà Nẵng"} Nẵng"}` | `failed` unless the repeat is dropped by ID | a redelivered event must not be appended again |
| A delta for index 0 arrives after event 7 | unchanged | `failed` | fragment after `done` for that call |
| Any event arrives after event 9 | unchanged | `failed` | content after the terminal marker |

*Interpretation and limits.* The trace is a derivation on a synthetic protocol (**D**, CLM-009, CLM-021). Real event taxonomies, and whether duplicates or reordering can occur at all, depend on the provider and transport. The second rejection row is the one most often missed: a buffer such as `12` also parses, and the next delta could have made it `120`.

**Knowledge Check:**
1. Why is parsing each JSON fragment independently incorrect?
2. Does client disconnect prove provider-side cancellation?

**Guided Practice:**
(a) How many chunk partitions does the 9-byte stream `data: 1` plus a blank line have, and should the test be exhaustive? (b) A stream delivers `tool_call.start` for index 0, then one delta `12`, then the connection closes. State the terminal state and whether the arguments are validated. (c) The same stream instead continues with a delta `0`, `tool_call.done`, and `response.completed`. State the completed arguments. (d) Replay one response of your own over generated chunkings, duplicate/out-of-order cases permitted by the transport, missing terminators, and slow-consumer backpressure.

**Feedback Contract:**
- *Expected Evidence*: (a) $2^{8}=256$; exhaustive. (b) `truncated`; not validated, although `12` parses as JSON. (c) `120`, which shows why (b) must not be accepted as `12`. (d) Raw ordered events, stable IDs, an explicit terminal state, the same completed output across the tested splits, and a statement of which split families were exhaustive and which were sampled, with seed and count.
- *Common Failure*: Treating end-of-connection as successful completion; validating a buffer because it happens to parse; writing "all chunkings tested" for a sampled run.
- *Diagnostic Hint*: Which protocol event authorizes final validation? Which key did each fragment carry?
- *Concept to Revisit*: Streaming State Machine.

**Learning Outcome:**
Reconstruct the same completed logical response across legal chunkings and fail closed on ambiguous termination.

*(Effort: 50m instruction, 35m practice)*

---

### Lesson 13.5 — Error Taxonomy, Repair, Retry, and Fallback

**Engineering Question:**
Which failure classes justify retry, repair, fallback, abstention, or immediate stop under one root deadline?

**Concepts & Definitions:**
- **Stage/class**: where and why a failure occurred.
- **Root budget**: attempts, deadline, tokens, and cost shared across all recovery layers.
- **Semantic retry**: another stochastic sample, not restoration of a transport exchange.

**Mechanism Explanation:**

Normalize both stage and class:

| Stage/class | Default decision direction |
|---|---|
| Local schema/config validation | reject or repair before dispatch |
| Authentication/permission/policy | do not retry unchanged |
| Connection/transport with no response | bounded retry if deadline permits |
| Rate limit/overload | honor server signal; budgeted backoff/fallback |
| Deadline/cancellation | stop work or return explicit unfinished state |
| Truncation/context/output limit | change request/budget; blind resample is insufficient |
| Parse/schema failure | validate backend support; bounded repair/resample |
| Refusal/content policy | preserve as typed outcome, not transport error |
| Semantic/evidence failure | new sample, validator-guided repair, abstain, or escalate |

All attempts share root request ID, absolute deadline, maximum attempts, token/cost budget, and cancellation. A semantic retry is another stochastic sample, not restoration of a failed network exchange. Never allow nested SDK, gateway, harness, and application retries to multiply invisibly (**D**, CLM-010).

**Quantitative Model / Derivation:**
If layer $j$ independently permits $r_j$ attempts, hidden nesting can allow up to $\prod_j r_j$ dispatches. A root attempt budget converts this multiplication into one explicit bound.

where: $j$ indexes the retry layers (for example SDK, gateway, application); $r_j$ is the attempt limit of layer $j$, counting the first attempt; $\prod_j r_j$ is the product of all the $r_j$.

*Why multiply, not add.* Each attempt of an outer layer starts a fresh run of the next inner layer, which can again use all of its own attempts. So the counts nest: $r_1$ outer attempts, each with up to $r_2$ inner ones, gives $r_1 r_2$, and a third layer multiplies again. With one shared ledger, every physical dispatch decrements the same counter, so the bound is the root maximum $R$ (precisely $\min(R,\prod_j r_j)$).

The same idea applies to time. With an absolute deadline $T_{dl}$ and elapsed time $t$, an attempt may use at most $\min(\text{configured timeout},\,T_{dl}-t)$. A fresh per-layer timeout ignores $t$ and can overrun the root deadline (Guided Practice (b)).

**Worked Example (synthetic fixture):**
*Input.* An SDK, a gateway, and the application each allow three attempts.
*Steps.* Uncoordinated, each application attempt can trigger three gateway attempts, each of which can trigger three SDK attempts: $3\times3\times3$.
*Result.* Up to 27 physical dispatches for one root request. With a root maximum of three, every layer draws from one ledger and one absolute deadline, so the bound is 3.
*Interpretation and limits.* 27 is a worst-case count, not an expected value; the expected number depends on failure probabilities that this example does not give.

**Knowledge Check:**
1. Why should permission denial not be retried unchanged?
2. When is a fallback incompatible even if it returns parseable output?

**Guided Practice:**
(a) Layers allow 2, 3, and 2 attempts. Give the uncoordinated worst case and the bound under a root budget of 4. (b) A request with a 10 s absolute deadline has used 7 s; the next attempt's timeout is configured as 5 s. What timeout may the attempt use? (c) Map each error-table row to a decision, evidence requirement, deadline behavior, and terminal outcome.

**Feedback Contract:**
- *Expected Evidence*: (a) $2\times3\times2=12$ dispatches uncoordinated; 4 under the root budget. (b) At most 3 s, the remaining deadline; a fresh 5 s would end at 12 s. (c) Stage/class, effect certainty, shared deadline/attempt ledger, and fallback capability proof for every row.
- *Common Failure*: Nested retries, adding the layer limits (7) instead of multiplying them, or a fresh deadline at every layer.
- *Diagnostic Hint*: How many physical dispatches can one root request create, and which clock does each attempt read?
- *Concept to Revisit*: Attempt Amplification.

**Learning Outcome:**
Explain why each retry or fallback is safe, useful, and still inside the original contract.

*(Effort: 35m instruction, 25m practice)*

---

### Lesson 13.6 — Conformance, Replay, and Attempt Accounting

**Engineering Question:**
What evidence localizes drift to request construction, backend behavior, streaming, normalization, validation, or recovery?

**Concepts & Definitions:**
- **Replay manifest**: immutable lineage for logical request, effective attempts, raw events, validation, and terminal state.
- **Conformance fixture**: a known contract probe, not a promise of bitwise model replay.
- **Attempt amplification**: work across all attempts relative to root requests or accepted output.

**Mechanism Explanation:**

A replay manifest records resolved endpoint/model, harness/adapter versions, canonical and effective requests, prompt/tool/schema IDs, sampling/stopping settings, seed if supported, timestamps, raw events, normalized output, validation results, attempts, usage, and terminal status. This supports diagnosis but cannot guarantee bitwise replay when kernels are stochastic, providers revise hidden components, or state is unavailable (**D**, CLM-012).

**Quantitative Model / Derivation:**
For sequential attempts:

$$
L_{e2e}=L_{pre}+\sum_{i=1}^{N}
(L_{serialize,i}+L_{queue/connect,i}+L_{stream,i}+L_{validate/repair,i}+L_{backoff,i}).
$$

where: $L_{e2e}$ is the end-to-end latency of one root request; $L_{pre}$ is the one-time work before the first attempt; $N$ is the number of attempts for this root request, indexed by $i$; each $L_{\ldots,i}$ term is the time attempt $i$ spends serializing, waiting for queue/connection, streaming, validating or repairing, and backing off before the next attempt. The plain sum is valid only because the attempts run one after another.

Use the critical path for overlapping work. For example, if two fallback probes run concurrently and take 300 ms and 500 ms, the caller waits about 500 ms (the longer branch), not 800 ms. Tokens and cost, by contrast, are paid on every branch. Sum tokens and cost across every attempt. Track:

$$
A=\frac{N_{attempts}}{N_{root\ requests}},\qquad
A_{tok}=\frac{T_{all\ attempts}}{T_{final\ accepted}}.
$$

State denominator policy when no output is accepted. These ratios measure amplification, not quality (**D**, CLM-013).

where: $N_{attempts}$ is the total number of attempts across all root requests in the window; $N_{root\ requests}$ is the number of root (user-level) requests, successful or not; $T_{all\ attempts}$ is tokens summed over every attempt; $T_{final\ accepted}$ is tokens of only the attempts whose output was accepted. $A=1$ and $A_{tok}=1$ mean no amplification. This $A$ is unrelated to the adapter $A_p$ of Lesson 13.1, and $N_{attempts}$ is a window total, while $N$ in $L_{e2e}$ is one request's count.

The wasted share of tokens follows directly: $1-T_{final\ accepted}/T_{all\ attempts}=1-1/A_{tok}$.

Run conformance fixtures for canonical requests, schema vocabulary, stream splitting, errors, cancellation, usage, and raw-versus-normalized preservation. The claim that such probes catch adapter drift earlier is a hypothesis (**H**, CLM-017): compare time-to-detection and user-impacting failures against outcome-only monitoring.

**Worked Example (synthetic fixture):**
*Input.* Two root requests with one and three attempts. All attempts together consume 2,400 tokens; the final accepted attempts consume 800.
*Steps.* $A=(1+3)/2$. $A_{tok}=2{,}400/800$.
*Result.* $A=2$ attempts per root request and $A_{tok}=3$.
*Interpretation and limits.* Two thirds of the tokens bought nothing that was accepted: $1-1/A_{tok}=1-1/3=2/3$, i.e. 1,600 of 2,400 tokens. If a root request has no accepted output, its tokens stay in the numerator and the report must say how the denominator was handled; dropping the request hides the worst cases.

**Knowledge Check:**
1. Why can a complete manifest still fail to reproduce identical tokens?
2. Which latency formula applies to overlapping fallback probes?

**Independent Practice:**
(a) Three root requests use 1, 2, and 4 attempts and 5,600 tokens in total. Two requests end with an accepted output, and those accepted attempts used 1,400 tokens; the third exhausts its budget with nothing accepted. Report $A$, $A_{tok}$, and the zero-accepted count. (b) Change one serializer, adapter, schema subset, endpoint, and event shape at a time; use raw/effective diffs and fixtures to rank the failing boundary.

**Feedback Contract:**
- *Expected Evidence*: (a) $A=7/3\approx2.33$; $A_{tok}=5{,}600/1{,}400=4$; one of three root requests has no accepted output and is reported, not dropped. (b) Complete attempts, raw and normalized data, critical-path timing, all-attempt usage/cost, and a bounded reproducibility claim.
- *Common Failure*: Keeping only the final accepted response, or computing $A$ over the two successful requests only ($3/2$).
- *Diagnostic Hint*: What is the earliest representation that differs? Which requests left the denominator?
- *Concept to Revisit*: Replay Manifest and Attempt Ledger.

**Learning Outcome:**
Attribute regressions to request construction, target capability, provider behavior, stream assembly, normalization, validation, or recovery.

*(Effort: 35m instruction, 30m practice)*

---

## 05 Literature & Production Source Map

Reading status is stated per entry. *Opened 2026-10-01* means the page or file was read on that date for this revision. *Abstract-level* means only the abstract was read. Entries without such a note keep their 2026-09 access record in the registry and were not re-read.

**REFERENCE / BASELINE**

- [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12), [Validation specification](https://json-schema.org/draft/2020-12/json-schema-validation) — §6.2 (numeric keywords), §6.5.3 (`required`), §7 (format vocabularies). Opened 2026-10-01.
  - *Scope*: dialect and assertion semantics for Lesson 13.3 (**O**, CLM-005). It does not say which subset a provider or decoder implements.
- [HTML Standard, Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) — §9.2.5 (parsing an event stream), §9.2.6 (interpreting an event stream). Opened 2026-10-01.
  - *Scope*: the framing rules used by the Lesson 13.4 fixture (**O**, CLM-018). It is one wire format; other transports frame events differently.
- [Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning](https://arxiv.org/abs/2305.13971) — Geng et al., EMNLP 2023. Not re-read in this revision.
  - *Scope*: the reference mechanism of masking invalid continuations (**O**, CLM-006).
- [XGrammar](https://arxiv.org/abs/2411.15100) — Dong et al., MLSys 2025. Not re-read in this revision.
  - *Scope*: design of one constrained-decoding engine (**O**, CLM-014). Its reported performance is specific to its models, grammars, and hardware and is not used here.

**RECOMMENDED ENGINEERING BASELINE** (this module's derivation; earlier revisions labeled this list "CURRENT DEFAULT")

Canonical/effective request separation; versioned prompts, schemas, and adapters; explicit capability checks; preservation of raw protocol data; staged validation; stream state machines; typed errors; a bounded shared retry budget; complete attempt accounting.

These items follow from stated assumptions (**D**, CLM-001, CLM-002, CLM-004, CLM-007, CLM-009, CLM-010, CLM-012, CLM-013). They are a recommendation. They are not the default of any runtime, and this module has not measured how widely they are adopted.

**ONE IMPLEMENTATION'S DEFAULT** (observed, scoped to the named source)

- XGrammar at the pinned commit: `compile_json_schema` defaults to `strict_mode=True`, `any_whitespace=True`, `any_order=False` (**O**, CLM-015). This is one library at one revision.
- JSON Schema Draft 2020-12: `format` is an annotation unless the format-assertion vocabulary is in use (**O**, CLM-005). This is a specification default, not a statement about validators in the field.
- One hosted API's function-calling guide documents incremental argument events keyed by item and output index (**O**, CLM-019). Live page as served on 2026-10-01; no page revision date was shown. Other providers use other event taxonomies.

**INDUSTRY PREVALENCE:** not established by this module. The JSONSchemaBench abstract calls constrained decoding the dominant technology for enforcing structured outputs; that is the authors' framing in an abstract, not an adoption survey, and no survey was opened (`TODO_VERIFY`, CLM-020).

**WORKLOAD-DEPENDENT:** prompt format, schema complexity, constrained-decoding backend, repair strategy, semantic validator, retry count/backoff, fallback set (**D**, CLM-011), and retention/redaction policy.

**FRONTIER** (each scoped to what was read)

- [JSONSchemaBench](https://arxiv.org/abs/2501.10868) — arXiv v3, 2025-02-27. Abstract-level, opened 2026-10-01. It evaluates efficiency, coverage of constraint types, and output quality on about 10K real-world schemas plus the official JSON Schema Test Suite, across six constrained-decoding frameworks (**O**, CLM-008). No result figure from it is used in this module.
- [XGrammar-2](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation) — maintainer blog, 2026-05-04. Opened 2026-10-01. Structural tags for mixed free-form/structured protocols, batch APIs, and speculative-decoding integration (**O**, CLM-016). Maintainer-reported; not reproduced; its adoption is unknown.
- [When Punctuation Matters](https://aclanthology.org/2025.findings-emnlp.1109/) — Seleznyov et al., Findings of EMNLP 2025. Abstract-level, opened 2026-10-01. A comparison of prompt-robustness methods on eight models from three open model families over 52 Natural Instructions tasks (**O**, CLM-003). This is the only prompt-robustness source behind the earlier phrase "2025–2026 prompt-robustness methods"; no 2026 source on that topic was opened (`TODO_VERIFY`, CLM-020).

None of these establishes universal portability or zero overhead.

**LEGACY / INSUFFICIENT:** concatenate strings into a prompt; parse JSON with regex; trust “JSON mode” as schema or semantic correctness; assume OpenAI-shaped endpoints are semantically identical; parse every stream chunk independently; retry all exceptions; record only the final accepted response; treat a seed as deterministic replay.

**PRODUCTION SOURCE TRACE**

- Repository: `mlc-ai/xgrammar`
- Revision: `4221346f3d26b8306b46de19cb40c8a525c4871c`
- Verified: 2026-09-26; the three files and the symbols below were re-read at this revision on 2026-10-01. Static inspection only on both dates.
- Files/symbols: `python/xgrammar/compiler.py::GrammarCompiler.compile_json_schema`, `python/xgrammar/matcher.py::{GrammarMatcher.fill_next_token_bitmask,GrammarMatcher.accept_token}`, and `python/xgrammar/contrib/hf.py::LogitsProcessor.__call__`.
- Execution path: schema + tokenizer information → compiled grammar → matcher state → valid-next-token bitmask → logits masking → sampled token → matcher advance.
- Observed in the re-read: each of the three Python methods delegates to a native handle (`self._handle.…`); `fill_next_token_bitmask` is documented as not changing matcher state; `LogitsProcessor.__call__` creates one matcher per batch row on first use, skips acceptance on the first call, and moves scores to CPU for masking when the device is not CUDA.
- Scope: one pinned implementation (**O**, CLM-015). It is not proof of every schema feature, of any hosted provider, of semantic output, or of any benchmark speed. The native compiler and matcher behind the handles were not read.

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`.

### LAB A — Canonical Request and Adapter Conformance

- **Objective**: Build canonical request/response/event/error contracts and demonstrate semantic conformance across two deliberately different adapters.
- **Pre-Registered Hypothesis**: Capability probes will identify at least one incompatibility hidden by transport-level success before production outcome monitoring does (**H**, CLM-017).
- **Independent Variables**: Adapter, target, capability, serializer, and request fixture.
- **Dependent Variables**: Contract preservation, explicit degradation/rejection, drift detection, latency, and attempt cost.

- Define typed canonical request/response/event/error models and two deliberately different adapters.
- Build a capability registry and canonical fixtures for roles, content blocks, tools, output schemas, stops, usage, refusal, and errors.
- Break with unknown fields, silent defaults, prompt escaping, reordered examples, unsupported parameters, changed context limits, and false compatible fallbacks.
- Artifact: canonical/effective request diff with a per-field status (preserved, renamed, transformed, emulated, weakened, dropped, fabricated), capability evidence, and portability matrix.
- **Break & Falsify**: Inject unknown fields, silent defaults, escaping changes, unsupported parameters, and an incompatible fallback; any silent required-field loss falsifies conformance.
- **Alignment**: Lessons 13.1–13.2.
- **Effort Estimate**: 3h total.

### LAB B — Structured-Output Validation Ladder

- **Objective**: Compare post-hoc and constrained structured-output paths across schema features and semantic checks.
- **Pre-Registered Hypothesis**: Structural constraints will improve supported structural validity but will not eliminate semantic or task failures.
- **Independent Variables**: Schema feature, backend, constraint path, model/seed, and payload length.
- **Dependent Variables**: Compile/cache time, decode latency, feature coverage, parse/schema/semantic/task outcomes, tokens, and cost.

- Implement unconstrained JSON prompting, post-hoc parse/repair, schema validation, and constrained decoding.
- Cross schemas with enums, nested unions, recursion/references, optional/required fields, numeric/string constraints, `format`, additional/unevaluated properties, impossible/empty domains, and long tool sets.
- Measure compile time/cache, decode latency, feature coverage, parse/schema success, semantic/task quality, refusals, truncation, tokens, and cost.
- Break “valid JSON means correct”; trace the pinned XGrammar path and report unsupported semantics.
- Artifact: per-stage outcome table (parse, schema, invariant, authority) naming the first failing stage for every fixture, plus the feature-coverage list with unsupported keywords.
- **Break & Falsify**: Include impossible domains, unsupported keywords, and structurally valid semantic violations; a result where semantic errors remain falsifies equivalence of grammar and correctness.
- **Alignment**: Lesson 13.3.
- **Effort Estimate**: 3h total.

### LAB C — Streaming and Recovery Chaos

- **Objective**: Prove stream assembly and recovery semantics under legal chunking and injected protocol/error faults.
- **Pre-Registered Hypothesis**: Every tested chunk partition will yield the same completed objects, while ambiguous termination remains non-success under the declared protocol.
- **Independent Variables**: Chunking/order, event fault, error class, timeout, retry/fallback policy, and consumer speed.
- **Dependent Variables**: Assembly agreement, terminal-state correctness, attempts, latency, tokens/cost, and unsafe success.

- Replay identical logical responses under chunk splits, including fragmented Unicode and parallel tool arguments. Enumerate all $2^{n-1}$ partitions only for fixtures of about 16 bytes or fewer. For larger streams run all single cuts, all double cuts where affordable, one byte per chunk, and seeded random partitions.
- Separately vary how the same arguments are cut into deltas (delta-boundary invariance), and interleave two tool calls to show that fragments are never merged across keys.
- Inject missing terminal events, duplicate/out-of-order events, disconnect, cancellation race, 429/5xx, timeout, truncation, refusal, invalid schema, and semantic failure.
- Verify typed terminal states, stable assembly, absolute deadlines, shared attempt budgets, no nested retry multiplication, and compatible fallback only.
- Artifact: event-state machine, per-event assembly trace, chunk-test report (exhaustive versus sampled families, seed, partition count), fault matrix, attempt tree, and latency/cost amplification report.
- **Break & Falsify**: Inject missing terminal, duplication/order faults, disconnect, cancellation race, overload, truncation, refusal, invalid schema, and nested retries; any ambiguous success or budget escape falsifies the design.
- **Alignment**: Lessons 13.4–13.5.
- **Effort Estimate**: 3h total.

### LAB D — Replay and Drift Detection

- **Objective**: Operate versioned replay/conformance evidence and localize one-at-a-time harness drift.
- **Pre-Registered Hypothesis**: Raw/effective manifests plus targeted fixtures will localize seeded adapter drift earlier than end-outcome-only monitoring for at least one preregistered change.
- **Independent Variables**: Prompt/serializer, adapter, schema dialect, endpoint/model, and event shape.
- **Dependent Variables**: Detection delay, attribution accuracy, user-impacting escapes, false alerts, stage latency, and cost.

- Persist privacy-safe canonical/effective manifests, raw/normalized fixtures, validation results, usage, timing, and terminal status.
- Change prompt serializer, adapter version, schema dialect/subset, endpoint, model, and streaming event shape one at a time.
- Compare conformance probes with end-outcome monitoring for detection delay and user impact; include no-op and intentionally degraded changes.
- Artifact: replay manifest, amplification report ($A$, $A_{tok}$, zero-accepted policy), ranked regression attribution, canary policy, rollback evidence, and limits on reproducibility.
- **Break & Falsify**: Include no-op and intentionally degraded changes; failure to improve detection for the seeded set limits the hypothesis.
- **Alignment**: Lesson 13.6 and Incident 13.1.
- **Effort Estimate**: 3h total.

## 07 Break / Incident Scenarios

### Incident 13.1 — A “Compatible” Provider Migration Corrupts Tool Calls

- **Incident Symptoms**: A new endpoint passes smoke tests and lowers median latency, but streamed tool arguments intermittently fail parsing; retries amplify cost; fallback changes semantics; refusals become empty successes; usage disappears; and dashboards retain only final successes.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: Serializer drift, unsupported schema features, strictness/`format` differences, chunk assembly, tool-call identity, truncation, error normalization, nested retries, incompatible fallback, model drift, or survivorship bias.
  2. *Rank Initial Plausibility*: Use raw-event and effective-request differences without assuming the provider alone is causal.
  3. *Identify Missing Evidence*: Recover canonical/effective requests, versions, probes, dialect/subset, ordered raw events, terminal states, validation stages, all attempts/backoff, usage, spans, and fallback decisions.
  4. *Design Discriminating Tests*: Replay identical raw events through old/new assemblers, bypass normalization, disable retries/fallback, and run an independent semantic oracle; state falsifying outcomes.
  5. *Execute Causal Diagnosis*: Rank the earliest differing boundary and interacting retry/monitoring effects.
  6. *Prescribe Mitigation and Prevention*: Roll back incompatible routing, preserve raw failures, repair the earliest boundary, and gate targets with conformance probes.
  7. *Remeasure*: Schema coverage, semantic success, attempt/token amplification, latency, refusals/failures, and offered-request goodput.

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Portable Multi-Backend Model Gateway

Design a harness for two non-identical backends supporting text, tools, structured output, and streaming under one application contract.

**Required Deliverables**:
1. Canonical/effective request and raw/normalized response schemas.
2. Capability registry, probes, and degradation/rejection policy.
3. Versioned prompt/serializer fixtures and schema dialect/subset policy.
4. Validation ladder and pinned constrained-decoding source trace.
5. Streaming state machine and adversarial chunking evidence that states which split families were exhaustive and which were sampled.
6. Error taxonomy, shared retry/fallback budget, attempt ledger, and replay manifest.
7. Chaos/load evidence plus canary, kill, and rollback plan.
8. Evidence-backed diagnosis of Incident 13.1.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace

Submit a pinned trace of a constrained-decoding implementation. The reference trace uses XGrammar commit `4221346f3d26b8306b46de19cb40c8a525c4871c`. The trace must cover:

1. The entry point into schema compilation (`GrammarCompiler.compile_json_schema`), with the arguments and defaults in effect (`strict_mode`, `any_order`, whitespace).
2. How a compiled grammar becomes per-sequence matcher state.
3. Where the valid-next-token mask is produced (`fill_next_token_bitmask`) and where it is applied to logits.
4. Where the sampled token advances the matcher (`accept_token`), what happens when it is not accepted, and how termination is detected.
5. A list of what was not executed or verified: native code behind the Python handles, supported schema keywords, end-to-end generation, and any performance claim.

### Rubric Dimensions

- **Boundary and Portability**: *Insufficient* records one SDK object. *Competent* separates canonical/effective/raw/normalized forms and probes capabilities. *Strong* demonstrates drift localization and explicit degradation.
- **Structure and Streaming**: *Insufficient* equates JSON with correctness, validates a buffer because it parses, or merges fragments of different tool calls. *Competent* pins dialect/subset, names the first failing stage for each fixture, and implements terminal-aware assembly keyed by call identity. *Strong* shows chunk invariance exhaustively on small fixtures, reports sampled coverage honestly on large ones, and keeps layered validation correct under faults.
- **Recovery and Accounting**: *Insufficient* retries errors independently. *Competent* shares deadlines/budgets and records every attempt. *Strong* quantifies amplification and rejects incompatible fallback.
- **Diagnosis and Source Trace**: *Insufficient* blames a provider from symptoms. *Competent* traces a pinned path and compares raw/effective evidence. *Strong* uses discriminating replays, oracle stages, and remeasurement.

## 10 Capability Traceability Matrix

Deliverable numbers refer to Section 08, incident steps to Incident 13.1, and trace items to the Section 09 Required Artifact.

| Capability | Taught | Practiced | Assessed | Evidence (artifact reviewers open) |
|---|---|---|---|---|
| Canonical/effective model I/O contract | 13.1 field-diff table | 13.1 Guided Practice (a)–(b); LAB A | Mastery deliverable 1; Incident 13.1 steps 3, 5; rubric *Boundary and Portability* | LAB A canonical/effective request diff with per-field status |
| Capability negotiation and portability | 13.2 | 13.2 Independent Practice (a)–(b); LAB A; LAB D | Mastery deliverables 2–3; Incident 13.1 steps 1, 6; rubric *Boundary and Portability* | LAB A capability evidence and portability matrix; LAB D seeded-drift results |
| Structured output and validation ladder | 13.3 four-artifact ladder | 13.3 Guided Practice (a)–(b); LAB B | Mastery deliverable 4; Incident 13.1 step 4 (independent semantic oracle); rubric *Structure and Streaming* | LAB B per-stage outcome table and feature-coverage list |
| Pinned constrained-decoding source trace | 13.3 production source trace; Section 05 | LAB B (trace the pinned path) | Section 09 Required Artifact items 1–5; Mastery deliverable 4; rubric *Diagnosis and Source Trace* | Trace of `compile_json_schema`, `fill_next_token_bitmask`, `accept_token`, and `LogitsProcessor.__call__` at commit `4221346f…`, with the not-executed list |
| Stream assembly | 13.4 read/event/object trace | 13.4 Guided Practice (a)–(d); LAB C | Mastery deliverable 5; Incident 13.1 steps 3–4 (ordered raw events; replay through old and new assemblers); rubric *Structure and Streaming* | LAB C event-state machine, per-event trace, and chunk-test report with exhaustive/sampled statement |
| Error taxonomy and bounded recovery | 13.5 | 13.5 Guided Practice (a)–(c); LAB C | Mastery deliverables 6–7; Incident 13.1 steps 1, 4 (nested retries; retries and fallback disabled); rubric *Recovery and Accounting* | LAB C fault matrix and attempt tree under one root budget |
| Replay and attempt accounting | 13.6 | 13.6 Independent Practice (a)–(b); LAB D | Mastery deliverables 6, 8; Incident 13.1 steps 5, 7; rubric *Recovery and Accounting* | LAB D replay manifest, amplification report, ranked regression attribution |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner successfully completing Module 13 must be able to:
1. Preserve canonical intent, effective payloads, raw events, normalized output, and accepted outcomes.
2. Prove required capabilities and make unsupported/degraded behavior explicit.
3. Version prompts, schemas, serializers, adapters, and resolved targets.
4. Separate structural, semantic, authority, and effect validation.
5. Assemble streams by protocol state and fail closed on ambiguous termination.
6. Classify errors before bounded recovery and account for every attempt.
7. Trace pinned source without universalizing one implementation.

### Module Wrap-Up (Final Mental Model Reconstruction)

- **The Core Invariant**: The harness is a protocol compiler and evidence-preserving validator around an uncertain model endpoint.
- **The Data Path**: `application contract → canonical request → capability check → effective payload → raw events → assembly/normalization → validation → typed outcome`.
- Reliability comes from explicit capabilities, reversible transformations, staged validation, stateful streaming, bounded recovery, and complete attempt lineage.

### Cost & Telemetry Hook

- [Lesson 22.3](../22-ai-economics/README.md): record attempt-level usage, including retried, repaired, and rejected attempts, not only the accepted response.
- [Lesson 22.5](../22-ai-economics/README.md): a fallback or repair path is a cascade; cost each escalation step against the shared retry budget.
- [Lesson 23.1](../23-observability-reliability/README.md): emit canonical-vs-effective request diff, capability-check result, validation stage reached, and typed error per attempt; [Lesson 23.2](../23-observability-reliability/README.md) applies to partial or aborted streams.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
