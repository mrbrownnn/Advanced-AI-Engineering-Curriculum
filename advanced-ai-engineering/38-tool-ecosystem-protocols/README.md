# Module 38 — Tool Ecosystem & Protocols

## 00 Why This Module Exists
A production agent rarely calls tools that its own team wrote and deployed in the same process. It loads tools from many servers, written by other teams and vendors, through a wire protocol. Each server can change on its own schedule, speaks one of several protocol revisions, and may require its own OAuth grant. The model meets all of this only as text: a list of names, descriptions, and schemas that costs input tokens on every turn and decides which tool the model picks.

Three things go wrong at this boundary. More tools make every call more expensive and can make selection worse. A tool list that changes, reorders, or collides with another server's names breaks callers and caches without any error. An authorization setup that the specification calls mandatory can be optional in the SDK that implements it. This module treats the tool ecosystem as an engineering system: a protocol with a lifecycle, a catalog with a cost, a registry with versions, and an authorization profile with testable requirements.

```text
model context (paid every turn)
   ^  names + descriptions + schemas, in list order
   |
host / client ---- discovery, version, cache, subscription ----> MCP server(s)
   |                                                           |
   |-- selection: all tools | retrieved shortlist K            |-- tools/list (may vary by authorization)
   |-- pinning: registry version + hash of reviewed list       |-- tools/call -> result | isError | JSON-RPC error
   |-- auth: resource metadata -> token(resource=server) ---->  |-- audience check, scope check, step-up
```

**Why a module, not lessons in 12/13.** Lesson 12.2 owns what a tool contract contains: typed arguments, effect class, deadlines, the dispatch rule $D=V_{schema}\land V_{policy}\land V_{pre}$, and the observation envelope. Module 13 owns the model-I/O harness: capability negotiation (13.2), schema dialects and the validation ladder (13.3), and conformance fixtures for model endpoints (13.6). Module 18 owns adversarial tool descriptions, attack-success methodology, and the authority gateway (18.2, 18.4). None of them teaches a tool wire protocol, protocol versioning and era compatibility, tool-list caching and change notification, registries and manifest pinning, the economics of how many tools a model sees, or the OAuth profile that MCP servers use. Those topics share one object, the tool surface delivered across a protocol, and they interact: the ordering rule that protects the prompt cache is in the same specification section as the rule that lets the list vary by scope. Splitting them across three modules would teach each interaction twice or not at all.

This module owns protocol mechanics, discovery and caching, tool-surface economics, registry and versioning, OAuth plumbing, and conformance testing. It points back to Lesson 12.2 for contract semantics and effect verification and to Module 18 for attacks. No attack payloads appear here.

**Research cutoff:** 2026-11-10. Every source was opened on that date; abstract-only readings are marked in Section 05 and in the registry.

**Module Orientation**
- **Engineering Problem**: Expose the right tools to a model at a bounded context cost, keep the exposed surface stable and reviewable as servers change, and authorize each call to the right server with the least scope, all verified by tests rather than by specification text.
- **What You Will Do**: Measure selection accuracy against tool count and shortlist depth, build a local mock MCP server and client to test versioning, caching, subscriptions, and ordering, implement audience-bound tokens and step-up against a mock authorization server, write a specification-derived conformance suite and run it against a pinned SDK, diagnose a tool-surface incident, and design a tool gateway for an aggregated catalog.
- **Environment**: Python 3.10+ for the mock server, client, and auth server (LABs B–D run on CPU). LAB A needs a model: an open-weight model on Colab, or a hosted API you already have access to. No local GPU is assumed. Everything runs against local mocks; no production server or credential is used.
- **Evidence Rule**: Label source observations (**O**), explicit derivations (**D**), and telemetry-dependent hypotheses (**H**). A specification MUST is a requirement, not evidence that an implementation meets it. Vendor thresholds without a published method are labeled *vendor guidance*, never measurements.
- **If You Only Have One Week**: Lessons 38.2, 38.3, 38.5 — the protocol lifecycle and caching rules, the token budget and selection economics, and the authorization profile are the three places where a wrong default silently costs money, accuracy, or authority.

## 01 Baseline Assumptions
- **Tool contracts and effect evidence** (Module 12, Lessons 12.2, 12.3, 12.5): typed tool contracts, effect classes, the dispatch predicate, the `EFFECT_UNKNOWN` state after a timeout, and verifying a postcondition before retrying a write.
- **Harness contracts** (Module 13, Lessons 13.2, 13.3, 13.6): required versus verified capability sets, JSON Schema dialects and the validation ladder, and conformance fixtures that probe a contract instead of trusting an adapter label.
- **Context accounting and prefix caching** (Module 11, Lesson 11.1; Module 03, Lesson 3.6): every token in the request occupies context; an exact-prefix cache reuses work only up to the first differing token.
- **Retrieval metrics** (Module 09, Lesson 9.6): recall@K and why a reranker cannot recover an item that retrieval did not return.
- **Evaluation design** (Module 15, Lessons 15.1, 15.4): experimental unit, paired comparisons, and uncertainty before a release decision.
- **Threat model and authority** (Module 18, Lessons 18.2, 18.4): tool-description poisoning as an attack channel, the confused deputy, and authorization from inputs the attacker cannot write.
- **Durable idempotency** (Module 14, Lesson 14.2): idempotency keys and at-most-once effects across retries.

OAuth roles, protected resource metadata, resource indicators, and token audience are not taught earlier. Lesson 38.5 teaches them at first use.

## 02 Target Mastery

```yaml
depth_contract:
  conceptual: REQUIRED
  mechanistic: REQUIRED
  mathematical: SELECTIVE
  quantitative: REQUIRED
  implementation: REQUIRED
  source_code: REQUIRED
  instrumentation: REQUIRED
  experimental: REQUIRED
  statistical: SELECTIVE
  production_reasoning: REQUIRED
  failure_analysis: REQUIRED
  falsification: REQUIRED
  security: REQUIRED
  economics: SELECTIVE
  architecture_tradeoff: REQUIRED
  research_connection: SELECTIVE

estimated_effort:
  instruction: 4.5h      # sum of lesson instruction estimates: 40+45+50+40+45+50 min
  guided_practice: 3h    # sum of in-lesson practice: 30+30+30+30+30+30 min
  labs: 12h              # LAB A 3h + LAB B 3h + LAB C 3h + LAB D 3h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 38.1 0.5h
  source_trace: 1.5h     # Lesson 38.6 guided trace and the Section 09 Production Source Trace artifact, counted once here
  total: 24h
```
Each category is counted once. The pinned python-sdk trace is counted under `source_trace`, not again under Lesson 38.6 practice or LAB D. Lab analysis is not counted again as guided practice.

Security is REQUIRED because the module configures authorization and token handling. The scope is defensive: audience binding, scope minimization, passthrough prohibition, and conformance tests. Attack construction stays in Module 18.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map
Tool contract (Module 12) $\to$ **Description and schema as the selection interface** (38.1) $\to$ **Protocol lifecycle: stateless requests, discover, version errors, era compatibility** (38.2) $\to$ **List semantics: authorization-dependent sets, deterministic order, TTL and scope hints, change subscriptions** (38.2) $\to$ **Context cost and selection: tokens per tool, shortlist depth, chance-corrected depth** (38.3) $\to$ **Registry, pinning, collisions, schema evolution** (38.4) $\to$ **Authorization: resource metadata, resource indicator, audience, scope, step-up** (38.5) $\to$ **Conformance: specification requirement $\to$ wire test $\to$ deployed configuration** (38.6) $\to$ Cost and telemetry (Modules 22, 23).

Keep five objects apart throughout:

1. **The protocol revision** a request speaks (2026-07-28 or a legacy one).
2. **The registry entry**: immutable metadata for one published server version.
3. **The live tool list**: what `tools/list` returns now, for this authorization.
4. **The model-visible surface**: the subset of tools serialized into this model call, in this order.
5. **The token**: issued for one resource, with a set of scopes, for a lifetime.

## 04 Lessons

### Lesson 38.1 — Tool Descriptions as API Contracts

**Engineering Question:**
Lesson 12.2 says that natural-language descriptions help model selection but are not an execution contract. If the description is not the contract, what is it, and how should an engineer decide whether a better description is worth its cost?

**Concepts & Definitions:**
- **Selection interface**: the part of a tool definition the model reads to decide whether to call it and how to fill arguments: name, description, and input schema. One vendor's documentation states that the model decides based on the user's request and the tool's description, calls description detail by far the most important factor in tool performance, and recommends at least 3–4 sentences. That is vendor guidance without a published method (**O**, CLM-024).
- **Execution contract**: the Lesson 12.2 contract enforced outside the model (typed validation, authority, effect class, deadlines). The selection interface influences which contract gets invoked; it never grants or limits authority.
- **Tool definition fields (MCP 2026-07-28)**: `name`, `title`, `description`, `icons`, `inputSchema` (root `type: object`, JSON Schema 2020-12 by default), optional `outputSchema`, and `annotations`. A name SHOULD be 1–128 characters from `[A-Za-z0-9_.-]` and is unique only within a server (**O**, CLM-007).
- **Structured output**: with an `outputSchema`, a server MUST return conforming structured content and clients SHOULD validate it; for backward compatibility the tool SHOULD also return serialized JSON as text (**O**, CLM-010).
- **Annotations**: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`. The schema calls them hints and says clients should never make tool-use decisions based on annotations from untrusted servers (**O**, CLM-008).
- **Description smell**: a defect in a description against a rubric, such as an unclear purpose. One study of 856 tools on 103 servers reports at least one smell in 97.1% of tools and an unclear purpose in 56%, using an LLM-based scanner (**O**, CLM-034).
- **Two error kinds**: a protocol error (JSON-RPC error: unknown tool, malformed request) versus a tool execution error (`isError: true`, which includes input-validation failures). Clients SHOULD return execution errors to the model so it can correct itself (**O**, CLM-009).

**Mechanism Explanation:**
State → operation → state, for one tool across its life:
1. *Authoring*: a developer writes name, description, and schema. In the pinned python-sdk the description defaults to the function docstring, so an undocumented function ships an empty description (**O**, CLM-043).
2. *Listing*: the server returns the definition in `tools/list`. The client serializes it into the model request; the definition is now input tokens on every call (**O**, CLM-024).
3. *Selection*: the model ranks the visible tools against the request. Description text is a ranking signal written by whoever writes the description. An abstract-level study reports that edited descriptions received more than 10 times more usage against competing tools across its tested models (**O**, CLM-037). Module 18 treats the adversarial version of this; here it means that description edits change traffic between your own tools too.
4. *Argument filling and validation*: the model fills arguments; the server validates them. A validation failure is an execution error returned to the model, not a protocol failure.
5. *Effect classification*: before dispatch, the client applies Lesson 12.2's effect class. That class cannot come from `readOnlyHint` or `idempotentHint` on an untrusted server; it comes from a reviewed client-side contract, and a mismatching annotation is a review flag (**D**, CLM-063).

Improving descriptions does help on average in published work, and it is not monotone. On MCP-Universe, augmented descriptions raised median task success by 5.85 percentage points and partial completion by 15.12%, while raising execution steps by 67.46% and regressing 16.67% of cases; compact ablations kept reliability with fewer tokens (**O**, CLM-034). Abstract-level studies report smell remediation effects of +11.6% and +8.8% for two dimensions and better robustness from learned rewrites in large catalogs (**O**, CLM-035, CLM-036). Whether a rewrite pays off on your workload is a hypothesis to measure (**H**, CLM-069).

**Quantitative Model / Derivation:**
Adopt a description change only if its value exceeds its added cost, per episode (**D**, CLM-056):
$$\Delta s\cdot V \;>\; \Delta n_{steps}\cdot c_{step} \;+\; \Delta C_{tools}\cdot c_{tok}\cdot n_{turns}$$

*Where*: $\Delta s$ = change in verified success probability per episode, measured on paired episodes (regressions included); $V$ = value of one verified success (USD); $\Delta n_{steps}$ = change in execution steps per episode; $c_{step}$ = cost of one step (USD); $\Delta C_{tools}$ = added definition tokens per model call (summed over all loaded tools); $c_{tok}$ = input price (USD per token); $n_{turns}$ = model calls per episode.

*Step-by-step*:
1. Benefit per episode is the extra successes times their value: $\Delta s\cdot V$.
2. Extra steps cost $\Delta n_{steps}\cdot c_{step}$.
3. Definitions are re-sent on every model call, so added description tokens cost $\Delta C_{tools}\cdot c_{tok}$ per call and $n_{turns}$ times per episode.
4. Adopt when benefit minus both costs is positive.

*Assumptions*: costs add per episode; $\Delta s$ comes from a paired comparison with stated uncertainty (Module 15, Lesson 15.4); latency is not priced separately.

**Worked Example:**
*Inputs* (all synthetic, registry CLM-057): 30 loaded tools; $V=2.00$ USD; $c_{step}=0.004$ USD; baseline 6 steps; $n_{turns}=10$; $c_{tok}=3\times10^{-6}$ USD.

| Variant | $\Delta s$ | $\Delta n_{steps}$ | Added tokens per tool | $\Delta C_{tools}$ per call |
|---|---:|---:|---:|---:|
| Full rewrite | +0.06 | +4.0 | 150 | 4,500 |
| Compact rewrite | +0.04 | +1.2 | 50 | 1,500 |

*Full rewrite*:
- Benefit: $0.06\times2.00=0.120$ USD.
- Step cost: $4.0\times0.004=0.016$ USD.
- Token cost: $4{,}500\times3\times10^{-6}\times10=0.135$ USD.
- Net: $0.120-(0.016+0.135)=-0.031$ USD. Reject.

*Compact rewrite*:
- Benefit: $0.04\times2.00=0.080$ USD.
- Step cost: $1.2\times0.004=0.0048$ USD.
- Token cost: $1{,}500\times3\times10^{-6}\times10=0.045$ USD.
- Net: $0.080-0.0498=+0.0302$ USD. Adopt.

*Interpretation and limits.* The larger success gain lost because its definition tokens are paid on every call of every episode. The decision flips with $n_{turns}$: at 5 turns the full rewrite's token cost halves to 0.0675 and its net becomes $0.120-0.0835=+0.0365$. Prompt caching lowers $c_{tok}$ for the cached prefix, which favors longer descriptions (Lesson 38.3). The +4.0 steps echoes the order of the published step increase but is not taken from it.

**Knowledge Check:**
1. A server marks `delete_records` with `readOnlyHint: true`. What does the client do with that annotation, and where does the effect class come from?
2. A rewrite raises median success but regresses one case in six. Why can the median hide a reason not to ship it?
3. A tool's argument fails schema validation on the server. Is that a JSON-RPC error or a tool result, and why does the difference matter to the model?

**Guided Practice:**
Take three tools you own. For each, write (a) the description you would ship, (b) the Lesson 12.2 contract fields that the description must not be trusted to enforce, and (c) the inputs you need to evaluate the adoption rule for a rewrite: which quantities you can measure, and on what paired episode set.

**Feedback Contract:**
- *Expected Evidence*: Knowledge Check 1: the annotation is a hint from an untrusted source; the client ignores it for authorization, uses its reviewed effect class (destructive write), and flags the mismatch for review. Check 2: a median says nothing about tail regressions; a one-in-six regression on a high-value slice can outweigh the gain, so report the paired distribution and regressions. Check 3: a tool result with `isError: true`; the model sees it and can correct the arguments, whereas a protocol error indicates a malformed request or unknown tool. Practice: for each tool, a contract table with validation, authority, effect class, and deadline separate from the description, plus a paired evaluation plan.
- *Common Failure*: Treating "longer description" as free, or deriving effect class from `readOnlyHint`.
- *Diagnostic Hint*: Which of your numbers is paid once, and which is paid on every model call?
- *Concept to Revisit*: Proposal–Authorization–Effect Separation (Lesson 12.2).

**Learning Outcome:**
Write a tool's selection interface separately from its execution contract and decide on description changes with a cost-adjusted, paired measurement.

*(Effort: 40m instruction, 30m practice)*

---

### Lesson 38.2 — MCP Architecture and the 2026-07-28 Revision

**Engineering Question:**
What does a client need to know about a server before its first tool call under the current MCP revision, and which assumptions from older revisions now break?

**Concepts & Definitions:**
- **Host, client, server**: MCP uses JSON-RPC 2.0. A host is an LLM application that initiates connections, a client is a connector inside the host, and a server offers Resources, Prompts, and Tools. The architecture page states that the full conversation history stays with the host and that the host enforces security boundaries. The current revision is 2026-07-28; the previous one is 2025-11-25 (**O**, CLM-001).
- **Stateless request**: 2026-07-28 removes the `initialize`/`initialized` handshake and the `Mcp-Session-Id` header. Each request carries `_meta` keys `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities`. Servers MUST implement `server/discover`. A version mismatch returns `UnsupportedProtocolVersionError` (code −32022) with `data.supported`. `ping` and `logging/setLevel` are removed (**O**, CLM-002).
- **Result types and round trips**: SSE resumability (`Last-Event-ID`) is removed, so a broken stream means the request is re-issued with a new id. Every result has a `resultType` of `complete` or `input_required`. Multi Round-Trip Requests replace server-initiated sampling, elicitation, and roots requests. Tasks moved to the extension `io.modelcontextprotocol/tasks` (**O**, CLM-003).
- **Era**: *modern* means 2026-07-28; *legacy* means 2025-11-25 or earlier with the handshake. Modern client to legacy server fails; legacy client to modern server fails because legacy clients have no fall-forward mechanism; dual-era implementations work both ways. Clients SHOULD cache the era per process (stdio) or per origin (HTTP) (**O**, CLM-004).
- **List semantics**: `tools/list` returns the tools currently available to the requesting client. The set MUST NOT vary per connection but MAY vary by the authorization presented. Servers SHOULD return a deterministic order, for client caching and LLM prompt cache hit rates (**O**, CLM-005).
- **Cache hints and subscriptions**: list results carry `ttlMs` and `cacheScope` (`public` or `private`); a private result MUST NOT be shared across authorization contexts. `notifications/tools/list_changed` goes only to clients that opened `subscriptions/listen` with `toolsListChanged: true` (**O**, CLM-006).
- **Deprecations**: Roots, Sampling, and Logging; the HTTP+SSE transport; Dynamic Client Registration (in favour of Client ID Metadata Documents). The minimum deprecation window is 12 months (**O**, CLM-011).

**Mechanism Explanation:**
Client state machine for one server under 2026-07-28:
1. *Unknown era*: the client sends `server/discover` with its protocol version in `_meta`.
2. *Discovered*: the server returns supported versions, capabilities, and instructions. In the pinned python-sdk, `_handle_discover` returns the modern versions and capabilities computed at call time, and `listChanged` is advertised only when a `subscriptions/listen` handler is served (**O**, CLM-046). The client records the era per origin.
3. *Version error*: on −32022 the client reads `data.supported`, chooses a common version, or marks the server incompatible. It does not guess by comparing date strings. The pinned SDK keeps versions in enumerated tuples and treats an unknown version string as not at least any known version (**O**, CLM-042).
4. *Listed*: `tools/list` returns the set for this authorization, with `ttlMs`/`cacheScope`. The client caches it under (origin, authorization context) when the scope is private.
5. *Subscribed* (optional): `subscriptions/listen` with `toolsListChanged: true`. A change notification invalidates the cache entry.
6. *Calling*: each `tools/call` carries the version and capabilities again. If the stream breaks, the client must issue a new request with a new id. For a write whose effect is unknown, that is a new call, not a resume: verify the postcondition first or carry an idempotency key in the arguments (**D**, CLM-062).

**Trade-off Comparison** (where each protocol sits):

| | MCP 2026-07-28 | Provider function calling | OpenAPI 3.2.1 | A2A 1.0 |
|---|---|---|---|---|
| Boundary | host/client to tool server | application to model API | HTTP API description | agent to opaque agent |
| Who lists tools | server, per authorization | application, per request | API description document | Agent Card at `/.well-known/agent-card.json` |
| Identity of an operation | tool name, unique per server | function name per request | `operationId`, unique per API, case-sensitive | skill in the Agent Card |
| Schema default | JSON Schema 2020-12 | provider dialect; strict mode needs `additionalProperties: false` and every property required | `jsonSchemaDialect`, 2020-12 by default | `a2a.proto` is normative |
| Version signal | `_meta` per request; −32022 | API version of the provider | `openapi` field | `A2A-Version` header; empty means 0.3 |

Sources: MCP (**O**, CLM-002, CLM-005, CLM-007); one provider's strict mode and its normalization fallback (**O**, CLM-028); OpenAPI (**O**, CLM-030); A2A purpose, bindings, discovery, and versioning only (**O**, CLM-029). The A2A topic page's "MCP is vertical, A2A is horizontal" is vendor framing and is not used as evidence. An OpenAPI-to-MCP mapping can reuse `operationId` as a tool name inside one server but needs a server prefix when APIs are merged, and a 2020-12 parameter schema may still fail a provider's strict subset (**D**, CLM-075). One provider also supports deferred tool loading and namespaces that group tools by domain, with detailed guidance in the function description (**O**, CLM-027).

**Quantitative Model / Derivation:**
Deterministic ordering has a direct token consequence under an exact-prefix cache (Module 03, Lesson 3.6). If the serialized tool list precedes the conversation, the reusable prefix ends at the first tool that differs between two requests (**D**, CLM-059):
$$P_{reuse} = c_0 + \sum_{i=1}^{j-1} t_i$$

*Where*: $c_0$ = the fixed tool-use system-prompt tokens; $t_i$ = tokens of tool $i$ in list order; $j$ = position of the first tool that differs between the two requests; $P_{reuse}$ = tokens that can be served from the cache.

*Assumptions*: the provider cache matches exact token prefixes; tools are serialized in list order before messages. Check each provider's breakpoints and minimum cacheable lengths separately.

**Worked Example:**
*Inputs* (synthetic): $c_0=286$ tokens; 20 tools of 1,000 tokens each, so a full tool block of $286+20\times1{,}000=20{,}286$ tokens.
- Deterministic order: both requests serialize tools 1–20 identically, so all 20,286 tokens are reusable.
- A server that returns tools in an order that varies per request swaps tools 3 and 4: $j=3$, so $P_{reuse}=286+2\times1{,}000=2{,}286$ tokens, $2{,}286/20{,}286=11.3\%$ of the block.
- A change in tool 15's description only: $j=15$, $P_{reuse}=286+14{,}000=14{,}286$, about 70%.

*Interpretation and limits.* The specification's SHOULD on ordering protects 88.7 points of reuse in this example. The pinned python-sdk lists tools in registration order, which is stable inside one process but can differ between two deployments that register in a different order (**O**, CLM-044). Conversation tokens after the tool block are unaffected by tool order, but nothing after the first difference is reusable.

**Knowledge Check:**
1. A legacy client connects to a 2026-07-28-only server. What happens, and which side must change?
2. Why must a cache entry for a `private` tool list be keyed by authorization context?
3. A `tools/call` for `send_invoice` loses its stream. Why is re-issuing it not a resume, and what does Lesson 12.3 require first?

**Guided Practice:**
Write the client's era table for three servers: one modern, one legacy, one dual-era. For each, give the first request, the expected response, the cache key, and the action on −32022. Then compute $P_{reuse}$ for your own tool list if one server inserts a new tool at position 2.

**Feedback Contract:**
- *Expected Evidence*: Check 1: the connection fails; legacy clients have no fall-forward, so the client must upgrade or the server must be dual-era. Check 2: the list MAY differ by authorization, so a shared entry could show one caller another caller's tools and the specification forbids sharing private results. Check 3: there is no session or resumable stream; the new request has a new id and the server cannot link it, so the effect is unknown and the postcondition must be checked or an idempotency key used. Practice: a table with discover/handshake behavior, era cache key (process or origin), and a reuse calculation whose $j=2$.
- *Common Failure*: Comparing protocol version strings as dates, or caching one tool list per server for every user.
- *Diagnostic Hint*: Which request carries the version now that there is no handshake?
- *Concept to Revisit*: Capability Negotiation (Lesson 13.2); Unknown Effect (Lesson 12.3).

**Learning Outcome:**
Drive an MCP 2026-07-28 client through discovery, version errors, caching, and subscriptions, and place MCP correctly against provider function calling, OpenAPI, and A2A.

*(Effort: 45m instruction, 30m practice)*

---

### Lesson 38.3 — Tool Count, Tool Retrieval, and Context Cost

**Engineering Question:**
How many tools should a model see on one call, what does each one cost, and when does retrieving a shortlist beat loading everything?

**Concepts & Definitions:**
- **Tool surface cost**: definitions are billed as input tokens and count against the context window on every call; one provider injects them into the system message (**O**, CLM-024, CLM-027). One vendor lists a fixed tool-use overhead on top of the definitions, for example 286 tokens for Opus 5.5 with `tool_choice` auto (**O**, CLM-024).
- **Measured per-server cost**: one vendor's engineering report gives GitHub 35 tools ≈26K tokens, Slack 11 ≈21K, Sentry 5 ≈3K, Grafana 5 ≈3K, Splunk 2 ≈2K (58 tools ≈55K), and has seen 134K tokens of definitions before optimization. It names wrong tool selection and incorrect parameters as the most common failures, especially with similar names (**O**, CLM-026).
- **Vendor guidance on count** (not measurements): one vendor says to aim for fewer than 20 functions at the start of a turn, calls it a soft suggestion, and says to evaluate with different numbers of functions (**O**, CLM-027). Another says selection degrades past 30–50 tools and that tool search typically cuts definition tokens by over 85% (**O**, CLM-025). Neither publishes a method. Where the knee lies for your model and catalog is a hypothesis to measure (**H**, CLM-067).
- **Tool retrieval**: select a shortlist of $K$ tools from $N$ with a retriever, then show only those. One vendor's `defer_loading` keeps a tool out of context until searched; search returns 5 by default (**O**, CLM-025).
- **Bits over random (BoR)**: for a shortlist of $K$ from $N$ with $R_q$ relevant tools, $P_{rand}=1-\binom{N-R_q}{K}/\binom{N}{K}$ ($=K/N$ when $R_q=1$) and $\mathrm{BoR}=\log_2(P_{obs}/P_{rand})$, with maximum $\log_2(N/K)$ for one relevant tool (**O**, CLM-032).
- **BFCL** (Berkeley Function Calling Leaderboard): a function-calling benchmark with AST, live, multi-turn, agentic, and format-sensitivity categories, including irrelevance categories; unevaluated categories count as 0 in summary columns (**O**, CLM-040).

**Mechanism Explanation:**
A call is correct only if three things happen in sequence: the gold tool reaches the model, the model chooses it, and the arguments are valid. Loading everything makes the first certain and the second harder. Retrieval does the reverse. The measured evidence shows both sides:
- On RAG-MCP's web-search subset (one model, 20 trials, LLM judge), showing all descriptions gave 13.62% accuracy at 2,133.84 prompt tokens; a keyword pre-filter 18.20% at 1,646; retrieval then show 43.13% at 1,084. Its stress test shows success above 90% for targets at positions below 30 and failures dominating past ~100 (read from a figure) (**O**, CLM-031). The paper conflates server and tool selection.
- On BFCL (370 tools), a learned depth policy reached 90.3% coverage with about 7 tools against 90.8% with 50. On ToolBench (3,251 tools), fixed $K=5$ had higher aggregate coverage (64.7% vs 61.9%) but found nothing on hard queries. Downstream with one model, the adaptive policy reached 93.1% vs 87.1% for always-5 (**O**, CLM-033).
- An abstract-level filter study reports improved selection as tools grow, with no numbers (**O**, CLM-038). Earlier baselines built retrievers over 16k APIs; one reports that the prior evaluation protocol gave an artificially high pass rate (**O**, CLM-039).

Retrieval moves the scaling problem into recall: at fixed $K$, the chance that the gold tool is shortlisted can fall as $N$ grows even if choice among shown tools holds (**H**, CLM-070). Benchmarks are instruments too: an abstract-level paper reports that the BFCL multi-turn scorer ignores should-ask turns and that one prompt line moved a model's official score by 13.5–23.5 points without changing decision accuracy (**O**, CLM-041).

**Quantitative Model / Derivation:**
*Context cost* (**D**, CLM-051):
$$C_{tools}=c_0+\sum_{i\in\text{loaded}}(t_{name,i}+t_{desc,i}+t_{schema,i}),\qquad f=\frac{C_{tools}}{W}$$

*Tokens-per-tool budget* (**D**, CLM-052). Given a share $f$ of window $W$ reserved for definitions:
$$n_{max}=\left\lfloor\frac{fW-c_0}{\bar t}\right\rfloor$$

*Retrieval–selection decomposition* (**D**, CLM-053):
$$P(\text{correct})=P(\text{gold}\in S_K)\cdot P(\text{choose gold}\mid\text{gold shown},K)\cdot P(\text{args valid}\mid\text{gold chosen})$$

*Where*: $c_0$ = fixed tool-use overhead (tokens); $t_{\cdot,i}$ = tokens of each part of tool $i$; $W$ = context window (tokens); $f$ = fraction of $W$ given to definitions; $\bar t$ = mean definition tokens per loaded tool; $S_K$ = retrieved shortlist of size $K$; $N$ = catalog size.

*Step-by-step for the budget*:
1. Reserve $fW$ tokens for the tool block.
2. Subtract the fixed overhead $c_0$, which is paid once per call whatever the tool count.
3. Divide by the mean per-tool cost and round down, because a partial tool cannot be loaded.
4. Check: $n_{max}$ scales as $1/\bar t$, so a mean from a few large tools overstates how many small ones fit and vice versa. For a mixed catalog, add the actual costs of the chosen tools.

*Step-by-step for the decomposition*: the three events happen in order, so the chain rule gives the product. Adding tools to a shortlist cannot remove the gold tool, so the first factor never decreases with $K$; the second factor's dependence on $K$ is empirical.

**Worked Example:**
*Step 1 — per-tool cost from the reported figures* (**D**, CLM-051): $55{,}000/58=948.3$ tokens per tool on average; per server: GitHub $26{,}000/35=742.9$, Slack $21{,}000/11=1{,}909.1$, Sentry and Grafana $3{,}000/5=600$, Splunk $2{,}000/2=1{,}000$. The mean hides a 3.2× spread.

*Step 2 — share of a window.* With a declared $W=200{,}000$ (an exercise value), 55K is $f=0.275$; the reported 134K is $f=0.67$. Uncached, a 20-turn session re-sends $20\times(286+55{,}000)=1{,}105{,}720$ tokens.

*Step 3 — budget at $f=0.10$*: $fW-c_0=20{,}000-286=19{,}714$ tokens. Then $n_{max}=\lfloor19{,}714/948.3\rfloor=20$; GitHub-like $\lfloor19{,}714/742.9\rfloor=26$; Slack-like $\lfloor19{,}714/1{,}909.1\rfloor=10$. With the 200-tokens-per-description figure from the depth paper, $\lfloor19{,}714/200\rfloor=98$, but that figure excludes schemas and is not comparable (**D**, CLM-052).

*Step 4 — chance-corrected depth on BFCL* (**D**, CLM-054), $N=370$, one gold tool:
- $K=7$, $P_{obs}=0.903$: $P_{rand}=7/370=0.0189$; $\mathrm{BoR}=\log_2(0.903\times370/7)=5.58$ bits of a maximum $\log_2(370/7)=5.72$.
- $K=50$, $P_{obs}=0.908$: $P_{rand}=0.135$; $\mathrm{BoR}=\log_2(0.908\times370/50)=2.75$ bits of a maximum 2.89.
- Coverage differs by 0.5 points; the smaller shortlist carries about twice the information per tool shown, and at Step 1's mean it costs $7\times948\approx6.6$K tokens instead of 47.4K.

*Step 5 — the RAG-MCP trade* (**D**, CLM-055): prompt tokens fell by $1-1{,}084/2{,}133.84=49.2\%$ and accuracy rose by $43.13/13.62=3.17\times$ on that testbed. The abstract's "over 50%" compares against unspecified baselines.

*Step 6 — decomposition with illustrative factors* (synthetic, not measured): shortlist coverage 0.903, choice 0.95, arguments 0.97 gives $0.903\times0.95\times0.97=0.832$; coverage 0.908 with choice 0.90 gives $0.793$. A 0.5-point coverage gain is outweighed by a 5-point choice loss.

*Interpretation and limits.* The vendor thresholds and the per-server token counts are approximate and unmethodical; they size the problem, they do not set your $K$. Whether the choice factor actually falls between $K=7$ and $K=50$ on your model is the LAB A measurement.

**Knowledge Check:**
1. Why does the 948-token mean overstate how many Sentry-sized tools fit in a 20K budget, and understate how many Slack-sized tools take?
2. Fixed $K=5$ beats an adaptive policy on aggregate ToolBench coverage. Why might you still reject it?
3. A team quotes "fewer than 20 functions" as a measured limit. What is wrong with the citation, and what replaces it?

**Guided Practice:**
For a catalog of your own (or the Mastery fixture's 240 tools at 948 tokens), compute $C_{tools}$ for loading everything, $f$ at $W=200$K, and $n_{max}$ at $f=0.10$. Then choose two shortlist depths and write which of the three decomposition factors each one trades, and what you would measure in LAB A to choose between them.

**Feedback Contract:**
- *Expected Evidence*: Check 1: Sentry tools cost 600 tokens, so 32 fit, not 20; Slack tools cost 1,909, so 10 fit; a mean from mixed servers misallocates the budget. Check 2: aggregate coverage hides that hard queries get nothing; slice by difficulty before deciding. Check 3: it is vendor guidance, a soft suggestion without a method; replace it with a measured knee on your catalog. Practice: $C_{tools}=286+240\times948=227{,}806$ tokens, $f=1.14$ (it does not fit at all), $n_{max}=20$; two depths with recall@K, choice accuracy, and tokens as dependent variables.
- *Common Failure*: Comparing the 200-token description figure with full definition cost; reporting aggregate coverage only.
- *Diagnostic Hint*: Which factor of the decomposition does your change move, and which does it leave to luck?
- *Concept to Revisit*: Recall@K (Lesson 9.6); Context Accounting (Lesson 11.1).

**Learning Outcome:**
Derive a tokens-per-tool budget and a shortlist depth for a catalog, report chance-corrected depth, and treat vendor tool-count thresholds as guidance to replace with measurement.

*(Effort: 50m instruction, 30m practice)*

---

### Lesson 38.4 — Registry, Discovery, and Versioning

**Engineering Question:**
When a client pins a server version from a registry, what exactly is pinned, and how does it keep the model-visible tool surface stable and collision-free as servers evolve?

**Concepts & Definitions:**
- **MCP Registry**: the official centralized metadata repository for publicly accessible MCP servers, currently in preview, where breaking changes or data resets may occur. It hosts metadata, not packages; namespaces are DNS-verified; private servers are not supported (host your own registry instead); aggregators are expected to pull infrequently, for example once per hour; and it defines an OpenAPI specification that other registries can implement. A live sample returned a `server.json` with a 2025-12-11 schema and a `streamable-http` remote (**O**, CLM-022).
- **Immutable registry version**: a `version` string is required, MUST be unique per publication, and once published neither it nor its other metadata can change. SemVer is recommended, not required; a non-SemVer version is always marked latest (**O**, CLM-023).
- **Live tool list**: what `tools/list` returns now. It can change at any time and MAY vary by authorization (**O**, CLM-005).
- **Manifest pin**: a hash of the canonical tools/list result that was reviewed, per authorization context.
- **Name collision**: two servers exposing the same tool name. Names are unique only within a server; aggregators SHOULD disambiguate, for example by a server-id prefix, and `serverInfo.name` is not guaranteed to be unique (**O**, CLM-007). An abstract-level study reports "73% repeated tool names" across 10,831 servers; its definition was not audited (**O**, CLM-035).
- **Schema compatibility**: input-compatible if every old valid argument object is still valid; output-compatible if every new valid result is still valid for old consumers (**D**, CLM-076).
- **Cache defaults in one SDK**: the pinned python-sdk `CacheHint` defaults to `ttl_ms=0` and `cache_scope="private"`; handler values win per field; input-required results get no hints (**O**, CLM-048).

**Mechanism Explanation:**
Two lifetimes meet at the client:
1. *Registry*: version $v$ of server X has fixed metadata. Pinning $v$ fixes where the server is and what it claims to be.
2. *Server*: at request time, X returns a tool list $L(t, a)$ that depends on time $t$ and authorization $a$. The registry does not version $L$.
3. *Client*: pins $v$ **and** a hash $h(L, a)$ of the reviewed list. On every refresh it recomputes $h$; a changed hash means the model would see unreviewed text, so the client refuses or quarantines the change until review. This is the Module 18 manifest-pinning control applied to the protocol objects (**D**, CLM-064).
4. *Aggregation*: tools from many servers merge into one namespace. If keys are bare names and the policy keeps one, $s$ identically named tools leave $s-1$ unreachable. In the pinned python-sdk the same thing happens inside one server: a duplicate registration logs a warning and returns the existing tool (**O**, CLM-043). Prefixing with a client-assigned server id removes collisions if the ids are unique and the result stays within 128 characters (**D**, CLM-060).
5. *Refresh*: by TTL or by subscription. A subscription bounds staleness by notification latency rather than by $T$ (**H**, CLM-068).

**Quantitative Model / Derivation:**
For tool-list changes arriving as a Poisson process with rate $\lambda$ and a client that re-fetches every $T$ seconds, the probability that a read at a uniformly random time sees a stale list is (**D**, CLM-058):
$$p_{stale}=1-\frac{1-e^{-\lambda T}}{\lambda T}$$

*Where*: $\lambda$ = changes per second; $T$ = TTL in seconds; $u$ = time since the last fetch, uniform on $[0,T]$.

*Step-by-step*:
1. A read at time $u$ after a fetch is stale if at least one change happened in $[0,u]$: probability $1-e^{-\lambda u}$.
2. Average over $u$ uniform on $[0,T]$: $\frac1T\int_0^T(1-e^{-\lambda u})\,du=1-\frac{1}{T}\cdot\frac{1-e^{-\lambda T}}{\lambda}$.
3. For small $\lambda T$, $p_{stale}\approx\lambda T/2$: half the expected changes per interval.

*Assumptions*: constant-rate Poisson changes; fetch latency negligible; reads uniform in time. Deployments are bursty, so treat the formula as a sizing aid.

**Worked Example:**
*Inputs* (synthetic): $T=300$ s.

| Change rate | $\lambda T$ | $p_{stale}$ |
|---|---:|---:|
| 1 per day | 0.00347 | 0.17% |
| 1 per hour | 0.0833 | 4.05% |
| 1 per 10 min | 0.5 | 21.3% |

Check the small-$x$ approximation on the second row: $0.0833/2=4.17\%$, close to 4.05%.

*Schema-change classification* (**D**, CLM-076), for a tool `search(query, limit)`:

| Change | Input-compatible? | Why |
|---|---|---|
| Add optional `lang` | yes | every old argument object is still valid |
| Raise `limit` maximum from 50 to 100 | yes | accepted set widens |
| Make `lang` required | **no** | old callers omit it |
| Narrow `sort` enum from {date, score} to {score} | **no** | old `date` calls now fail |
| Add required `next_cursor` to `outputSchema` | yes for consumers that allow extra properties | new results still satisfy old schema unless it sets `additionalProperties: false` |

*Interpretation and limits.* At one change per day a 5-minute TTL is nearly always fresh; at one change per 10 minutes a fifth of reads are stale and a subscription is worth its stream. The registry pin would not change in any of these rows, because a live description edit does not publish a new registry version. Only the manifest hash changes.

**Knowledge Check:**
1. You pin version 1.4.0 of a server in the registry. A week later its `search` description changes. What detects it?
2. Three servers expose `search`. Your aggregator keeps the first. What does the merged list look like, and why is there no error?
3. Why must the manifest hash be computed per authorization context?

**Guided Practice:**
Design the pinning record for one server: registry name and version, era, authorization context, canonical serialization rule (sorted keys, list order), hash, reviewer, and the action on a hash change. Then classify three real or invented schema changes from a tool you use as input- or output-compatible.

**Feedback Contract:**
- *Expected Evidence*: Check 1: only a manifest hash over the live `tools/list` result; the registry version is immutable and unchanged. Check 2: one `search` that silently belongs to the first server; the list is schema-valid, so nothing fails until a call lands on the wrong backend. Check 3: the list MAY differ by authorization, so one hash cannot describe what each caller sees. Practice: a record with all fields and a refuse-or-quarantine policy; classifications with the widen/narrow argument stated.
- *Common Failure*: Treating a registry version as a pin on tool text; keying merged tools by bare name.
- *Diagnostic Hint*: Which of the five objects in Section 03 did each control actually fix?
- *Concept to Revisit*: Manifest pinning (Lesson 18.4 table); Schema dialect (Lesson 13.3).

**Learning Outcome:**
Separate immutable registry metadata from live tool lists, pin what the model sees, prevent name collisions, and classify schema changes by compatibility direction.

*(Effort: 40m instruction, 30m practice)*

---

### Lesson 38.5 — Authorization and Per-Tool Scoping

**Engineering Question:**
How does an MCP client get a token that only one server will accept, with only the scopes it needs, and what must the server check so that the specification's guarantees hold in a deployment?

**Concepts & Definitions:**
- **OAuth roles in MCP**: authorization is OPTIONAL; HTTP transports SHOULD conform and stdio SHOULD NOT (stdio takes credentials from the environment). The profile follows OAuth 2.1. The MCP server is an OAuth *resource server* (it accepts tokens); the MCP client is an OAuth *client* (it obtains them from an *authorization server*) (**O**, CLM-013).
- **Protected Resource Metadata (RFC 9728)**: a document at `/.well-known/oauth-protected-resource` that tells clients which authorization server protects a resource; a `resource_metadata` parameter in `WWW-Authenticate` points to it (**O**, CLM-021). MCP servers MUST publish it and clients MUST use it to find the authorization server (**O**, CLM-013).
- **Resource indicator (RFC 8707)**: the `resource` request parameter naming the server the token is for; the authorization server SHOULD audience-restrict tokens accordingly (**O**, CLM-021). MCP clients MUST send it, set to the server's canonical URI, in both the authorization and token requests, even if the authorization server ignores it (**O**, CLM-014).
- **Audience binding**: servers MUST validate that a token was issued for them and MUST NOT accept or transit any other token; clients MUST NOT send other tokens; tokens never go in query strings and go on every HTTP request (**O**, CLM-015).
- **Token passthrough**: forwarding a token received from a client to another service. The maintainer best-practices page calls it an explicitly forbidden anti-pattern; risks are bypassed controls, lost audit attribution, and crossed trust boundaries (**O**, CLM-019).
- **Scope selection and step-up**: initially use the `scope` from the 401 challenge; if absent, all of `scopes_supported`. At runtime a 403 `insufficient_scope` with `scope=` triggers step-up; clients SHOULD request the union of granted and challenged scopes. The specification says `scopes_supported` is meant to be the minimal set for basic functionality (**O**, CLM-016).
- **Issuer check**: authorization servers SHOULD send the RFC 9207 `iss` parameter, and clients MUST validate it when present; credentials are keyed by issuer (**O**, CLM-017).
- **OAuth-proxy confused deputy**: a narrow case on the best-practices page that needs four conditions together (static client ID at a third-party authorization server, dynamic registration at the proxy, a consent cookie at the third party, no per-client consent at the proxy); proxies MUST implement per-client consent (**O**, CLM-018). The general agent-as-deputy case is Lesson 18.4.

**Mechanism Explanation:**
1. *Challenge*: the client calls the server without a token and gets 401 with `resource_metadata` and maybe `scope`.
2. *Discovery*: the client reads the protected resource metadata and finds the authorization server.
3. *Grant*: the client requests authorization with `resource=<server canonical URI>` and the challenged scope (or `scopes_supported`), validates `iss` if present, and exchanges the code for a token with the same `resource`.
4. *Call*: every HTTP request carries the token. The server checks expiry, **audience**, and scope.
5. *Step-up*: a tool needing more scope returns 403 `insufficient_scope` with the needed scope; the client re-authorizes with the union and retries.
6. *Downstream*: if the server calls another API, it uses its own credential for that API. It never forwards the client's token.

`tools/list` MAY return only the tools the caller's scopes permit (**O**, CLM-005), so scope also shapes the model-visible surface. Servers MUST validate inputs, enforce access control, rate-limit, and sanitize outputs; clients SHOULD confirm sensitive operations and keep a human who can deny invocations in the loop (**O**, CLM-012).

The pinned python-sdk shows where these guarantees depend on configuration:
- `BearerAuthBackend` rejects expired tokens and, only when given a `resource_server_url`, rejects a token whose verifier-reported resource differs. `RequireAuthMiddleware` returns 401 `invalid_token` or 403 `insufficient_scope` with `resource_metadata`, and its required scopes are set per HTTP app; no per-tool scope gate appeared in the files read (**O**, CLM-049). Whether one exists elsewhere is unverified (`TODO_VERIFY`, registry CLM-074).
- `AuthSettings.validate_token_resource` defaults to `None`; with a `resource_server_url` set this warns and behaves as False, so the middleware performs no audience check unless the setting is on or the token verifier checks the audience itself. The field description says 3.0 makes True the default (**O**, CLM-050).

The specification's no-challenge fallback and the best-practices warning against listing every scope are consistent only if servers keep `scopes_supported` minimal; a server that lists all scopes turns the fallback into an omnibus first grant (**D**, CLM-065). Best practices recommend a minimal starting set and incremental elevation, and list wildcard scopes and trusting token scope claims without server-side authorization logic as common mistakes (**O**, CLM-020).

**Quantitative Model / Derivation:**
Use scope-seconds as a simple exposure proxy (**D**, CLM-061):
$$E=\sum_{k}|S_k|\cdot\Delta t_k$$

*Where*: $k$ indexes intervals during which the granted scope set is constant; $|S_k|$ = number of scopes granted in interval $k$; $\Delta t_k$ = interval length in seconds while the token is valid.

*Step-by-step*:
1. A stolen or misused token can exercise any granted scope until it expires or is revoked.
2. Exposure grows with both how much is granted and for how long, so multiply and sum over intervals.
3. Step-up keeps $|S_k|$ small until a scope is needed, at the price of one extra authorization round trip per elevation.

*Assumptions*: scopes are weighted equally (weight them by impact for real use); no revocation before expiry. The measure is a proxy for blast radius, not a risk score.

**Worked Example:**
*Inputs* (synthetic): a server defines 4 scopes; token lifetime 3,600 s; the task needs `files:read` from the start and `files:write` from 1,800 s.
- *All of `scopes_supported` at first grant*: $E=4\times3{,}600=14{,}400$ scope-seconds.
- *Challenged scope, then step-up to the union*: $E=1\times1{,}800+2\times1{,}800=5{,}400$ scope-seconds, $14{,}400/5{,}400=2.67\times$ less, for one extra round trip.
- *Audience*: the client requests the token with `resource=https://files.example/mcp`. A second server at `https://tickets.example/mcp` receives that token by mistake. With audience validation it rejects it; with the pinned SDK's default and a verifier that does not check audience, the token would be accepted.

*Interpretation and limits.* Minimizing scope halves exposure here only if `scopes_supported` itself is minimal; if the server lists all four and sends no `scope` in its challenge, the specification's fallback reproduces the first row. Audience validation is the control that makes a token useless at the wrong server; scope minimization limits what it can do at the right one. Neither stops a correctly scoped token from being used for the wrong task, which is Module 18's gateway.

**Knowledge Check:**
1. The authorization server ignores the `resource` parameter. Which check still prevents a token for server A from working at server B?
2. Your MCP server needs to call a ticketing API on the user's behalf. Why can it not forward the user's MCP token, and what does it use instead?
3. A client receives 403 `insufficient_scope` with `scope="files:write"` while holding `files:read`. What does it request, and why the union?

**Guided Practice:**
For one server you would deploy, write: the canonical resource URI; the contents of its protected resource metadata; `scopes_supported` (minimal); which tools need which scopes and whether `tools/list` filters by scope; the 401 and 403 responses; the audience check and where it runs (middleware or verifier); and the setting values for the pinned python-sdk that make the audience check happen.

**Feedback Contract:**
- *Expected Evidence*: Check 1: the MCP server's own audience validation (MUST), which is independent of the authorization server. Check 2: passthrough is forbidden (bypassed controls, lost attribution, crossed boundary); the server uses its own credential for the downstream API, obtained under its own authority. Check 3: the union `files:read files:write`, so the re-issued token does not lose a scope needed for earlier operations. Practice: a minimal scope list, a tool-to-scope table, 401/403 examples with `resource_metadata`, and either `validate_token_resource=True` with `resource_server_url`, or a verifier that checks `aud`, stated explicitly.
- *Common Failure*: Assuming the SDK enforces the specification's audience MUST by default; publishing every scope in `scopes_supported`.
- *Diagnostic Hint*: Which component would reject a token issued for a different server, and is it on in your configuration?
- *Concept to Revisit*: Confused Deputy (Lesson 18.4); Authority outside the model (Lesson 12.5).

**Learning Outcome:**
Configure MCP authorization so that tokens are audience-bound, never passed through, minimally scoped, and elevated by step-up, and verify each property in the deployed configuration.

*(Effort: 45m instruction, 30m practice)*

---

### Lesson 38.6 — Conformance and the Pinned SDK Trace

**Engineering Question:**
How do you prove that a tool server or client meets the protocol requirements you rely on, rather than assuming it does because it uses an official SDK?

**Concepts & Definitions:**
- **Conformance test**: a request whose expected wire response is fixed by a specification requirement, run against a *deployed configuration*. It extends Lesson 13.6's conformance fixtures from model endpoints to tool servers.
- **Configuration-dependent conformance**: an implementation meets a MUST only for some settings. A suite must assert observable wire behavior, because a passing reference configuration says nothing about the deployed one (**D**, CLM-066).
- **Fail-open path**: a check that, when it cannot run, lets the request through.
- **Static versus executed evidence**: reading source establishes that code exists; only a test establishes what a deployed build does.

**Mechanism Explanation:**
Trace of `modelcontextprotocol/python-sdk` at commit `77001c96f9507be0f4d009b83210a384a5e08bda` (main, newer than release v2.3.0). Static reads on 2026-11-10; nothing executed.

| Specification requirement | Pinned SDK behavior | Conformance consequence |
|---|---|---|
| Unknown tool is a protocol error (JSON-RPC) (**O**, CLM-009) | `ToolManager.call_tool` raises `ToolError("Unknown tool: …")`; `_handle_call_tool` re-raises only `MCPError` and returns every other exception as `CallToolResult(is_error=True)` (**O**, CLM-045) | Test the wire response: on this path an unknown tool arrives as an isError result, which a client may feed to the model as a correctable error |
| Input-validation failures are tool execution errors (**O**, CLM-009) | Pydantic `ValidationError` becomes `ToolError`, then an isError result; logs record field names only (**O**, CLM-045) | Matches; assert `isError: true` and that values are absent from logs |
| Tool names SHOULD match `[A-Za-z0-9_.-]{1,128}` and are unique per server (**O**, CLM-007) | Names are validated with a warning only; a duplicate name keeps the first registration and logs a warning (**O**, CLM-043) | Test that a second registration of the same name does not silently replace, and that your build fails on invalid names if you require it |
| Deterministic `tools/list` order (SHOULD) (**O**, CLM-005) | Registration order from a dict; pagination params ignored (**O**, CLM-044) | Deterministic per process; test order across two deployments of the same build |
| Versions negotiated by enumerated list (**O**, CLM-002) | Enumerated version tuples; unknown versions compare conservatively; `server/discover` returns modern versions and call-time capabilities (**O**, CLM-042, CLM-046) | Test −32022 with an unknown version and the era of a legacy request |
| Audience MUST (**O**, CLM-015) | Checked by middleware only when `validate_token_resource` is truthy; default `None` warns and skips (**O**, CLM-050) | Send a token for another resource; the default configuration accepts it unless the verifier checks audience |
| Header values validated against the input schema | `get_tool_input_schema` runs before middleware, is not scoped to the caller, and if it raises, the call is served unvalidated (**O**, CLM-047) | Inject a raising lookup; the call goes through |

The three divergences are a hypothesis until executed: unknown tool as isError, wrong-audience token accepted under defaults, and header validation skipped on lookup failure (**H**, CLM-071). Benchmark scores carry the same caveat at the model level: unevaluated BFCL categories count as zero and one scorer ignores should-ask turns (**O**, CLM-040, CLM-041).

**Quantitative Model / Trade-off Comparison:**
A suite's claim is a conjunction over tested requirements $r$ for one configuration $\kappa$:
$$\text{Conformant}(\kappa)=\bigwedge_{r\in R_{tested}}\big[\text{observed}(r,\kappa)=\text{expected}(r)\big]$$

*Where*: $R_{tested}$ = the requirements with a test; $\kappa$ = the deployed configuration (SDK commit, settings, verifier, middleware order); observed and expected are wire-level responses.

Two consequences: a requirement without a test is outside the claim, so report coverage $|R_{tested}|/|R_{relied}|$, where $R_{relied}$ is the set your design depends on; and a change of $\kappa$ (an SDK upgrade that flips a default, as announced for 3.0) invalidates the result until re-run.

**Worked Example:**
*Input.* A team relies on 8 requirements: era handling, −32022, deterministic order, private cache scope, unknown-tool error, validation-as-isError, audience binding, and insufficient_scope. Their suite tests 6; it has no audience test and no cache-scope test. Configuration: pinned SDK with default `AuthSettings` and a verifier that returns the token's resource but does not check it.

*Steps.*
1. Coverage: $6/8=75\%$. The claim "conformant" covers 6 requirements only.
2. Static reading predicts that the unknown-tool test fails against the specification's expectation (isError instead of a JSON-RPC error).
3. The untested audience requirement is the one most likely to fail silently: under this configuration a token for another resource is accepted.
4. Adding the audience test and setting `validate_token_resource=True` with `resource_server_url` is predicted to turn that test from fail to pass.

*Result.* Before: 5 of 6 tests pass, coverage 75%, with an unknown audience behavior. After: 8 tests, the audience test passes under the hardened configuration, the unknown-tool divergence is documented and handled in the client.

*Interpretation and limits.* Steps 2–4 are predictions from static reading (**H**, CLM-071). LAB D runs them. The client side of the SDK, its list caching, and the TypeScript SDK were not inspected (`TODO_VERIFY`, registry CLM-074).

**Knowledge Check:**
1. Why is "we use the official SDK" not evidence of audience validation?
2. The unknown-tool response is an isError result in one implementation. Is that a security problem, a correctness problem, or neither? What would you test on the client?
3. An SDK upgrade changes a default. Which conformance results remain valid?

**Guided Practice:**
Trace, at the pinned commit, one `tools/call` from `MCPServer._handle_call_tool` to `ToolManager.call_tool` and back for (a) an unknown tool and (b) an argument that fails validation. Then trace how `MCPServer` constructs `BearerAuthBackend` and state which setting decides whether `_issued_for_this_resource` runs. Write which statements are static observations and which need LAB D to confirm. This trace is the `source_trace` effort.

**Feedback Contract:**
- *Expected Evidence*: Check 1: the requirement is met only under a setting that defaults off, or by a verifier you write; test the deployed configuration. Check 2: a correctness and conformance divergence; it can let the model "self-correct" toward a tool that does not exist; the client should treat a name not in its pinned list as a client-side error before dispatch. Check 3: none for the changed behavior; re-run the suite under the new configuration. Practice: a call path naming `_handle_call_tool`, `call_tool`, `ToolManager.call_tool`, `ToolError`, and `CallToolResult(is_error=True)`, and the construction `resource_server_url=... if validate_token_resource else None`, each marked static.
- *Common Failure*: Reporting a code reading as a passed test, or citing a docstring as runtime proof.
- *Diagnostic Hint*: For each claim, can you name the request you sent and the bytes you got back?
- *Concept to Revisit*: Conformance fixtures and replay (Lesson 13.6).

**Learning Outcome:**
Translate specification requirements into wire-level conformance tests, run them against the deployed configuration, and report coverage and configuration-dependent divergences in a pinned SDK.

*(Effort: 50m instruction, 30m practice. The pinned trace is counted once, under `source_trace`.)*

---

## 05 Literature & Production Source Map

All entries were opened on 2026-11-10. *Abstract-only* means only the abstract was read; no methods, denominators, or baselines were audited, and no section numbers are given. Every number below is author- or vendor-reported; none was reproduced.

**REFERENCE / BASELINE**
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728.txt) — Jones, Hunt, Parecki, 2025; [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707.txt) — Campbell, Bradley, Tschofenig, 2020 (**O**, CLM-021).
  - *Scope*: metadata discovery and the resource parameter for Lesson 38.5. RFC 9207 and OAuth 2.1 were not opened; their roles are taken from the MCP specification's text.
- [OpenAPI Specification 3.2.1](https://github.com/OAI/OpenAPI-Specification/blob/4f86429f134f7e48b6351563549b4502fc69de52/versions/3.2.1.md) at `4f86429f134f7e48b6351563549b4502fc69de52` (**O**, CLM-030).
  - *Scope*: `operationId` uniqueness and the 2020-12 dialect default, for the Lesson 38.2 contrast. It contains nothing LLM-specific.
- Gorilla/APIBench (arXiv:2305.15334v1), ToolLLM/ToolBench (arXiv:2307.16789v2), AnyTool (arXiv:2402.04253v1) — abstract-only (**O**, CLM-039).
  - *Scope*: retrieval over large API sets as a baseline idea, and one finding that an older protocol inflated pass rates.
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard) README and TEST_CATEGORIES at `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (**O**, CLM-040).
  - *Scope*: category definitions for LAB A. No leaderboard number was read.

**CURRENT DEFAULT** (each scoped to the named specification or vendor; adoption not surveyed)
- [MCP specification, revision 2026-07-28](https://github.com/modelcontextprotocol/modelcontextprotocol/tree/632a0c9a65c9b4fbc889d8aface5c9cf5edef3b7/docs/specification/2026-07-28) at `632a0c9a65c9b4fbc889d8aface5c9cf5edef3b7`: `index`, `changelog`, `basic/versioning`, `server/tools`, `server/discover`, `basic/authorization/index`, and `schema/2026-07-28/schema.ts` (**O**, CLM-001, CLM-002, CLM-003, CLM-004, CLM-005, CLM-006, CLM-007, CLM-008, CLM-009, CLM-010, CLM-011, CLM-012, CLM-013, CLM-014, CLM-015, CLM-016, CLM-017).
  - *Scope*: normative intent. It is not evidence that any client or server enforces it. Many requirements are SHOULD-level. The `draft` revision and full SEP texts were not read.
- [MCP Security Best Practices](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/632a0c9a65c9b4fbc889d8aface5c9cf5edef3b7/docs/docs/2026-07-28/tutorials/security/security_best_practices.mdx) (2026-07-28 docs tree; maintainer guidance, Tier 2) (**O**, CLM-018, CLM-019, CLM-020).
  - *Scope*: OAuth-proxy confused deputy, passthrough, and scope minimization. Other sections were read at heading level only.
- Claude API tool-use docs ([overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview), define-tools, [tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)) and Anthropic Engineering, [Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use), 2025-11-24 (**O**, CLM-024, CLM-025, CLM-026).
  - *Scope*: one vendor. Thresholds (30–50 tools, >85% reduction) and the internal accuracy numbers are **vendor guidance without a published method**.
- OpenAI [Function calling guide](https://developers.openai.com/api/docs/guides/function-calling) (**O**, CLM-027, CLM-028).
  - *Scope*: one vendor. "Fewer than 20 functions" is a soft suggestion by the vendor's own wording.
- [A2A Protocol specification](https://github.com/a2aproject/A2A/blob/9fff3e08ab33bb7674e7d98fc7c08ae8592bb4bd/docs/specification.md) at `9fff3e08ab33bb7674e7d98fc7c08ae8592bb4bd`, version 1.0.0 (**O**, CLM-029).
  - *Scope*: purpose, bindings, discovery, and versioning only. Security and push-notification sections were not read. Multi-agent design belongs to Module 37.

**WORKLOAD-DEPENDENT** (this module's derivations; measure before relying on them): tokens-per-tool budget and its sensitivity to per-tool spread (**D**, CLM-051, CLM-052); shortlist depth and the retrieval–selection decomposition (**D**, CLM-053); description-rewrite adoption (**D**, CLM-056); TTL staleness (**D**, CLM-058); prefix reuse under reordering (**D**, CLM-059); scope exposure (**D**, CLM-061). The selection knee, subscription freshness, rewrite effects, and retrieval recall under growth are hypotheses (**H**, CLM-067, CLM-068, CLM-069, CLM-070).

**FRONTIER**
- [MCP Registry](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/632a0c9a65c9b4fbc889d8aface5c9cf5edef3b7/docs/registry/about.mdx) about and versioning docs; preview (**O**, CLM-022, CLM-023).
  - *Scope*: registry semantics. The registry repository (head `d39ac76b…`, release v1.8.8) and `server.schema.json` were not read; one live API request was sampled.
- RAG-MCP, Gan & Sun, [arXiv:2505.03275v1](https://arxiv.org/abs/2505.03275v1), 2025, HTML full text, Table 1 (**O**, CLM-031).
  - *Scope*: one base model, small testbed, LLM judge; conflates server and tool selection.
- "How Many Tools Should an LLM Agent See?", Repantis et al., [arXiv:2605.24660v2](https://arxiv.org/abs/2605.24660v2), 2026, HTML full text (**O**, CLM-032, CLM-033).
  - *Scope*: shortlist presentation, not execution; small policies; single-seed MetaTool (author-stated).
- "MCP Tool Descriptions Are Smelly!", Hasan et al., [arXiv:2602.14878v3](https://arxiv.org/abs/2602.14878v3), 2026, HTML full text (**O**, CLM-034).
  - *Scope*: LLM-based scanner; dataset from prior studies (authors' §8.1).
- Abstract-only: "From Docs to Descriptions", Wang et al., [arXiv:2602.18914v1](https://arxiv.org/abs/2602.18914v1) (**O**, CLM-035); Trace-Free+, Guo et al., [arXiv:2602.20426v2](https://arxiv.org/abs/2602.20426v2) (**O**, CLM-036); "Tool Preferences in Agentic LLMs are Unreliable", Faghih et al., [arXiv:2505.18135v2](https://arxiv.org/abs/2505.18135v2) (**O**, CLM-037); JSPLIT, Antonioni et al., [arXiv:2510.14537v1](https://arxiv.org/abs/2510.14537v1) (**O**, CLM-038); "Asking Earns Nothing", Liu & Han, [arXiv:2610.04429v1](https://arxiv.org/abs/2610.04429v1) (**O**, CLM-041).
  - *Scope*: pointers and effect sizes as stated in abstracts. Nothing in this module depends on their methods.

**LEGACY**
- MCP 2025-11-25 and earlier: the `initialize` handshake, `Mcp-Session-Id`, SSE resumability, server-initiated sampling/elicitation/roots, object-only tool schemas, and the HTTP+SSE transport (**O**, CLM-002, CLM-003, CLM-007, CLM-011). Supported through dual-era implementations during the deprecation window; not a design target.
- Practices this module rejects: loading every tool on every call without a budget; trusting annotations from untrusted servers; pinning only a registry version; keying merged tools by bare name; forwarding client tokens downstream; publishing every scope in `scopes_supported`.

**PRODUCTION SOURCE TRACE**
- Repository: `modelcontextprotocol/python-sdk`
- Revision: `77001c96f9507be0f4d009b83210a384a5e08bda` (main, 2026-11-05; newer than release v2.3.0 of 2026-10-02)
- Verified: 2026-11-10, static inspection only; nothing executed.
- Files/symbols:
  - `src/mcp/server/mcpserver/server.py::{MCPServer._handle_list_tools, MCPServer._handle_call_tool, MCPServer._tool_input_schema}` and the auth middleware construction around L1199–1202.
  - `src/mcp/server/mcpserver/tools/tool_manager.py::{ToolManager.add_tool, ToolManager.call_tool}`; `tools/base.py::Tool.from_function`.
  - `src/mcp/server/lowlevel/server.py::{Server.get_capabilities, Server._handle_discover}` and the `get_tool_input_schema` docstring (L431–437).
  - `src/mcp/server/caching.py::{CacheHint, apply_cache_hint}`.
  - `src/mcp/server/auth/middleware/bearer_auth.py::{BearerAuthBackend.authenticate, BearerAuthBackend._issued_for_this_resource, RequireAuthMiddleware}`; `src/mcp/server/auth/settings.py::AuthSettings.validate_token_resource`.
  - `src/mcp-types/mcp_types/version.py::{KNOWN_PROTOCOL_VERSIONS, MODERN_PROTOCOL_VERSIONS, is_version_at_least}`.
- Execution paths: `tools/call` → `_handle_call_tool` → `call_tool` → `ToolManager.call_tool` → `ToolError` | `Tool.run` → `CallToolResult`; HTTP request → `BearerAuthBackend.authenticate` → optional resource check → `RequireAuthMiddleware` scope loop.
- Observed: Lesson 38.6 table (**O**, CLM-042, CLM-043, CLM-044, CLM-045, CLM-046, CLM-047, CLM-048, CLM-049, CLM-050).
- Generalizable: the error taxonomy pattern and the RFC 8707 audience check are general; the defaults, duplicate handling, and fail-open lookup are specific to this SDK at this commit.
- Not inspected: the client side (`client/session.py`), other auth modules, and the TypeScript SDK (`TODO_VERIFY`, registry CLM-074).

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow:
$$\text{PREDICT} \to \text{BUILD} \to \text{MEASURE} \to \text{EXPLAIN} \to \text{BREAK} \to \text{IMPROVE} \to \text{FALSIFY}$$
Every lab runs against local mocks with synthetic catalogs, synthetic tokens, and reserved names (`.example`). Record a manifest: protocol revision, SDK commit, configuration, model and its revision (LAB A), seeds, and catalog hash. No production server or credential is used.

### LAB A — Tool Count, Shortlist Depth, and Description Cost
- **Objective**: Measure selection accuracy and token cost as a function of loaded tools $N$, shortlist depth $K$, and description variant, and locate the selection knee for one model and two catalogs.
- **Pre-Registered Hypothesis**: Accuracy stays within a declared tolerance up to a catalog-specific $N$ and then falls, earlier for a catalog with near-duplicate names (**H**, CLM-067). At fixed $K$, recall@K falls as $N$ grows while choice accuracy among shown tools holds (**H**, CLM-070). Compact rewrites keep at least half of a full rewrite's gain at under half its tokens (**H**, CLM-069).
- **Independent Variables**: $N\in\{5,10,20,40,80,160\}$; $K\in\{3,7,15,50\}$ with a BM25 or embedding retriever; catalog (distinct names vs near-duplicate names); description variant (original, compact, full).
- **Dependent Variables**: recall@K, choice accuracy given gold shown, argument validity, end-to-end accuracy, $C_{tools}$ per call, BoR, steps, and paired regressions. Include BFCL-style irrelevance queries (no correct tool) (**O**, CLM-040).
- **Break & Falsify**: (1) Flat accuracy across all $N$ for both catalogs falsifies the knee hypothesis for this model. (2) Recall flat while end-to-end accuracy falls places the loss in choice, falsifying the retrieval explanation. (3) Recompute the adoption rule (**D**, CLM-056) from measured deltas; a rewrite that wins on success and loses on the rule must be reported as rejected.
- **Alignment**: Lessons 38.1 and 38.3.
- **Hardware**: COLAB-GPU for an open-weight model arm; CPU for the retrieval-only arm (recall@K needs no model). A hosted API you already use may replace the Colab arm; record model and date.
- **Effort Estimate**: 2h implementation and runs, 1h analysis (3h total).

### LAB B — Mock MCP Server: Versions, Caching, Subscriptions, Ordering
- **Objective**: Build a local mock MCP 2026-07-28 server and client (or use the pinned python-sdk server) and test discovery, version errors, era handling, cache hints, subscriptions, ordering, and pinning.
- **Pre-Registered Hypothesis**: A subscribed client's stale-read fraction is far below the TTL model's at frequent changes (**H**, CLM-068); measured TTL staleness matches $p_{stale}$ within sampling error (**D**, CLM-058).
- **Independent Variables**: protocol version in `_meta` (modern, legacy, unknown); change rate $\lambda\in\{1/\text{day},1/\text{hour},1/10\text{min}\}$ (time-compressed); TTL $T$; subscription on/off; server ordering (deterministic, shuffled); authorization context for `private` lists; schema changes from the Lesson 38.4 table.
- **Dependent Variables**: −32022 responses with `data.supported`; era cache hits; stale-read fraction; notify-to-refresh latency; shared prefix length of serialized tool blocks between consecutive calls (**D**, CLM-059); manifest-hash changes detected; compatibility classification per schema change; outcome of a re-issued write after a dropped stream (**D**, CLM-062).
- **Break & Falsify**: (1) Shuffle order per request and measure prefix collapse; a stable prefix under shuffling means your serializer sorts and the test is invalid. (2) Change a description without changing the registry version; failure to detect it with the manifest hash falsifies the pin. (3) Share a `private` list across two authorization contexts and show the client refuses (**O**, CLM-006). (4) Drop the stream of a non-idempotent mock write after commit; any duplicate effect falsifies your retry policy.
- **Alignment**: Lessons 38.2 and 38.4.
- **Hardware**: CPU.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB C — Audience-Bound Tokens, Scope Minimization, and Step-Up
- **Objective**: Implement a mock authorization server and two mock MCP resource servers; demonstrate audience binding, passthrough refusal, minimal initial scopes, and step-up.
- **Pre-Registered Hypothesis**: With audience validation on, a token issued for server A is rejected by server B in every trial; step-up from a minimal set reduces scope-seconds against an all-scopes grant by the ratio predicted for the declared task timeline (**D**, CLM-061).
- **Independent Variables**: authorization server honors `resource` (yes/no) (**O**, CLM-014); server audience check on/off, including the pinned SDK's `validate_token_resource` unset/True (**O**, CLM-050); `scopes_supported` minimal vs full (**D**, CLM-065); challenge with or without `scope`; `iss` present/absent/wrong (**O**, CLM-017).
- **Dependent Variables**: accept/reject decisions per (token audience, server); 401/403 bodies including `resource_metadata` and `scope` (**O**, CLM-049); granted scope sets over time and scope-seconds; downstream calls made with a forwarded token (must be zero) (**O**, CLM-019).
- **Break & Falsify**: (1) Turn off both the authorization server's audience restriction and the server's check; the cross-server token must be accepted, proving the harness detects the failure. (2) Publish all scopes and send no `scope` in the challenge; the first grant must be the full set (the fallback). (3) Any accepted wrong-issuer response falsifies the client's `iss` handling.
- **Alignment**: Lesson 38.5.
- **Hardware**: CPU.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB D — Specification-Derived Conformance Suite and Pinned SDK Run
- **Objective**: Write a conformance suite for the requirements your design relies on and run it against the python-sdk at commit `77001c96f9507be0f4d009b83210a384a5e08bda` in its default and a hardened configuration.
- **Pre-Registered Hypothesis**: Under defaults, the unknown-tool test, the wrong-audience test, and the failing-lookup header test diverge from the specification; the audience test passes with `validate_token_resource=True` (**H**, CLM-071).
- **Independent Variables**: configuration (default, hardened); request class (unknown tool, invalid arguments, duplicate registration, unknown version, wrong-audience token, insufficient scope, raising schema lookup).
- **Dependent Variables**: wire response per test (JSON-RPC error vs `isError` result, status codes, headers); log content for rejected arguments (field names only); coverage $|R_{tested}|/|R_{relied}|$ (**D**, CLM-066).
- **Break & Falsify**: (1) If any of the three predicted divergences behaves as the specification requires under defaults, the static reading missed a path: find it and record it. (2) Register two tools with the same name and assert which one runs (**O**, CLM-043). (3) Make the input-schema lookup raise and assert whether the call is served (**O**, CLM-047).
- **Alignment**: Lesson 38.6; also exercises Lessons 38.2 and 38.5.
- **Hardware**: CPU for the suite; TRACE for the pinned source reading.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total). The source trace itself is counted under `source_trace`.

---

## 07 Break / Incident Scenarios

### Incident 38.1: The Third Server
- **Incident Symptoms** (synthetic telemetry, registry CLM-073):
  On Tuesday a platform team connects a third MCP server (a search and knowledge server) to an internal assistant that already used two servers.
  - Input tokens per model call rise from 36,286 to 57,886 (1.60×).
  - Selection accuracy on the nightly fixed probe set falls from 0.91 to 0.78.
  - Cache-read tokens per call fall from about 34,000 to near zero.
  - Support tickets report that "search" now returns knowledge-base articles when users ask about code.
  - No error rate change; every `tools/list` and `tools/call` returns success.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: (A) the larger tool surface crossed this model's selection knee; (B) a name collision: two tools named `search`, one shadowing the other in the aggregator; (C) the new server returns tools in nondeterministic order, destroying the cached prefix; (D) the new server's descriptions are vague and attract selections; (E) a model or provider change on the same day; (F) probe-set drift.
  2. *Rank Initial Plausibility*: the "search returns articles" tickets point at B; the cache collapse points at C, since adding tools at the end of a deterministic list would leave the old prefix intact; the token growth alone supports A only weakly. Do not assume one cause.
  3. *Identify Missing Evidence*: the merged tool list with server origin per tool; serialized tool blocks from consecutive calls; per-tool selection counts before and after; the aggregator's collision policy; model and provider revision logs; the probe set's hash.
  4. *Design Discriminating Test*: (a) replay the probe set with the third server's tools prefixed by server id: if accuracy recovers to near 0.91 while tokens stay at 57,886, B dominates over A. (b) Compare two consecutive serialized tool blocks: a first difference inside the third server's section with identical content in different order confirms C. (c) Load only the 40 original tools plus 24 neutral filler tools of matched size: a drop there supports A independent of B. (d) Pin model revision and re-run to exclude E.
  5. *Execute Causal Diagnosis*: rank by the results; B and C can both hold, because one breaks accuracy and the other breaks cost. State what each test ruled out.
  6. *Prescribe Immediate Mitigation & Long-Term Intervention*: immediate: prefix tool names by server id; sort the aggregator's output deterministically; manifest-pin the new server. Long term: a tokens-per-tool budget with retrieval past it; a conformance test for order stability; a review gate for any new server's names and descriptions.
  7. *Remeasurement & Post-Mortem Defense*: probe accuracy with paired comparison against the pre-change baseline; cache-read tokens per call; per-tool selection counts for both `search` tools; tokens per call; a falsifier: if accuracy is still low after prefixing and ordering, revisit A and D.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Architecture Transfer Problem: Internal Tool Gateway
You are designing an internal tool gateway that sits between several agent applications and the company's MCP servers. It aggregates servers, controls what each model call sees, handles protocol eras, and terminates authorization.

**Workload Fixture (SYNTHETIC — exercise assumptions, not measurements; registry CLM-072):**

| Input | Fixture value |
|---|---|
| Servers | 9 MCP servers; 3 are legacy (2025-11-25 handshake only); 6 are modern |
| Catalog | 240 tools, mean 948 definition tokens per tool (227,520 tokens) |
| Context | Model window 200,000 tokens; at most 10% (20,000) for tool definitions; tool-use overhead $c_0=286$ |
| Session | 30 model calls per session |
| Names | 4 servers expose a tool named `search` |
| Authorization | 6 servers require OAuth; one publishes 12 scopes in `scopes_supported` |
| Write tool | `create_ticket` has no idempotency key |
| SDK | Servers built on python-sdk `77001c96…` with default `AuthSettings` |
| Unknown by design | Your model's selection knee, retrieval recall on this catalog, change rates per server. Measure them or carry them as symbols. |

You may replace any fixture value with your own measurement; record model, runtime, revision, and method. Answers are graded conditionally on declared inputs.

**Constraints**:
- No model call may carry more than 20,000 tokens of tool definitions.
- No token issued for one server may be accepted by another.
- Legacy servers must keep working during the 12-month deprecation window.
- No attack construction; security work is limited to the controls of Lessons 38.4–38.5 and references to Module 18.

**Required Deliverables**:
1. **Context budget**: $C_{tools}$ for loading everything, its share of the window, and uncached per-session cost; $n_{max}$ at the 10% budget; a retrieval design (retriever, $K$, deferred loading) with the three decomposition factors you will measure.
2. **Selection experiment plan**: the LAB A design adapted to this catalog: $N$, $K$, slices (difficulty, near-duplicate names, irrelevance), paired comparison, and the decision rule for $K$.
3. **Era and lifecycle plan**: how the gateway talks to legacy and modern servers (dual-era or adapters), its era cache, its −32022 handling, and how it re-issues broken calls for `create_ticket` without duplicate tickets.
4. **Catalog policy**: name prefixing scheme, deterministic ordering, manifest pinning per authorization context, TTL vs subscription per server with the staleness model, and schema-change classification rules.
5. **Authorization design**: audience binding (SDK settings or verifier), passthrough refusal for downstream calls, minimal `scopes_supported` for the 12-scope server, step-up flow, and `iss` handling.
6. **Conformance suite**: the requirements you rely on, the tests, expected wire responses, coverage, and the configuration they run against.
7. **Rejection of an inappropriate design**: analyze and reject at least one plausible alternative, for example loading all 240 tools with a larger context model, or relying on the registry version as the pin. Reject it only if your inputs support the rejection.
8. **Uncertainty, falsification, and rollback**: which measurement would overturn your $K$, your TTLs, or your scope design; production metrics and rollback triggers.

---

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Submit a pinned trace of an MCP server implementation. The reference trace uses `modelcontextprotocol/python-sdk` commit `77001c96f9507be0f4d009b83210a384a5e08bda`. It must map:
1. Registration: `MCPServer.add_tool` → `ToolManager.add_tool` → `Tool.from_function`, including what happens on a duplicate name and an empty docstring.
2. Listing: `_handle_list_tools` and the order and pagination behavior.
3. Calling: `_handle_call_tool` → `ToolManager.call_tool`, and how unknown tools, validation failures, `MCPError`, and unexpected exceptions reach the wire.
4. Authorization: how `MCPServer` builds `BearerAuthBackend`, which setting enables `_issued_for_this_resource`, and what `RequireAuthMiddleware` returns.
5. A list of what was not executed or verified: client side, other auth modules, the TypeScript SDK, and any runtime claim without a LAB D test.

Each finding records repository, commit, file, symbol, entry path, static or executed, and whether it generalizes beyond this SDK.

### Reference Checks for Mastery Deliverables 1, 4, 5 (fixture inputs only)
Reviewers use these to check arithmetic, not to grade a design.

- **Load everything**: $C_{tools}=286+240\times948=227{,}806$ tokens; $227{,}806/200{,}000=1.14$, so it does not fit the window at all. Uncached per session: $30\times227{,}806=6{,}834{,}180$ tokens.
- **Budget**: $n_{max}=\lfloor(20{,}000-286)/948\rfloor=\lfloor20.80\rfloor=20$ tools. A shortlist of $K=5$ costs $286+5\times948=5{,}026$ tokens per call, $30\times5{,}026=150{,}780$ per session. Chance rate at $K=5$, $N=240$: $5/240$, BoR maximum $\log_2(240/5)=5.58$ bits.
- **Collisions**: 4 servers expose `search`; an unprefixed first-wins merge leaves 3 unreachable; a server-id prefix leaves 0.
- **Staleness**: at $T=300$ s, $p_{stale}=0.17\%$, $4.05\%$, and $21.3\%$ for one change per day, hour, and 10 minutes.
- **Scopes**: requesting all 12 scopes for 3,600 s is $43{,}200$ scope-seconds; two scopes for 1,800 s then four for 1,800 s is $10{,}800$, a $4\times$ reduction, under the equal-weight assumption.
- **Audience**: under the fixture's default `AuthSettings`, the SDK middleware does not check token resource; the design must set `validate_token_resource=True` with `resource_server_url`, or use a verifier that checks audience, and LAB D must show it.

### Rubric Dimensions
- **Conditional Quantitative Answers** (Deliverables 1, 4, 5):
  - *Insufficient*: Quotes a vendor tool-count threshold as a limit, or uses a mean per-tool cost without noting spread.
  - *Competent*: Declares every input with its source (fixture, measurement, assumption), computes budget, staleness, and exposure with units, and labels vendor guidance as guidance.
  - *Strong*: Also gives sensitivity to the unknowns (knee, recall, change rates) and the measurement that would change each decision.
- **Mechanistic Reasoning**: *Insufficient* describes MCP as "a standard for tools". *Competent* explains the request lifecycle, era rules, list semantics, and auth flow step by step. *Strong* connects ordering to prompt-cache reuse, authorization-dependent lists to cache keys, and stream loss to effect verification.
- **Experiment Design**: *Insufficient* reports one accuracy number. *Competent* separates recall, choice, and argument validity on paired episodes. *Strong* includes irrelevance and near-duplicate slices and a falsification condition for each hypothesis.
- **Failure Diagnosis**: *Insufficient* blames "too many tools". *Competent* forms competing hypotheses for Incident 38.1 with discriminating tests. *Strong* separates causes that act on accuracy from those that act on cost and shows both can hold.
- **Security and Authorization**: *Insufficient* relies on the specification text or the SDK name. *Competent* designs audience binding, no passthrough, minimal scopes, and step-up. *Strong* proves each in the deployed configuration and states residual risk handed to Module 18.
- **Source Reasoning**: *Insufficient* cites docstrings as runtime proof. *Competent* traces the pinned paths and labels them static. *Strong* turns each divergence into an executed conformance test and states what it does not cover.

---

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| **Descriptions as selection interface; cost-adjusted rewrites** | Lesson 38.1 | Lesson 38.1 Guided Practice; LAB A description arm | Mastery Deliverable 2; Incident 38.1 hypothesis D | Paired rewrite table with adoption-rule computation |
| **MCP lifecycle, eras, version errors** | Lesson 38.2 | Lesson 38.2 Guided Practice (era table); LAB B | Mastery Deliverable 3 | Era table and −32022 test log |
| **List semantics, ordering, cache hints, subscriptions** | Lessons 38.2, 38.4 | LAB B ordering, TTL, and subscription runs | Incident 38.1 steps 2–4; Mastery Deliverable 4 | Prefix-length measurements; stale-read fractions vs model |
| **Context budget and shortlist depth** | Lesson 38.3 | Lesson 38.3 Guided Practice; LAB A | Mastery Deliverables 1–2; Section 09 reference checks | Budget calculation; recall@K / choice / end-to-end curves with BoR |
| **Registry, pinning, collisions, schema evolution** | Lesson 38.4 | Lesson 38.4 Guided Practice (pinning record); LAB B break tests 2–3 | Incident 38.1 steps 4–6; Mastery Deliverable 4 | Pinning record; collision test; compatibility classifications |
| **Audience binding, passthrough refusal, scopes, step-up** | Lesson 38.5 | Lesson 38.5 Guided Practice; LAB C | Mastery Deliverable 5; Section 09 reference checks | Accept/reject matrix; 401/403 captures; scope-seconds |
| **Conformance and pinned SDK trace** | Lesson 38.6; Section 05 trace | Lesson 38.6 Guided Practice (trace); LAB D | Mastery Deliverable 6; Section 09 Required Artifact items 1–5 | Conformance results under default and hardened configs; trace at `77001c96…` |

---

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 38 must be able to:
1. Write a tool's selection interface separately from its execution contract and decide on description changes with a cost-adjusted, paired measurement.
2. Drive an MCP 2026-07-28 client through discovery, version errors, era handling, caching, and subscriptions, and contrast it with provider function calling, OpenAPI, and A2A.
3. Derive a tokens-per-tool budget and shortlist depth, and replace vendor thresholds with a measured selection knee.
4. Pin the model-visible tool surface, prevent name collisions, and classify schema changes by compatibility direction.
5. Configure and prove audience-bound, minimally scoped MCP authorization without token passthrough.
6. Turn specification requirements into wire-level conformance tests against a deployed configuration and trace a pinned SDK.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: A tool ecosystem is a protocol-mediated interface whose text is paid for on every model call and whose guarantees exist only where a deployed component enforces them.
- **The Path of One Tool Call**:
  $$\text{Registry pin} \to \text{discover/era} \to \text{tools/list (per authorization, ordered, cached)} \to \text{shortlist } K \to \text{model choice} \to \text{tools/call with audience-bound token} \to \text{result, isError, or protocol error}.$$
- When a tool-using system degrades, check the five objects separately: protocol revision, registry entry, live list, model-visible surface, and token. Accuracy and cost can fail for different reasons at the same time.
- Next: Module 18 for adversarial descriptions and gateway policy; Module 22 for pricing the tool surface; Module 23 for its telemetry.

### Cost & Telemetry Hook

- [Lesson 22.3](../22-ai-economics/README.md): record tool-definition tokens as a separate line in per-call usage, so the cost of the tool surface is visible apart from conversation tokens.
- [Lesson 22.4](../22-ai-economics/README.md): price prefix-cache reuse of the tool block; deterministic ordering and manifest pinning are cache-economics controls.
- [Lesson 22.1](../22-ai-economics/README.md): cost per verified task success must include sessions that failed because of selection errors or authorization denials.
- [Lesson 23.1](../23-observability-reliability/README.md): emit per call the tool-list hash, $K$, tools shown, tool chosen, server origin, protocol revision, and authorization decision (401/403/accepted) as span attributes.
- [Lesson 23.3](../23-observability-reliability/README.md): define a selection SLI over offered requests (correct tool and valid arguments), not over completed calls.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
