# Module 37 — Multi-Agent Systems

## 00 Why This Module Exists
A multi-agent system splits one task across several model-driven controllers that exchange messages, hand over control, or critique each other. The split can buy parallel breadth, a fresh context per subtask, and role specialization. It also costs tokens, latency, and information: every boundary between agents is a channel that can drop evidence, duplicate context, or carry an error forward.

Published evidence points both ways. One vendor reports a large gain from an orchestrator–worker research system, but against a baseline that spent fewer tokens (**O**, CLM-001). When studies hold the token budget fixed, multi-agent designs gain on decomposable tasks and lose on sequential ones (**O**, CLM-006, CLM-012). Whether to build one is therefore an experimental question about a specific workload, not an architectural preference.

```text
task + budget
     |
decomposable? ---- no ----> single agent (Module 12) + more budget
     |
    yes
     |
topology: orchestrator-worker | handoff | debate | agent-as-tool
     |
context regime at each boundary: forward-all | shared state | isolated return
     |
errors propagate across boundaries ---> verifiers (correlated?)
     |
tokens x calls x latency ---> cost per verified success
     |
trajectory evaluation: which agent, which step?
```

This module owns the decision to use more than one agent, topology selection, inter-agent context passing, cross-agent error propagation, multi-agent cost multipliers, and multi-agent trajectory attribution. It does not re-teach the single-agent controller (Module 12), durable execution (Module 14), evaluation statistics (Module 15), or security (Module 18).

**Research cutoff:** 2026-11-10. All sources were opened on 2026-11-09/10. Sources marked *abstract only* supply no figures beyond their abstracts.

**Module Orientation**
- **If You Only Have One Week**: Lessons 37.1, 37.3, 37.5 — they cover the adopt-or-not decision under an equal budget, what each boundary loses or duplicates, and what the design costs per verified success; topologies, error propagation, and attribution build on those three.
- **Engineering Problem**: Decide whether, and in which topology, to split a task across agents so that verified task success per unit of budget improves over the best single agent at the same spend, without hiding context loss, cascading errors, or unattributable failures.
- **What You Will Do**: Run budget-matched single- versus multi-agent comparisons on synthetic tasks with controlled decomposability, implement and measure four context-passing regimes, seed errors into a multi-stage pipeline and test verifiers, derive token/call/latency/cost multipliers, trace handoff and supervisor code at pinned commits, attribute failures in seeded trajectories, and defend a design for a mixed workload.
- **Environment**: Python 3.10+ on CPU. Labs use a scripted mock LLM with seeded, configurable error rates, or optionally a small local model; no hosted API key or local GPU is required. Hardware tiers are `CPU`, `COLAB` (one free-tier Colab GPU, optional), and `TRACE` (static source reading).
- **Evidence Rule**: **O** is a source observation, **D** a derivation under stated assumptions, **H** an engineering hypothesis that needs measurement. `CLM-xxx` identifiers point to `research-registry/37-multi-agent-systems.yaml`. Every worked-example input in this module is **synthetic** and labeled so.

## 01 Baseline Assumptions
- **Bounded single-agent controller** (Module 12, Lesson 12.1): controller-owned state, proposals versus validated actions, explicit terminal reasons. Each agent in this module is such a controller.
- **Budgets and reservation** (Module 12, Lesson 12.3): the budget vector, pre-dispatch reservation $C_r+R_r\le B_r$, root deadlines, and nested-retry multiplication $k_c(1+k_s)$. Multi-agent budgets are root budgets shared by all agents.
- **Single-agent trajectory evaluation** (Module 12, Lesson 12.6): offered-episode accounting, critical-path latency, and all-attempt cost.
- **Context accounting and lossy compression** (Module 11, Lessons 11.1 and 11.5): token accounting of a serialized request, and summaries as derived, lossy views.
- **Durable fan-out and effect identity** (Module 14, Lessons 14.2 and 14.4): receiver idempotency, persisted child membership, join predicates, and $\max_i T_i$ join latency.
- **Evaluation contracts and statistics** (Module 15, Lessons 15.1, 15.3, 15.4): experimental units, paired comparisons, graders as instruments, and intervals. This module uses an unpaired normal approximation in Lesson 37.1 only as a screening check and defers the release-grade analysis to Lesson 15.4.
- **Denominator-safe cost** (Module 22, Lesson 22.1): cost per successful offered task, including failed attempts.

**Boundary.** Module 12 owns the single-agent controller; here each agent is assumed to be one. Module 14 owns durable orchestration. A workflow engine's "orchestrator" that persists and replays activity steps is not multi-agent coordination, even when its activities call models. Module 15 owns evaluation statistics, and Module 18 owns security, including injection carried between agents.

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
  security: NOT_APPLICABLE
  economics: SELECTIVE
  architecture_tradeoff: REQUIRED
  research_connection: SELECTIVE

estimated_effort:
  instruction: 4h        # lesson instruction: 40+45+40+40+35+40 min = 240 min
  guided_practice: 3h    # lesson practice: 30+30+30+35+25+30 min = 180 min
  labs: 12h              # LAB A 3h + LAB B 3h + LAB C 3h + LAB D 3h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 37.1 0.5h
  source_trace: 2h       # Lesson 37.2 guided trace and the Section 09 Production Source Trace artifact, counted once
  total: 24h
```
Each category is counted once. The source trace is not also counted as Lesson 37.2 practice or as LAB B time, and lab analysis is not counted again as guided practice.

The learner must be able to: run a budget-matched baseline before adopting agents; state what a topology assumes about decomposability and control; quantify what each context boundary duplicates or loses; model and measure error propagation with an explicit metric; derive token, call, latency, and cost-per-success multipliers; trace a pinned handoff and supervisor implementation; and attribute multi-agent failures with a validated method.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map
Task and Budget $\to$ Decomposability and Single-Agent Ceiling $\to$ Budget-Matched Baseline $\to$ Topology (who owns control) $\to$ Context Regime at Each Boundary (what each agent sees) $\to$ Error Propagation and Verification $\to$ Token, Call, Latency, and Cost Multipliers $\to$ Trajectory Evaluation and Attribution $\to$ Adopt, Reshape, or Revert.

Keep five questions separate:
1. **Benefit:** does the split beat a single agent given the same measured budget?
2. **Control:** which agent owns the next decision, and can several act at once?
3. **Visibility:** what does each agent see, and what was deleted to get it there?
4. **Propagation:** when one agent is wrong, which check stops the error?
5. **Attribution:** when the run fails, which agent and step caused it, and how sure are we?

## 04 Lessons

### Lesson 37.1 — When Multiple Agents Beat One

**Engineering Question:**
Under what task and budget conditions does splitting work across agents improve verified success over the best single agent at the same spend?

**Concepts & Definitions:**
- **Multi-agent system (MAS)**: two or more model-driven controllers, each with its own context, that exchange messages or transfer control to complete one task. Several tool calls from one controller are not a MAS.
- **Single-agent system (SAS) baseline**: one controller given the same tools. A *native* baseline uses its default budget. A *budget-matched* baseline gets the same measured tokens, calls, and tool access as the MAS, spent as longer thinking, more samples, or more tool calls.
- **Equal-budget comparison**: success compared at matched measured spend. Requested budgets are not enough: one study found an API model's visible thinking plateaued near 1,462 tokens at a 10k request while a multi-call MAS spent about 5,785 (**O**, CLM-013).
- **Decomposability**: the degree to which subtasks can be solved without each other's intermediate results. Breadth search over independent sources is decomposable; sequential planning, where each step depends on the last, is not.
- **Capability ceiling**: the single-agent success level above which added agents tend to hurt. Kim et al. report negative returns once single-agent accuracy exceeds about 45% (β = −0.236, p = 0.004) on their six benchmarks (**O**, CLM-007).
- **Information loss at a boundary**: if agent B sees only messages $M=g(C)$ computed from agent A's context $C$, B cannot hold more information about the answer than $C$ did (**O**, CLM-012).

**Mechanism Explanation:**
A MAS can help through three distinct mechanisms, and each predicts a different experiment:
1. **More spend.** Extra agents spend more tokens. Anthropic reports that token usage alone explained 80% of BrowseComp performance variance and that MAS "work mainly because they help spend enough tokens" (**O**, CLM-002). Its 90.2% gain on an internal eval came against a single agent that spent less (**O**, CLM-001). A budget-matched SAS tests this mechanism.
2. **Fresh context per subtask.** Each worker starts with a short, focused context. This helps when one long context is degraded. Tran and Kiela found a Sequential MAS beat the SAS only under heavy masking or substitution noise (degradation 0.7), not with clean context (**O**, CLM-013).
3. **Parallel breadth.** Independent subtasks run at once and shorten wall time (Lesson 37.2).

Against those, every boundary loses information and adds coordination turns. With budgets matched, Kim et al. report MAS effects from +80.8% on decomposable financial reasoning to −70.0% on sequential planning, and every MAS variant lost on PlanCraft (**O**, CLM-006). Tran and Kiela found the SAS best or statistically tied at every budget above 100 thinking tokens on multi-hop QA (**O**, CLM-012). Abstract-level results agree in direction: a single agent matched homogeneous workflows and gained from KV-cache reuse (**O**, CLM-014), and a Planner–Executor–Critic team with a frozen 7B model did not beat a single agent at fixed calls (**O**, CLM-015). One abstract reports small equal-compute gains for debate and mixture-of-agents over self-consistency (**O**, CLM-016).

**Quantitative Model / Derivation:**
The information-loss argument (**O**, CLM-012) is a data-processing inequality. Let $Y$ be the correct answer, $C$ the full context, and $M=g(C)$ the message another agent receives. Because $M$ is computed from $C$ alone, $Y\to C\to M$ is a Markov chain, so
$$I(Y;C)\ \ge\ I(Y;M),\qquad P_e(C)\ \le\ P_e(M)$$
*Where*: $I(\cdot;\cdot)$ is mutual information; $P_e(X)$ is the minimum achievable error probability when predicting $Y$ from $X$.

*Step-by-step*:
1. $g$ cannot add information about $Y$ that $C$ lacks; it can only keep or discard it.
2. So the best predictor from $M$ cannot beat the best predictor from $C$.
3. The inequality assumes *perfect use* of $C$. A real model may use a long $C$ poorly. That is the only way a split can win at equal budget: when the receiver uses a short $M$ better than one agent uses the full $C$, or when the split buys more compute.

A stylised model states the same trade-off as a crossover: decomposition wins once the attention cost avoided by resetting context exceeds the handoff cost (abstract only; **O**, CLM-017).

*Decision rule (D, from the mechanisms above):* try a MAS only if (a) the task decomposes, (b) a budget-matched SAS has been measured, and (c) the native SAS is below the ceiling region. Kim et al.'s cross-validated $R^2$ is 0.373, so treat 45% as a screening heuristic, not a law (**O**, CLM-007).

**Worked Example:**
*Inputs (synthetic, CLM-041).* 200 independent tasks per arm. Native SAS: success 0.44 at 10,000 tokens per task. MAS: 0.55 at 37,500. Budget-matched SAS: 0.52 at 37,500.

*Step 1: the headline gain.* MAS − native SAS $=0.55-0.44=0.11$. With independent arms, $SE=\sqrt{0.55\cdot0.45/200+0.44\cdot0.56/200}=0.0497$, so a 95% normal interval is $0.11\pm1.96\times0.0497=[0.013,\,0.207]$.

*Step 2: the budget-matched gain.* MAS − matched SAS $=0.03$, $SE=0.0499$, interval $[-0.068,\,0.128]$. It includes zero.

*Step 3: efficiency.* Success per 1K tokens: native SAS $0.44/10=0.044$; MAS $0.55/37.5=0.0147$; matched SAS $0.52/37.5=0.0139$.

*Interpretation and limits.* About 0.08 of the 0.11 gain is explained by spend in this fixture. The remaining 0.03 is not distinguishable from zero at $n=200$. The native SAS (0.44) is just under the 45% screening threshold, so the heuristic does not rule a MAS out here. The interval is a screening check; a release decision needs the paired design of Lesson 15.4.

**Knowledge Check:**
1. A vendor reports a 90% gain for a MAS over a single agent. Which two facts do you need before believing the architecture caused it?
2. Why does the data-processing inequality not prove that a MAS can never win?
3. A task's native SAS succeeds 70% of the time. What does the capability-ceiling result predict, and how confident should you be?

**Guided Practice:**
For a workload of your choice, list three tasks. For each, state whether subtasks need each other's intermediate results, estimate native SAS success, and write the budget-matched SAS arm you would run (what the extra tokens buy: longer thinking, $k$ samples with a vote, or more tool calls). Predict the sign of MAS − matched SAS before running anything.

**Feedback Contract:**
- *Expected Evidence*: (1) the baseline's token spend and whether it was matched; the task mix (Anthropic's was breadth-first research) (**O**, CLM-001, CLM-003). (2) It assumes perfect context use; degraded or overloaded context or extra compute can let a split win. (3) Negative returns are predicted, but the regression explains little out-of-sample variance; run the matched baseline. Guided Practice: each task has a coupling statement, a matched arm, and a predicted sign written before measurement.
- *Common Failure*: Comparing a MAS against the native SAS and reporting the gap as an architecture effect; matching requested rather than measured budgets.
- *Diagnostic Hint*: If you gave the single agent the same tokens, what would it do with them?
- *Concept to Revisit*: Equal-Budget Comparison; Information Loss at a Boundary.

**Learning Outcome:**
Decide whether a task justifies multiple agents by measuring a budget-matched single-agent baseline and stating which of spend, fresh context, or parallel breadth explains any gain (**H**, CLM-036).

*(Effort: 40m instruction, 30m practice)*

---

### Lesson 37.2 — Topologies and What Each Assumes

**Engineering Question:**
Which topology fits a task's coupling and control needs, and how does a pinned implementation actually transfer control and context?

**Concepts & Definitions:**
- **Orchestrator–worker**: a lead agent decomposes the task, dispatches workers, and synthesizes their results. Anthropic's research system uses this with parallel subagents (**O**, CLM-001). Assumes decomposable subtasks and a lead that can judge results.
- **Handoff**: control moves to another agent, which becomes the single owner of the conversation. Assumes the receiver needs the conversation so far and that only one agent should act at a time.
- **Debate / critique**: peers answer, read each other's answers, and revise over rounds (**O**, CLM-020). Assumes errors are partly independent across agents so that disagreement carries information.
- **Agent-as-tool**: a worker runs in its own context, called like a tool, and returns a string; the caller keeps control. Assumes the subtask can be specified in one argument and summarized in one result.
- **Conversational and SOP pipelines** (historical): agents that converse freely (AutoGen) or follow encoded procedures in an assembly line (MetaGPT), the latter motivated by cascading hallucinations from naive chaining (abstract only; **O**, CLM-040).
- **Synchronous barrier**: the orchestrator waits for all dispatched workers before continuing. Anthropic reports its subagents run synchronously, so the lead cannot steer them mid-run and they cannot coordinate (**O**, CLM-004).

**Mechanism Explanation:**
The topologies differ in two state variables: **who owns control** and **what each agent sees**.

| Topology | Control owner | Receiver sees | Fan-out | Assumes |
|---|---|---|---|---|
| Orchestrator–worker | lead, always | task slice from lead | yes | decomposable, judgeable results |
| Handoff | moves to receiver | full transcript (by default in the SDK traced below) | no | one actor at a time |
| Debate | shared, by round | peers' answers | yes, per round | partly independent errors |
| Agent-as-tool | caller | the tool argument only | optional | specifiable subtask |

*Pinned trace 1 — handoff (openai-agents-python, commit `d68961657843c7d81dfc25834484f8aac2957eb3`, static read)* (**O**, CLM-030):
1. A `Handoff` is exposed to the model as a tool named `transfer_to_<agent name>` (`Handoff.default_tool_name`).
2. When the model calls it, `execute_handoffs` in `src/agents/run_internal/turn_resolution.py` runs. If the model emitted more than one handoff in the same turn, `multiple_handoffs` is true; the extra calls get the tool output "Multiple handoffs detected, ignoring this one." and only `run_handoffs[0]` executes. Handoff is strictly single-owner here.
3. Context: the `input_filter` docstring says the new agent sees the entire conversation history by default. A per-handoff `input_filter` overrides `run_config.handoff_input_filter`.
4. The step returns `NextStepHandoff(new_agent)`.

*Pinned trace 2 — agent-as-tool (same commit)* (**O**, CLM-031): `Agent.as_tool` runs a nested `Runner.run` and returns `run_result.final_output` (or a `custom_output_extractor` result, or a fallback to the last non-empty message or tool output). The worker sees only the generated tool input; the parent sees only the returned string. Its default `failure_error_function` returns error text to the parent instead of raising.

*Pinned trace 3 — supervisor (langgraph-supervisor-py, commit `88859b34017ac3569bbd4a3092c7e77593a0a960`, static read; repository archived; LEGACY)* (**O**, CLM-032): `create_supervisor` builds a star: `START → supervisor`, and every worker node has an edge back to the supervisor. Defaults are `parallel_tool_calls=False`, `output_mode="last_message"`, `add_handoff_messages=True`. Workers are invoked with the full shared state. `create_handoff_tool` returns `Command(goto=agent_name, graph=Command.PARENT)`, and fans out with one `Send` per worker only when the model emits several handoff calls. The README at this commit recommends the supervisor pattern "directly via tools" instead of this library for most uses.

A different alternative removes agents altogether: one abstract reports compiling a MAS into a single agent that selects skills, competitive at lower cost until the skill library grows large enough that selection accuracy drops sharply (abstract only; **O**, CLM-018).

**Quantitative Model / Trade-off Comparison:**
For an orchestrator that plans for $t_p$, dispatches $w$ workers with durations $t_1,\dots,t_w$, and synthesizes for $t_s$:
$$T_{parallel}=t_p+\max_i t_i+t_s,\qquad T_{sequential}=t_p+\sum_i t_i+t_s$$
Total tokens and work are the same in both; only wall time changes (**D**, CLM-034). This is Lesson 14.4's all-of join applied to agents. Anthropic reports parallel subagents and parallel tool calls cut research time by up to 90% on its workload (**O**, CLM-004).

**Worked Example:**
*Inputs (synthetic, CLM-034).* Plan 20 s; three workers of 40, 55, and 30 s; synthesis 15 s.
- Parallel: $20+\max(40,55,30)+15=90$ s.
- Sequential: $20+40+55+30+15=160$ s.
- Wall-time saving: $1-90/160=43.75\%$. Token spend is unchanged.

*Interpretation and limits.* The 55 s worker sets the critical path; speeding up the 30 s worker changes nothing. Provider rate limits or a shared serving queue can serialize "parallel" calls (Module 04), so measure start and end times. Under a synchronous barrier, a worker that hangs holds the whole run; the barrier needs a deadline from the root budget (Lesson 12.3).

**Knowledge Check:**
1. In the pinned OpenAI SDK, what happens if the model asks to hand off to two agents in one turn? What does that tell you about using handoffs for fan-out?
2. A code-review task needs one agent to edit and another to check style. Which topology keeps a single writer, and why?
3. Why is the archived supervisor library classified LEGACY even though its code still runs?

**Guided Practice:**
At the two pinned commits, locate `execute_handoffs`, `Agent.as_tool`, `create_supervisor`, and `create_handoff_tool`. For each, write: entry point, who owns control after the call, what the receiver sees, and whether fan-out is possible. Mark each finding as static read and as generalizable YES, NO, or PARTIAL.

**Feedback Contract:**
- *Expected Evidence*: (1) Only the first executes; extras are answered "ignored". Handoffs are not a fan-out mechanism here; use agent-as-tool or an orchestrator. (2) Agent-as-tool or orchestrator–worker with the editor as owner and the checker returning findings; handoff would also work if strictly sequential. (3) The repository is archived and its README recommends another approach; the code is a reference implementation, not a current default. Guided Practice: table rows match CLM-030 to CLM-032 and mark field names as SDK-specific (PARTIAL).
- *Common Failure*: Treating "supervisor", "orchestrator", and "handoff" as synonyms; describing default behavior from documentation prose instead of the pinned code.
- *Diagnostic Hint*: After the call returns, whose loop runs next?
- *Concept to Revisit*: Control Owner versus Visibility.

**Learning Outcome:**
Select a topology from task coupling and control needs, and trace how a pinned implementation transfers control and context.

*(Effort: 45m instruction, 30m practice on the worked example and knowledge checks. The guided source trace is counted once, under `source_trace`.)*

---

### Lesson 37.3 — Shared State and Context Passing

**Engineering Question:**
At each boundary between agents, what context is duplicated, what is lost, and how can either be measured?

**Concepts & Definitions:**
- **Forward-all**: the receiver gets the whole transcript. Nothing is lost; everything is re-read. The pinned OpenAI handoff does this by default (**O**, CLM-030).
- **Shared message state**: all agents read and append to one log. Under the pinned supervisor's `full_history`, a worker's whole message list is appended; under `last_message` (the default), only its final message (or last two if the last is a tool message) plus synthetic `transfer_back` messages (**O**, CLM-032).
- **Isolated return**: the worker sees only its argument; the caller sees only the returned string (**O**, CLM-031).
- **Summarized or filtered forwarding**: the transcript is compressed or stripped before handoff. In the pinned SDK, `nest_handoff_history` is an opt-in beta that formats history into one summary message by string formatting, not by an LLM, and "does not redact sensitive data"; `remove_all_tools` drops tool calls and tool outputs from the forwarded history (**O**, CLM-030). Both are lossy views (Lesson 11.5).
- **Implicit decision**: a choice an agent makes while acting (a naming convention, a data format) that other agents must follow but were never told. Cognition argues "actions carry implicit decisions, and conflicting decisions carry bad results" and recommends sharing full traces (**O**, CLM-019).

**Mechanism Explanation:**
Each regime sets a different trade-off:

| Regime | What the receiver loses | What is duplicated | Typical failure |
|---|---|---|---|
| Forward-all | nothing | whole transcript, re-read per call | token and latency growth |
| Shared state, `full_history` | nothing | every worker trace, re-read by all | quadratic re-read cost |
| Shared state, `last_message` | worker's tool evidence and reasoning | final messages only | supervisor cannot check worker claims |
| Isolated return | sibling work; caller's context | almost nothing | conflicting implicit decisions |
| Summarized / filtered | whatever the summary dropped | summary | missing evidence downstream |

MAST's taxonomy includes failure modes for these boundaries: loss of conversation history (2.8% of annotated failures), conversation reset (2.2%), information withholding (0.85%), and ignored other agent's input (1.9%) in its 1,642-trace dataset (**O**, CLM-026). Anthropic's system routes subagent output through a filesystem instead of the lead's context to avoid a "game of telephone" (**O**, CLM-004). That is isolated return plus a shared artifact store.

When parallel workers write parts of one artifact without seeing each other, they can make incompatible implicit decisions (**H**, CLM-037). Cognition's example is qualitative (**O**, CLM-019). The effect size has to be measured on your task.

**Quantitative Model / Derivation:**
Consider a supervisor whose shared state starts at $S_0$ tokens. Each round, one worker reads the state, works, and appends $d$ tokens: its whole trace under `full_history`, or its last message plus handoff messages under `last_message`. Over $m$ rounds:
$$\text{Supervisor input}=\sum_{j=0}^{m}(S_0+j\,d),\qquad \text{Worker input}=\sum_{j=1}^{m}\big(S_0+(j-1)\,d\big)$$
*Where*: $j$ indexes supervisor calls (call $j$ sees $j$ appended rounds); worker $j$ reads the state before its own append.

*Step-by-step*: each call re-reads everything appended so far, so the $j$-th call pays $j\,d$ extra. Summing gives $(m+1)S_0+d\,m(m+1)/2$ for the supervisor: linear in $S_0$, quadratic in rounds through $d$. Changing `full_history` to `last_message` shrinks $d$, not the shape (**D**, CLM-033).

**Worked Example:**
*Inputs (synthetic, CLM-033).* $S_0=2{,}000$ tokens; $m=3$ rounds, one worker each. `full_history` appends $d=6{,}000$; `last_message` appends $d=450$. Agent-as-tool alternative: each worker reads a 300-token argument and returns 400 tokens.

| Regime | Supervisor input (4 calls) | Worker input (3 calls) | Total |
|---|---:|---:|---:|
| `full_history` | $2{,}000+8{,}000+14{,}000+20{,}000=44{,}000$ | $2{,}000+8{,}000+14{,}000=24{,}000$ | 68,000 |
| `last_message` | $2{,}000+2{,}450+2{,}900+3{,}350=10{,}700$ | $2{,}000+2{,}450+2{,}900=7{,}350$ | 18,050 |
| agent-as-tool | $2{,}000+2{,}400+2{,}800+3{,}200=10{,}400$ | $3\times300=900$ | 11,300 |

*Interpretation and limits.* `full_history` costs $68{,}000/18{,}050\approx3.8\times$ the input tokens of `last_message` here. The cheaper regimes are cheaper because they delete worker evidence: under `last_message` the supervisor cannot inspect a worker's tool outputs, and under agent-as-tool workers cannot see each other. Token counts say nothing about which deletion hurts. Measure that with seeded facts (Guided Practice).

**Knowledge Check:**
1. A writer agent receives a handoff filtered with `remove_all_tools` and then cites a source. What can it not check, and why?
2. Why does `last_message` reduce the growth rate but not change the quadratic shape?
3. Four isolated workers each write one section of an API spec. Which decision should be fixed before dispatch, and where should it live?

**Guided Practice:**
Seed 20 facts into worker tool outputs, each needed by the final answer. Run the four regimes (forward-all, `full_history`, `last_message`, isolated return) plus a summarized handoff. For each, measure input tokens and the fraction of seeded facts the final agent can still reproduce verbatim. Predict the ranking on both axes before running.

**Feedback Contract:**
- *Expected Evidence*: (1) The tool calls and outputs it would cite were removed, so it cannot verify the citation against the source text; it can only repeat claims made in assistant messages. (2) Each call still re-reads everything appended before it; `last_message` only makes each append smaller. (3) Shared conventions (naming, error format, versioning) in a decision record that every worker receives; or one writer. Guided Practice: a two-axis table (tokens, seeded-fact recall) per regime with the prediction recorded first; no regime should be called "best" without a task-value weighting.
- *Common Failure*: Treating lower token counts as free; assuming a summary preserved evidence because the final answer still sounds confident.
- *Diagnostic Hint*: For each seeded fact, which boundary was the last one it crossed intact?
- *Concept to Revisit*: Context Regimes; Lossy Views (Lesson 11.5).

**Learning Outcome:**
Quantify what each context-passing regime duplicates and loses, and choose one from measured token cost and evidence recall.

*(Effort: 40m instruction, 30m practice)*

---

### Lesson 37.4 — Error Propagation and Cascading Failure

**Engineering Question:**
How does one agent's error reach the final answer, how should amplification be measured, and when does adding a verifier help?

**Concepts & Definitions:**
- **Error propagation**: a wrong output from one agent becomes another agent's input and is treated as fact.
- **MAST taxonomy**: 14 failure modes in three categories derived from multi-agent traces: system design (41.8% of annotated failures; step repetition 15.7%, disobey task specification 11.8%, unaware of termination 12.4%), inter-agent misalignment (36.9%; reasoning–action mismatch 13.2%, task derailment 7.4%, fail to ask clarification 6.8%), and task verification (21.3%; incorrect verification 9.1%, no or incomplete verification 8.2%, premature termination 6.2%). Human inter-annotator κ was 0.88 (**O**, CLM-026). Prevalences describe seven 2023–24 frameworks, not your system.
- **Error amplification**: how much more often errors occur or survive in a MAS than in a SAS. The definition decides the number. Kim et al.'s trace-level $A_e$ counts errors in traces (**O**, CLM-008); a task-level ratio compares final failure rates (**D**, CLM-011).
- **Verifier**: an agent or check that accepts or rejects another agent's output. MAST reports that existing verifiers in the studied systems performed only superficial checks such as compilation or leftover TODOs (**O**, CLM-027).
- **Correlated verifier**: a verifier that shares the producer's model, prompt, or context and so tends to miss the same errors.

**Mechanism Explanation:**
An error crosses a boundary when the receiver has no independent evidence to reject it. That happens when the receiver lacks the source (context loss, Lesson 37.3), when a verifier checks form rather than content, or when the verifier shares the producer's blind spot.

The interventions MAST tested were modest. ChatDev on ProgramDev-v0 went from 25.0 to 34.4 with prompt changes and 40.6 with a new topology; AG2 on GSM-Plus with GPT-4 went from 84.75 to 89.75 (prompt) and 85.50 (topology). The authors say these "do not constitute substantial improvements" (**O**, CLM-027). Across seven open-source MAS they report failure rates from 41% to 86.7%.

**Quantitative Model / Derivation:**
*Compounding.* For $k$ sequential stages that each succeed independently with probability $p$:
$$P_{end}=p^{k}$$
*Verifier.* If a verifier catches a fraction $v$ of stage errors and a retry then succeeds:
$$p_{eff}=1-(1-p)(1-v)$$
*Correlation.* If the verifier misses errors correlated with the producer's so that its effective catch rate is $v(1-\rho)$:
$$p_{eff,\rho}=1-(1-p)\big(1-v(1-\rho)\big)$$
*Where*: $k$ = number of stages; $p$ = per-stage success; $v\in[0,1]$ = catch rate on an independent error; $\rho\in[0,1]$ = a synthetic correlation parameter (0 independent, 1 the verifier never catches what the producer gets wrong).

*Step-by-step*: (1) independence turns $k$ success events into a product; (2) a stage now fails only if the error occurs ($1-p$) and is missed ($1-v$); (3) correlation shrinks $v$ (**D**, CLM-028).

*Which amplification?* From Kim et al.'s Table 5, task-level $A_{task}=(1-S_{MAS})/(1-S_{SAS})$ with $S_{SAS}=0.466$ gives Independent $0.630/0.534=1.18$, Decentralized $0.98$, Centralized $1.01$, Hybrid $1.03$. The paper's trace-level $A_e$ for the same architectures is 17.2, 7.8, 4.4, and 5.1 (**D**, CLM-011). Many more errors appear in MAS traces, but final failure rates are nearly the same for three of four architectures, so most of those errors are absorbed or irrelevant. The paper also reports that after controlling for other metrics the error-amplification main effect on success is not significant (β = 0.014, p = 0.658) (**O**, CLM-008). Report which definition you use.

**Worked Example:**
*Inputs (synthetic, CLM-028).* $p=0.95$ per stage.
- $k=10$: $0.95^{10}=0.599$. $k=20$: $0.95^{20}=0.358$.
- Add an independent verifier with $v=0.5$: $p_{eff}=1-0.05\times0.5=0.975$; $k=20$: $0.975^{20}=0.603$.
- Same verifier, correlated with $\rho=0.6$: effective catch $0.5\times0.4=0.2$; $p_{eff}=1-0.05\times0.8=0.96$; $k=20$: $0.96^{20}=0.442$.

*Interpretation and limits.* The independent verifier lifts 20-stage success from 0.358 to 0.603; the correlated one only to 0.442. The correlation, not the verifier's existence, decides the value. Real errors are correlated and some are fixed downstream, so these numbers are intuition, not forecasts. Measure catch rate and correlation directly (**H**, CLM-038).

**Knowledge Check:**
1. A team reports "error amplification 17×" for their MAS. What question do you ask before comparing it with your system?
2. Why can a verifier using the same model and prompt as the producer add cost without adding much reliability?
3. MAST's largest single mode is step repetition (15.7%). Which Module 12 mechanism addresses it, and why is it still a multi-agent problem?

**Guided Practice:**
Build a 6-stage pipeline with a scripted mock LLM whose per-stage error rate you set. Seed errors and add (a) no verifier, (b) a verifier using the producer's prompt and seed, (c) a deterministic checker with access to ground truth for one field. Predict end-to-end success with the formulas, then measure catch rate, producer–verifier error correlation, and end-to-end success for each.

**Feedback Contract:**
- *Expected Evidence*: (1) Trace-level or task-level, which baseline, and on which tasks; the same paper's two metrics differ by an order of magnitude. (2) Its misses correlate with the producer's errors, so its effective catch rate falls; it still costs a full call. (3) Progress and stall detection (Lesson 12.4); across agents, repetition can arise when one agent re-requests work another already did, which a single-agent detector does not see. Guided Practice: predicted versus measured success per arm, with the correlation estimate explaining any gap.
- *Common Failure*: Adding an LLM critic and counting its existence as verification; quoting amplification without a definition.
- *Diagnostic Hint*: On the errors the producer makes, how often does the verifier also say "looks fine"?
- *Concept to Revisit*: Correlated Verification; Metric Definition.

**Learning Outcome:**
Model error compounding and verifier effect, measure amplification with a declared metric, and code failures with a published taxonomy.

*(Effort: 40m instruction, 35m practice)*

---

### Lesson 37.5 — Token and Cost Multiplication

**Engineering Question:**
How many more tokens, calls, and dollars per verified success does a multi-agent design cost, and which term dominates?

**Concepts & Definitions:**
- **Token multiplier**: tokens per task of a design divided by those of a baseline. Anthropic reports agents use about 4× the tokens of chat and multi-agent systems about 15× (**O**, CLM-003). If both multiples share a chat baseline and workload, a MAS uses about $15/4=3.75\times$ a single agent's tokens; Anthropic does not state this ratio (**D**, CLM-005).
- **Coordination overhead**: tokens and turns spent on planning, messaging, and synthesis instead of task work. Kim et al. report overhead of 58% (Independent) to 515% (Hybrid) and turns rising from 7.2 (SAS) to 44.3 (Hybrid) (**O**, CLM-008).
- **Cost per verified success**: total spend over all offered tasks divided by verified successes (Lesson 22.1). It falls when success rises faster than spend.
- **Self-consistency (sampling and voting)**: $n$ independent samples with a majority vote, no communication. Token use grows in proportion to $n$; gains are larger for harder tasks and weaker models (**O**, CLM-023). It is the equal-compute baseline for any debate claim.

**Mechanism Explanation:**
A MAS multiplies three things: calls (one per agent per round), input tokens (each call re-reads its context), and turns (coordination). Debate is the clearest case. Du et al. ran 3 agents for two rounds and compared against a single agent and a majority-of-3, both of which use fewer calls (**O**, CLM-020). Later work found debate did not reliably beat self-consistency or ensembling at GPT-3.5 and was more hyperparameter-sensitive (**O**, CLM-022); one abstract reports small equal-compute gains for debate and mixture-of-agents over self-consistency (**O**, CLM-016).

At fixed budget, more agents means less reasoning per turn. Kim et al. fit $T=2.72\,(n+0.5)^{1.724}$ turns ($R^2=0.974$ on aggregated means) and state that per-agent capacity becomes "prohibitively thin beyond 3–4 agents" (**O**, CLM-009).

**Quantitative Model / Derivation:**
*Debate tokens.* $n$ agents, $R$ rounds after the initial answers, question length $q$, answer length $a$, full concatenation:
$$\text{calls}=n(R+1),\qquad \text{input tokens}\approx n(R+1)\,q+n^{2}R\,a$$
*Step-by-step*: round 0 has $n$ calls reading $q$; each later round has $n$ calls reading $q$ plus all $n$ previous answers, $q+na$; summing gives the expression. The $n^2$ term is why Du et al. switched to summarization for many agents (**D**, CLM-024).

*Budget per turn.* With budget $B$ and $T(n)$ turns, tokens per turn $\approx B/T(n)$ (**D**, CLM-010).

*Majority vote.* For odd $n$ independent agents with accuracy $p>0.5$:
$$P_{maj}(n)=\sum_{k>n/2}\binom{n}{k}p^{k}(1-p)^{n-k}$$
Correlated errors pull $P_{maj}$ back toward $p$ (**D**, CLM-025); one abstract describes this as a "shared-failure floor" (**O**, CLM-017).

**Worked Example:**
*Inputs (synthetic).* Prices 3 USD per million input and 15 USD per million output tokens. SAS: 10,000 input and 1,500 output tokens at success 0.44. MAS: 32,000 and 5,500 at 0.55 (CLM-035). Debate: $q=300$, $a=200$, $R=2$ (CLM-024).

*Step 1: cost per success* (**D**, CLM-035).
- SAS: $10{,}000\times3\times10^{-6}+1{,}500\times15\times10^{-6}=0.030+0.0225=0.0525$ USD per run; $0.0525/0.44=0.119$ per success.
- MAS: $0.096+0.0825=0.1785$ per run; $0.1785/0.55=0.325$ per success.
- Multipliers: $3.4\times$ per run, $2.72\times$ per success.

*Step 2: debate.* $n=3$: 9 calls, $9\times300+9\times2\times200=6{,}300$ input tokens. $n=5$: 15 calls, $4{,}500+10{,}000=14{,}500$. Single agent: 1 call, 300. Majority-of-3: 3 calls, 900. Going from 3 to 5 debaters multiplies input by $2.3\times$; the $n^2Ra$ term is 57% of the $n=3$ total and 69% of the $n=5$ total.

*Step 3: budget thinning* (**D**, CLM-010). At $B=4{,}800$: $T(1)=5.5$, $T(3)=23.6$, $T(5)=51.4$ turns, so about 877, 204, and 93 tokens per turn.

*Step 4: voting* (**D**, CLM-025). $p=0.6$: $n=1$: 0.600; $n=3$: 0.648; $n=5$: 0.683. If half the time all five agents share one outcome: $0.5\times0.6+0.5\times0.683=0.641$.

*Interpretation and limits.* In this fixture the MAS is worth it only if a success is worth more than about $0.325$ USD and the 0.11 gain survives a budget-matched baseline (Lesson 37.1). Whether the "two rounds" in Du et al. includes the initial generation is unverified; if it does, their call count is 6, not 9.

**Knowledge Check:**
1. Why can a design with a 3.4× per-run cost have a smaller per-success multiplier?
2. Which term of the debate formula dominates as $n$ grows, and what does summarization change?
3. You have budget for 15 calls. Give two designs that spend it, and the baseline any debate result must beat.

**Guided Practice:**
For a design of your own, write the call and input-token formula, plug in measured or declared $q$, $a$, rounds, and agent count, and compute cost per verified success against (a) the native SAS and (b) self-consistency with the same number of calls. State the success value at which the MAS breaks even.

**Feedback Contract:**
- *Expected Evidence*: (1) Success rose, so the denominator grew; per-success multiplier $=$ per-run multiplier $\times$ (SAS success / MAS success) $=3.4\times0.44/0.55=2.72$. (2) $n^2Ra$; summarization replaces $na$ with a bounded summary length. (3) e.g. 5 agents × 3 calls of debate, or 15-sample self-consistency; the latter is the baseline (**O**, CLM-023). Guided Practice: formula, inputs labeled measured or synthetic, both baselines, and a break-even value.
- *Common Failure*: Reporting tokens per call instead of per offered task; omitting output tokens; comparing debate with a single sample.
- *Diagnostic Hint*: What would the same number of calls buy as independent samples?
- *Concept to Revisit*: Cost per Verified Success (Lesson 22.1).

**Learning Outcome:**
Derive token, call, and cost-per-success multipliers for a multi-agent design and compare them with single-agent and voting baselines.

*(Effort: 35m instruction, 25m practice)*

---

### Lesson 37.6 — Evaluating Multi-Agent Trajectories

**Engineering Question:**
When a multi-agent run fails, how do we determine which agent and which step caused it, and how much should we trust that attribution?

**Concepts & Definitions:**
- **Multi-agent trajectory**: the interleaved, timestamped record of every agent's inputs, outputs, tool calls, handoffs, and returns for one task, with a stable task ID and per-agent IDs.
- **Failure attribution**: identifying the responsible agent and the decisive error step, the earliest step whose correction would have prevented failure. Who&When annotates logs from 127 MAS with both (**O**, CLM-029).
- **All-at-once, step-by-step, binary search**: three judge-based attribution methods that read the full log once, walk it step by step, or bisect it. They trade agent-level accuracy, step-level accuracy, and token cost (**O**, CLM-029).
- **Counterfactual replay**: re-run the trajectory from a candidate step with a corrected message, holding the rest fixed, and see if the task succeeds. Needs a replayable environment (Module 14).
- **End-state evaluation**: grading the final state rather than every turn. Anthropic prefers it for agents that mutate state and used a single LLM-judge call with a rubric (factual accuracy, citation accuracy, completeness, source quality, tool efficiency) to start (**O**, CLM-004).

**Mechanism Explanation:**
Trajectory evaluation for one agent (Lesson 12.6) asks whether the episode succeeded and what it cost. With several agents, a failure also needs an owner, because the fix differs: a planner error calls for a planning change, a worker error for a tool or prompt change, a handoff error for a context-regime change. Attribution is itself a measurement. In Who&When the best method identified the responsible agent 53.5% of the time but the decisive step only 14.2%; some methods were below random, and o1 and R1 did not reach practical usability (**O**, CLM-029). MAST's o1 annotator reached κ = 0.77 against humans for coarser failure-mode labels (**O**, CLM-026).

So validate attribution before acting on it: seed known decisive errors into synthetic trajectories, run each method, and measure accuracy with intervals. Counterfactual replay may localize more reliably at higher cost (**H**, CLM-039).

**Quantitative Model / Derivation:**
For a method that is correct on $k$ of $n$ seeded trajectories, report $\hat{a}=k/n$ with a Wilson 95% interval:
$$\frac{\hat a+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat a(1-\hat a)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},\qquad z=1.96$$
*Where*: $n$ = seeded trajectories; $k$ = correct attributions; $\hat a$ = observed accuracy. The Wilson interval stays inside $[0,1]$ for small $n$ and accuracies near 0 or 1, where the plain normal interval does not.

*Cost.* Binary search over $L$ steps needs about $\lceil\log_2 L\rceil$ judge calls; step-by-step needs up to $L$; all-at-once needs 1 call that reads all $L$ steps. Replay needs one re-run per candidate step.

**Worked Example:**
*Inputs (synthetic).* 50 trajectories of 20 steps, each with one seeded decisive error. An all-at-once judge names the right agent in 27 and the right step in 7.
- Agent level: $\hat a=27/50=0.54$, Wilson interval $[0.404,\,0.670]$.
- Step level: $\hat a=7/50=0.14$, interval $[0.070,\,0.262]$.
- Binary search on 20 steps: $\lceil\log_2 20\rceil=5$ judge calls per trajectory.

*Interpretation and limits.* With these intervals, a step-level attribution is wrong most of the time; acting on it would usually fix the wrong step. Agent-level attribution is a better triage signal but still close to a coin flip at the lower bound. These synthetic rates resemble Who&When's published 53.5% and 14.2% only by construction; your rates must be measured on your system.

**Knowledge Check:**
1. Why is agent-level attribution easier than step-level?
2. A judge blames the writer agent for 70% of failures. What experiment checks that before you rewrite the writer?
3. When is end-state evaluation insufficient for a multi-agent system?

**Guided Practice:**
Take 30 synthetic trajectories from LAB C with a known decisive step. Run all-at-once and binary-search judging (scripted mock judge or small local model) and record accuracy at both levels with Wilson intervals and calls per trajectory. Then run counterfactual replay on 10 and compare.

**Feedback Contract:**
- *Expected Evidence*: (1) An agent covers many steps, so a coarse guess hits more often; the decisive step must be the earliest causal one, and later steps often look wrong too. (2) Seeded-error validation of the judge on your trajectories, or counterfactual replay on a sample. (3) When the final state is correct by luck, when intermediate effects (Module 14) matter, or when a fix needs to know which agent failed. Guided Practice: per-method accuracy at both levels with intervals and cost, plus a statement of whether replay was deterministic enough to compare.
- *Common Failure*: Treating a judge's blame label as ground truth; reporting attribution accuracy without an interval or a seeded ground truth.
- *Diagnostic Hint*: If you correct the blamed step and replay, does the run succeed?
- *Concept to Revisit*: Attribution Is a Measurement (Lesson 15.3 graders as instruments).

**Learning Outcome:**
Evaluate multi-agent trajectories and attribute failures to agents and steps with a method whose accuracy has been measured.

*(Effort: 40m instruction, 30m practice)*

---

## 05 Literature & Production Source Map

All entries were opened on 2026-11-09/10. Each *Scope* line says what this module uses the source for and nothing more.

**REFERENCE / BASELINE**
- *Why Do Multi-Agent LLM Systems Fail?* (Cemri et al., arXiv:2503.13657, v3 2025-10-26). — [arXiv](https://arxiv.org/abs/2503.13657)
  - *Read*: Sections 1, 3–5; Appendices A, B, H.
  - *Scope*: The MAST taxonomy and its prevalence on seven 2023–24 frameworks (**O**, CLM-026); failure rates and modest intervention gains (**O**, CLM-027). No single-agent comparison.
- *Which Agent Causes Task Failures and When?* (Zhang et al., arXiv:2505.00212, v3). — [arXiv](https://arxiv.org/abs/2505.00212)
  - *Read*: abstract; Sections 3–4 skimmed.
  - *Scope*: The Who&When attribution benchmark and its agent- and step-level accuracies (**O**, CLM-029).
- *Improving Factuality and Reasoning in Language Models through Multiagent Debate* (Du et al., arXiv:2305.14325, 2023). — [arXiv](https://arxiv.org/abs/2305.14325)
  - *Read*: Section 3; Tables 1–2. *Scope*: origin of the debate pattern; baselines are not compute-matched (**O**, CLM-020).
- *More Agents Is All You Need* (Li et al., arXiv:2402.05120, TMLR 2024). — [arXiv](https://arxiv.org/abs/2402.05120)
  - *Read*: Sections 1, 4 skimmed. *Scope*: sampling-and-voting as the equal-compute baseline (**O**, CLM-023).
- *Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?* (Wang et al., arXiv:2402.18272, 2024; venue not verified) and *Should we be going MAD?* (Smit et al., arXiv:2311.17371, v3 2024; venue not verified). — [arXiv](https://arxiv.org/abs/2402.18272) · [arXiv](https://arxiv.org/abs/2311.17371)
  - *Scope*: negative results for discussion and debate against well-prompted single agents and self-consistency at GPT-3.5 (**O**, CLM-021, CLM-022).

**CURRENT DEFAULT** (scoped to the named system; no adoption survey was opened)
- *How we built our multi-agent research system* (Anthropic Engineering, published 2025-06-13). — [anthropic.com](https://www.anthropic.com/engineering/multi-agent-research-system)
  - *Read*: full text.
  - *Scope*: one vendor's production orchestrator–worker research agent: its unmatched 90.2% internal-eval gain, BrowseComp variance attribution, token multiples, and operational practices (**O**, CLM-001, CLM-002, CLM-003, CLM-004). Not evidence of an industry default.
- The single-agent-first baseline taught in this module (budget-matched comparison before adoption, explicit context regime per boundary, independent verification, validated attribution) is this module's recommendation, derived in Lessons 37.1–37.6. It is not a surveyed default.

**WORKLOAD-DEPENDENT**
- *Don't Build Multi-Agents* (Walden Yan, Cognition, 2025-06-12). — [cognition.ai](https://cognition.ai/blog/dont-build-multi-agents)
  - *Read*: full text. *Scope*: the write-coupled position (share full traces; implicit decisions conflict) (**O**, CLM-019). Argument by example, no measurements. Pair it with the Anthropic post: the two describe different task couplings.
- Topology choice, worker count, context regime, verifier placement, debate rounds, and attribution method all depend on decomposability, coupling, and task value.

**FRONTIER**
- *Towards a Science of Scaling Agent Systems* (Kim et al., arXiv:2512.08296, v3 2026-04-08). — [arXiv](https://arxiv.org/abs/2512.08296)
  - *Read*: abstract; Sections 1, 3, 4.1, 4.4; Tables 4–5.
  - *Scope*: equal-budget results across six benchmarks, the ~45% capability ceiling, Table 5 overhead and amplification, and the turn power law (**O**, CLM-006, CLM-007, CLM-008, CLM-009). One group's study.
- *Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets* (Tran and Kiela, arXiv:2604.02460, v2, preprint). — [arXiv](https://arxiv.org/abs/2604.02460)
  - *Read*: Sections 3–5; Tables 1, 4. *Scope*: DPI argument, equal-budget negative result, noise crossover, budget-control artifacts (**O**, CLM-012, CLM-013).
- Abstract only, figures limited to the abstract: Xu et al., arXiv:2601.12307 (**O**, CLM-014); Dylan, Brennan et al., arXiv:2609.04217 (**O**, CLM-015); Wunderlich et al., arXiv:2605.01566 (**O**, CLM-016); Anchan and Sen, arXiv:2610.06069 (**O**, CLM-017); Li, arXiv:2601.04748 (**O**, CLM-018).

**LEGACY**
- AutoGen (Wu et al., arXiv:2308.08155) and MetaGPT (Hong et al., arXiv:2308.00352), abstract only: conversational and SOP-pipeline framework designs, historical reference (**O**, CLM-040).
- `langchain-ai/langgraph-supervisor-py`: archived; its README at the pinned commit recommends the supervisor pattern via tools instead (**O**, CLM-032). Used below as a reference implementation of shared message state.
- Insufficient practices: adopting a MAS from an unmatched comparison; debate without a self-consistency baseline; an LLM critic counted as verification; blame labels without seeded validation.

**PRODUCTION SOURCE TRACE**
- Repository: `openai/openai-agents-python`. Revision: `d68961657843c7d81dfc25834484f8aac2957eb3` (commit date 2026-11-10). Verified 2026-11-10, static read; nothing executed.
  - Files/symbols: `src/agents/handoffs/__init__.py::{Handoff, Handoff.default_tool_name, HandoffInputData}`; `src/agents/run_internal/turn_resolution.py::execute_handoffs`; `src/agents/run_config.py::RunConfig.nest_handoff_history`; `src/agents/handoffs/history.py::default_handoff_history_mapper`; `src/agents/extensions/handoff_filters.py::remove_all_tools`; `src/agents/agent.py::Agent.as_tool`.
  - Entry path: model emits `transfer_to_<agent>` → `execute_handoffs` → multiple-handoff check, `run_handoffs[0]` → `input_filter` or nesting → `NextStepHandoff(new_agent)`. For as-tool: tool call → nested `Runner.run` → `final_output` string.
  - Observed (**O**, CLM-030, CLM-031): see Lesson 37.2 traces 1–2.
  - Generalizable: PARTIAL. Tool-call-as-transfer and isolated-return are common patterns; names and defaults are this SDK's.
- Repository: `langchain-ai/langgraph-supervisor-py` (archived). Revision: `88859b34017ac3569bbd4a3092c7e77593a0a960` (commit date 2026-07-15). Verified 2026-11-10, static read.
  - Files/symbols: `langgraph_supervisor/supervisor.py::{create_supervisor, _make_call_agent, OutputMode}`; `langgraph_supervisor/handoff.py::{create_handoff_tool, _remove_non_handoff_tool_calls}`.
  - Observed (**O**, CLM-032): see Lesson 37.2 trace 3.
  - Not traced: `langgraph` core `Command` and `Send` semantics; described from call sites only.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow the loop:
$$\text{PREDICT} \to \text{BUILD} \to \text{MEASURE} \to \text{EXPLAIN} \to \text{BREAK} \to \text{IMPROVE} \to \text{FALSIFY}$$
Labs run on a scripted mock LLM whose per-call correctness, output length, and error type are drawn from seeded, declared distributions, so results are reproducible on CPU. A small local model may replace the mock for a COLAB extension. All task data is **synthetic** and labeled; no result from a mock describes a real model.

### LAB A — Equal-Budget Baseline and the Capability Ceiling
- **Objective**: Build a harness that runs SAS-native, SAS-budget-matched, orchestrator–worker, and debate arms on synthetic tasks whose decomposability and base difficulty are controlled, and measures success, tokens, calls, and wall time per offered task.
- **Pre-Registered Hypothesis**: On decomposable tasks the MAS retains a gain over the budget-matched SAS; on sequential tasks the gain over the native SAS disappears or reverses once budget is matched (**H**, CLM-036). State the minimum practically important difference before running.
- **Independent Variables**: Architecture (4 arms), token budget, task decomposability (independent subtasks vs a chain), native SAS success level (0.3–0.8 via mock difficulty), agent count $n\in\{1,3,5\}$.
- **Dependent Variables**: Verified success with intervals, measured tokens and calls per offered task, success per 1K tokens, cost per success under declared prices, critical-path wall time.
- **Break & Falsify**: Give the SAS arm a requested budget that the mock silently caps (the API artifact reported in CLM-013) and show how matching on requested rather than measured tokens inflates the MAS gain. A sequential-task result where the MAS still beats the matched SAS outside the interval falsifies the hypothesis for that setup.
- **Alignment**: Lessons 37.1, 37.2 (fan-out latency), 37.5.
- **Hardware**: CPU (optional COLAB extension with a small local model).
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB B — Context-Passing Regimes and Seeded-Fact Recall
- **Objective**: Implement forward-all, shared state with `full_history` and `last_message` semantics, isolated return, and summarized handoff; measure what each duplicates and loses; then map each regime to its pinned implementation.
- **Pre-Registered Hypothesis**: Input tokens rank `full_history` > forward-all ≥ `last_message` > isolated return, while seeded-fact recall at the final agent ranks roughly the reverse. Separately, isolated parallel writers produce more convention conflicts than a single writer or writers given a shared decision log (**H**, CLM-037).
- **Independent Variables**: Context regime, number of rounds, appended tokens per round, summary length, presence of a shared decision log, number of parallel writers.
- **Dependent Variables**: Input tokens per call and per task (compare with the Lesson 37.3 formula), fraction of seeded facts reproduced verbatim, convention-conflict rate from a deterministic checker, task success.
- **Break & Falsify**: Apply a `remove_all_tools`-style filter before a citation step and measure citation verifiability. If conflict rates do not differ between isolated and shared regimes on a coupled task, the conflict hypothesis is falsified for that task.
- **Alignment**: Lessons 37.2 and 37.3. The pinned trace in Section 05 is this lab's reference; its time is counted under `source_trace`.
- **Hardware**: CPU (trace component is TRACE).
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB C — Error Cascades and Verifier Correlation
- **Objective**: Build a $k$-stage multi-agent pipeline with seeded stage errors, add verifiers of three kinds, and measure end-to-end success, catch rate, correlation, and amplification under two definitions. Code every failed trace with MAST categories.
- **Pre-Registered Hypothesis**: A verifier sharing the producer's prompt and seed has a lower catch rate on producer errors than a deterministic checker, and end-to-end success falls below the independence prediction $p_{eff}^k$ (**H**, CLM-038).
- **Independent Variables**: Stage count $k\in\{3,6,10\}$, per-stage error rate, verifier type (none, same-context LLM, different-context LLM, deterministic), injected error correlation.
- **Dependent Variables**: End-to-end success versus the Lesson 37.4 prediction, per-verifier catch rate, producer–verifier error correlation, trace-level and task-level amplification, MAST category counts with inter-coder agreement if two people code.
- **Break & Falsify**: Set correlation to zero and confirm the measurements match $p_{eff}^k$ within sampling error; a mismatch there means the harness, not the hypothesis, is wrong. If the same-context verifier matches the independence prediction at nonzero correlation, the hypothesis is falsified.
- **Alignment**: Lesson 37.4; its failed traces feed LAB D.
- **Hardware**: CPU.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB D — Trajectory Attribution Under Seeded Ground Truth
- **Objective**: Log multi-agent trajectories with stable task, agent, and step IDs; seed one decisive error per trajectory; compare all-at-once, binary-search, and counterfactual-replay attribution for accuracy and cost.
- **Pre-Registered Hypothesis**: Agent-level accuracy exceeds step-level for every judge-based method, and replay achieves higher step-level accuracy than judge-based methods at higher cost (**H**, CLM-039).
- **Independent Variables**: Attribution method, trajectory length, number of agents, decisive-step position, mock-judge noise.
- **Dependent Variables**: Agent- and step-level accuracy with Wilson intervals, judge or replay calls per trajectory, and the fraction of cases where the fix chosen from the attribution makes the replayed run succeed.
- **Break & Falsify**: Insert a later, more visible error after the decisive one; measure how often each method blames the later step. If replay cannot be made deterministic (state the source of nondeterminism), report the comparison as inconclusive rather than favorable.
- **Alignment**: Lesson 37.6 and Incident 37.1.
- **Hardware**: CPU (optional COLAB extension with a small local judge model).
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

---

## 07 Break / Incident Scenarios

### Incident 37.1: The Report That Got Worse and More Expensive
- **Incident Symptoms**:
  An internal research assistant uses an orchestrator–worker design: a lead agent, up to five search workers, and a writer agent that receives a handoff. A release changes three things at once: workers return through shared state in `full_history` mode instead of `last_message`; the writer handoff now applies a tool-stripping filter "to save tokens"; and the lead's prompt raises the maximum worker count from 3 to 5. Over the following week (all figures synthetic):
  - Mean tokens per offered task rise $2.6\times$; P95 wall time rises from 70 s to 140 s.
  - Rubric-judged citation accuracy falls from 0.81 to 0.66; overall pass rate falls from 0.62 to 0.55.
  - The LLM judge attributes 70% of failures to the writer agent.
  - Mean workers spawned per task rises from 2.4 to 4.6, including on single-fact queries.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: At least four, for example: (A) the tool-stripping filter removed the source text the writer needs to cite (context loss at handoff; MAST loss of history / information withholding) (**O**, CLM-030, CLM-026); (B) `full_history` bloat pushed the lead's context to a length it uses poorly, degrading planning (**D**, CLM-033); (C) over-spawning on simple queries added cost and conflicting worker claims (**O**, CLM-004); (D) the judge's writer attribution is wrong (**O**, CLM-029); (E) an unrelated change in a search tool or model version.
  2. *Rank Hypotheses by Initial Plausibility*: Use the symptoms without treating any as conclusive. Citation accuracy falling while tokens rise fits A and B; the spawn count fits C; the 70% writer blame fits A but is also what D would produce.
  3. *Identify Missing Evidence*: Per-task trajectories with agent and step IDs; the exact message list the writer received before and after the filter; lead input tokens per call; workers spawned by query class; search-tool and model versions; judge rubric and judge version; offered-task denominators.
  4. *Design Discriminating Tests*: Replay a fixed sample of failed tasks with one change reverted at a time (filter, output mode, worker cap), holding tools and models fixed. Check whether the cited sources were present in the writer's input. Validate the judge's attribution on 30 tasks with seeded citation errors. State the outcome that would falsify each hypothesis (for example, A is falsified if restoring tool outputs leaves citation accuracy unchanged).
  5. *Execute Causal Diagnosis*: Rank supported mechanisms; expect interacting causes, since the release changed three things.
  6. *Prescribe Immediate Mitigation & Long-Term Intervention*: Immediate: revert the filter and restore `last_message` while keeping a citation-evidence channel (artifact store), cap workers by query class. Long-term: change one boundary at a time behind a budget-matched evaluation; add a deterministic citation check against the source text instead of a same-model critic.
  7. *Remeasurement & Post-Mortem Defense*: Predeclare targets for citation accuracy, pass rate, tokens and cost per verified success, and P95 wall time over offered tasks; rerun the same task sample and the judge-validation set.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Architecture Transfer Problem: Support-Engineering Research Assistant
You own the agent design for an internal assistant that answers support engineers' questions. Two task classes arrive:
1. **Breadth research**: compare behavior across several product versions, documents, and tickets. Subtasks are largely independent.
2. **Sequential troubleshooting**: follow a diagnostic chain where each step depends on the previous result.

A proposal says to replace the current single agent with an orchestrator–worker design for all traffic. You must decide what to deploy, show the evidence, and define how you would know you were wrong.

**Workload Fixture (SYNTHETIC — exercise assumptions, not measurements of any real model, product, or price list; registry CLM-041):**

| Input | Breadth (55% of tasks) | Sequential (45% of tasks) |
|---|---|---|
| Native SAS: tokens per task, success | 18,000; 0.38 | 14,000; 0.62 |
| Budget-matched SAS: tokens, success | 70,000; 0.45 | not run |
| Orchestrator–worker: tokens, success | 70,000; 0.57 | 61,000; 0.58 |
| Tasks per arm | 400 | 400 |
| Blended price | 5 USD per million tokens | same |
| Unknown by design | tail latency, attribution accuracy on your traces, verifier correlation | same |

**Constraints**:
- Cost per verified success over all offered tasks may rise at most $2.5\times$ over the current all-SAS design.
- Overall verified success must not fall.
- Every design claim must name its evidence class (fixture, measurement, or assumption).

**Required Deliverables**:
1. **Baseline analysis**: For each slice, compute MAS − native SAS and MAS − matched SAS (where available) with intervals, and state which mechanism (spend, fresh context, breadth) your numbers support. Identify the missing arm and what it would cost to run.
2. **Topology and context design**: Choose a topology per slice. For the multi-agent path, specify control ownership, the context regime at every boundary, what each boundary deletes, and how worker evidence reaches the final answer.
3. **Cost model**: Compute cost per task and per verified success for all-SAS, all-MAS, and a routed design (MAS for breadth, SAS for sequential). Recompute under 10% misrouting in both directions.
4. **Error and verification plan**: Where can an error cross a boundary unchecked? Place verifiers, state their independence from producers, and give the predicted end-to-end effect with your assumptions.
5. **Trajectory evaluation and attribution**: Specify trajectory logging, the attribution method, and how you will validate it before acting on its output.
6. **Pinned source trace**: Map your chosen design to the pinned handoff and agent-as-tool code (Section 09 artifact).
7. **Rejection of an inappropriate design**: Reject at least one plausible alternative (for example all-MAS, or debate on troubleshooting) using the fixture or your measurements.
8. **Falsification and rollback**: State the workload change that would invalidate your design (for example breadth tasks becoming coupled) and the metrics and thresholds that would trigger rollback.

---

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Submit a verified trace of the handoff and agent-as-tool paths. The reference trace uses `openai/openai-agents-python` commit `d68961657843c7d81dfc25834484f8aac2957eb3`; the supervisor comparison uses `langchain-ai/langgraph-supervisor-py` commit `88859b34017ac3569bbd4a3092c7e77593a0a960` (archived). The trace must map:
1. How a handoff is exposed to the model and where `execute_handoffs` resolves it, including what happens to a second handoff in the same turn.
2. What history the receiver sees by default, where `input_filter` and `nest_handoff_history` change it, and what `remove_all_tools` deletes.
3. How `Agent.as_tool` isolates the worker's context and what it returns to the parent, including the default error behavior.
4. How `create_supervisor` wires workers back to the supervisor and what `output_mode` keeps in shared state.
5. For each finding: repository, commit, file, symbol, static versus executed, and generalizable YES / NO / PARTIAL.

### Reference Checks for Mastery Deliverables 1 and 3 (fixture inputs only)
Reviewers use these to check arithmetic, not to grade a design. A submission with different declared inputs is checked against its own.

- **Cost per task** (tokens × $5\times10^{-6}$): breadth SAS 0.090, matched SAS 0.350, MAS 0.350; sequential SAS 0.070, MAS 0.305 USD.
- **Cost per verified success**: breadth SAS $0.090/0.38=0.237$, matched SAS $0.350/0.45=0.778$, MAS $0.350/0.57=0.614$; sequential SAS $0.070/0.62=0.113$, MAS $0.305/0.58=0.526$.
- **Breadth gains** (independent arms, $n=400$): MAS − native SAS $=0.19$; MAS − matched SAS $=0.12$, $SE=0.0351$, 95% interval $[0.051,\,0.189]$. The gain survives budget matching in this fixture.
- **Sequential**: MAS − SAS $=-0.04$ at $4.4\times$ the tokens; no matched SAS is needed to reject MAS here.
- **Blended designs** (weights 0.55/0.45):
  - All-SAS: cost $0.55\times0.090+0.45\times0.070=0.081$, success $0.55\times0.38+0.45\times0.62=0.488$, per success $0.166$.
  - All-MAS: cost $0.330$, success $0.5745$, per success $0.574$ ($3.46\times$ all-SAS; violates the $2.5\times$ constraint).
  - Routed: cost $0.55\times0.350+0.45\times0.070=0.224$, success $0.55\times0.57+0.45\times0.62=0.5925$, per success $0.378$ ($2.28\times$; meets both constraints).
- **10% misrouting both ways**: cost $0.220$, success $0.580$; per success $0.380$. Routing errors cost little here because the slices' SAS–MAS success gaps partly offset; a submission must show this computation, not assume it.

### Rubric Dimensions
- **Baseline Discipline** (Deliverable 1):
  - *Insufficient*: Compares MAS with the native SAS only, or matches requested rather than measured budgets.
  - *Competent*: Reports gains against a budget-matched baseline with intervals per slice and names the supported mechanism.
  - *Strong*: Also identifies the missing arm, its cost, and the result that would change the decision.
- **Mechanistic Reasoning**: *Insufficient* names topologies without control or visibility semantics. *Competent* states control owner and context regime per boundary and what each deletes. *Strong* ties each boundary to the pinned code and to a measured recall or conflict result.
- **Quantitative Reasoning**: *Insufficient* quotes multipliers without assumptions. *Competent* derives tokens, calls, latency, and cost per verified success with declared inputs. *Strong* compares predictions with LAB measurements and explains discrepancies (correlation, re-read growth, budget artifacts).
- **Experiment Design**: *Insufficient* runs uncontrolled comparisons. *Competent* controls decomposability and budget and pre-registers predictions. *Strong* constructs conditions that discriminate spend, fresh-context, and breadth explanations.
- **Failure Diagnosis**: *Insufficient* accepts a judge's blame label. *Competent* forms competing hypotheses and validates attribution against seeded errors. *Strong* isolates interacting changes by one-at-a-time replay.
- **Architecture Defense**: *Insufficient* recommends multi-agent by default. *Competent* routes by task class with cost and success trade-offs. *Strong* gives falsifiers, rollback triggers, and misrouting sensitivity.

---

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| **Budget-matched adoption decision** | Lesson 37.1 | Lesson 37.1 Guided Practice; LAB A | Mastery Deliverable 1; Baseline Discipline rubric | LAB A report with matched arms and intervals |
| **Topology selection** | Lesson 37.2 | Lesson 37.2 Knowledge Checks; LAB A (fan-out latency) | Mastery Deliverables 2 and 7 | Topology and context design per slice |
| **Pinned implementation trace** | Lesson 37.2 traces; Section 05 | Lesson 37.2 Guided Practice; LAB B trace component | Section 09 Required Artifact items 1–5; Mastery Deliverable 6 | Source trace pinned to `d6896165…` and `88859b34…` |
| **Context-passing regimes** | Lesson 37.3 | Lesson 37.3 Guided Practice; LAB B | Mastery Deliverable 2; Incident 37.1 steps 1, 4 | Token versus seeded-fact recall table; conflict-rate results |
| **Error propagation and verification** | Lesson 37.4 | Lesson 37.4 Guided Practice; LAB C | Mastery Deliverable 4; Incident 37.1 step 6 | Predicted versus measured success; catch rate and correlation |
| **Cost multipliers** | Lesson 37.5 | Lesson 37.5 Guided Practice; LAB A | Mastery Deliverable 3; Section 09 Reference Checks | Cost model with per-success figures and misrouting sensitivity |
| **Trajectory attribution** | Lesson 37.6 | Lesson 37.6 Guided Practice; LAB D | Mastery Deliverable 5; Incident 37.1 steps 3–5 | Attribution accuracy with Wilson intervals and cost |
| **Multi-cause incident diagnosis** | Lessons 37.3–37.6 | LAB C, LAB D | Incident 37.1 steps 1–7; Failure Diagnosis rubric | One-at-a-time replay results and remeasurement plan |

---

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 37 must be able to:
1. Refuse a multi-agent design until a budget-matched single-agent baseline has been measured on the target tasks.
2. Select a topology from task coupling, control ownership, and context needs, and trace it in pinned code.
3. Quantify what each inter-agent boundary duplicates and loses.
4. Model and measure error propagation with a declared amplification metric and test verifier independence.
5. Derive cost per verified success for single-agent, multi-agent, voting, and routed designs.
6. Attribute multi-agent failures with a method whose accuracy was measured on seeded ground truth.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: Every boundary between agents is a lossy, costly channel. A multi-agent design earns its place only when parallel breadth or fresh context buys more verified success than a single agent could buy with the same budget.
- **The Decision Path**:
  $$\text{Decomposable?} \to \text{Matched baseline} \to \text{Topology} \to \text{Context regime per boundary} \to \text{Independent verification} \to \text{Cost per success} \to \text{Validated attribution}.$$
- When a multi-agent system degrades, align per-agent trajectories, change one boundary at a time, and validate any blame label before acting. The next modules reuse this: Module 15 for release statistics on paired arms, Module 18 for what an agent may trust from another agent's output.

### Cost & Telemetry Hook

- [Lesson 22.1](../22-ai-economics/README.md): cost per verified offered task counts every agent's calls, including spawned workers whose output was discarded and failed rounds.
- [Lesson 22.5](../22-ai-economics/README.md): routing between single- and multi-agent paths is a router decision; misrouting cost is part of the design, as in the Section 09 reference checks.
- [Lesson 23.1](../23-observability-reliability/README.md): emit one span per agent call with agent ID, role, parent span, context-regime label, and input/output token counts, so per-boundary duplication is visible.
- [Lesson 23.2](../23-observability-reliability/README.md): propagate trace context across handoffs and parallel workers so one task's trajectory can be reassembled for attribution.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
