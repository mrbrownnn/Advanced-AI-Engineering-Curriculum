# Module 21 — Multimodal AI Systems

## 00 Why This Module Exists

A text request reaches the model as token IDs. An image, audio clip, or video reaches it only after decoding, resizing, cropping, tiling, feature extraction, encoding, bridging, and placement into the language model. Each step is defined by one architecture and one processor configuration. Two systems given the same 1920×1080 image can put 576, 1,948, 1,222, 2,691, or 32 modality positions in front of the decoder (Lesson 21.2). Memory, latency, batching, caching, and quality all follow from that pipeline, not from the input resolution.

The outline this module replaces asked learners to "explain the KV fragmentation caused by visual tokens." That presumes a conclusion before any telemetry exists. A multimodal OOM or first-token regression has several competing causes: encoder work, padding, configured capacity, engine allocation, framework-allocator fragmentation, and host preprocessing. Each makes a different prediction and needs its own measurement (**H**, CLM-012).

```text
media bytes --decode/canonicalize--> processor --> encoder --> bridge/projector
                                       |             |              |
                     (resize, crop, tile, mel, pad)  |   (MLP, merge, learned queries,
                                                     |    or cross-attention memory)
                                                     v
              placeholder replacement / cross-attention --> LM prefill --> decode --> output
      caches: processor output | encoder output | decoder prefix KV   (different identities)
      clocks: per-medium sample clocks --> shared reference clock --> output deadlines
```

**Ownership.** This module owns modality preprocessing, architecture-specific token accounting, multimodal batching and caching, cross-stream timing, quality/latency/memory boundaries, and multimodal diagnosis. KV internals stay in Module 03, scheduler policy in Module 04, kernels in Module 05, and retrieval in Modules 09–10. Model-family depth belongs to the satellite tracks at the repository root: `vision-language-models` (architectures, training, OCR, evaluation), `speech-ai` (ASR, TTS, streaming speech), and `generative-media` (diffusion and video generation). This module cross-references them and does not duplicate them. Disaggregated encoder placement across machines belongs to Module 20.

**Research cutoff:** 2026-09-30.

**Module Orientation**
- **Engineering Problem**: Predict, measure, and control the work, memory, and timing that multimodal inputs add to an inference system, and diagnose regressions without presupposing a cause.
- **What You Will Do**: Build a stage ledger for a multimodal request. Compute decoder-visible modality tokens and logical KV from pinned processor code for four architectures. Measure padding and cache behavior under mixed batches. Align audio and video on a shared clock. Run a five-way discriminating diagnosis of an OOM. Choose a representation budget inside a measured quality boundary.
- **Environment**: Python 3.10+, PyTorch, Transformers at the pinned revision (`d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a`), Pillow, NumPy. Labs A, C, and the accounting parts of B and D run on CPU. Optional: one NVIDIA GPU and vLLM at the pinned revision (`25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`) for live encoder-cache and memory measurements.
- **Evidence Rule**: Source observations are **O**, explicit derivations **D**, and measurement-dependent hypotheses **H**, each with a `CLM-###` from the research registry. All worked-example inputs are labeled synthetic or pinned-configuration values. Author-reported numbers are quoted only with their scope, and frontier results inspected at abstract level are labeled as such.

## 01 Baseline Assumptions

- **Latency decomposition and TTFT/ITL boundaries**: Module 04, Lesson 4.1. This module extends the stage ledger to media stages in Lesson 21.1.
- **Continuous batching and iteration-level scheduling**: Module 04, Lesson 4.2. Little's Law and its boundary conditions: Module 04, Lesson 4.4.
- **Logical KV bytes per token**: Module 03, Lesson 3.1 ($2\times L\times H_{kv}\times d_h\times$ bytes per element, with stated exclusions).
- **Fragmentation hierarchy (internal, external, allocator-level) and paged KV**: Module 03, Lessons 3.2–3.3.
- **Prefix-cache identity**: Module 03, Lesson 3.6. Token equality is necessary but not sufficient for reuse.
- **Rotary position embedding**: Module 01, Lesson 1.4. This module only uses the idea that position IDs can be assigned per axis.
- **Honest timing and profiler timelines**: Module 02, Lessons 2.3 and 2.6.
- **Framework allocated versus reserved memory**: Module 19, Lesson 19.2. Lesson 21.5 restates the parts it uses and adds allocator snapshots.
- **Paired evaluation, slices, and declared margins**: Module 00, Lesson 0.3; Module 15, Lessons 15.2 and 15.4.

Not assumed, and taught at first use: image preprocessing and patch embedding (21.2), audio log-mel frontends (21.1, 21.3), projectors, patch merging, and learned-query resamplers (21.2), media clocks and RTP timestamp mapping (21.4), and allocator snapshot evidence (21.5).

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
  instruction: 4.5h      # 45+50+45+45+45+40 min across Lessons 21.1-21.6
  guided_practice: 2h    # 20+25+20+20+20+15 min across Lessons 21.1-21.6
  labs: 13h              # LAB A 3h + LAB B 3.5h + LAB C 3h + LAB D 3.5h
  assessment: 3h         # Mastery transfer 2.5h + Incident 21.1 0.5h
  source_trace: 2h       # Lesson 21.2 guided trace, LAB B trace step, and the Section 09 artifact are one activity, counted once
  total: 24.5h
```

Each category is counted once. The source trace is not also counted in Lesson 21.2 practice or in LAB B. Security is SELECTIVE because cache identity includes tenant isolation (Lesson 21.3); media-borne attacks belong to Module 18.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Media Bytes $\to$ Decode/Canonicalize $\to$ Modality Processor (resize, crop, tile, pad; or resample, frame, mel) $\to$ Encoder (patch embedding, transformer) $\to$ Bridge (MLP projector, spatial merge, learned queries, or cross-attention memory) $\to$ Decoder-Visible Length $\to$ Logical KV and Prefill Work $\to$ Scheduling, Batching, Caching (processor / encoder output / prefix KV) $\to$ Stream Alignment (media clocks $\to$ reference clock $\to$ deadlines) $\to$ Output. Cross-cutting: the stage ledger (21.1), the quality–latency–memory boundary (21.6), and competing-hypothesis diagnosis (21.5).

## 04 Lessons

### Lesson 21.1 — The Multimodal Request Path and Its Stage Ledger

**Engineering Question:**
Where does time go between receiving media bytes and emitting the first output, and how is that measured without double-counting overlapping stages?

**Concepts & Definitions:**
- **Modality contract**: the full set of choices that turns media bytes into model input: decoder library and version, color/sample format, resize/crop/tile or resample/frame policy, normalization, padding, processor configuration, and model revision. Two requests are comparable only under the same contract.
- **Stage ledger**: a per-request record of start and end timestamps for each declared stage on one clock: receive, media decode, processor, queue, encoder, bridge/projector, placeholder insertion, prefill, first-token sampling, and output emission.
- **Serialized versus overlapping stages**: if stages run strictly one after another, elapsed time is their sum. If stages overlap, for example CPU preprocessing of image 2 while the GPU encodes image 1, first-output time is the critical path, not the sum (**D**, CLM-001).
- **Audio frontend**: a feature extractor that converts a waveform at a fixed sample rate into frames. In the pinned Whisper extractor, audio is 16 kHz, windows are 25 ms with a 10 ms hop, and input is padded or truncated to 30 s. The encoder then expects that fixed mel length (**O**, CLM-008).
- **Stage-specific behavior**: ModServe characterizes multimodal inference as preprocessing, encoding, and generation and reports, on its evaluated models and A100 setup, that batching and parallelism effects differ by stage and architecture (**O**, CLM-016). This is author-reported evidence for those configurations, not a universal profile.

**Mechanism Explanation:**
State: media bytes and a request record. Operations: each stage consumes the previous stage's output and appends a span to the ledger. Transition: the request becomes schedulable for prefill only when every modality item it needs at the scheduled positions has been encoded (or found in a cache, Lesson 21.3). Text-only TTFT decomposition from Module 04 is the special case with zero media stages. Encoder work, unlike text tokenization, is model compute and can contend with prefill and decode on the same device.

**Quantitative Model / Derivation:**
For serialized stages $s\in S$ with spans $d_s$:
$$T_{first}=\sum_{s\in S} d_s$$
For overlapping work, let each stage $s$ occupy interval $[a_s,b_s]$. Then
$$T_{first}=t_{emit}-t_{arrive},\qquad \sum_s d_s \ge T_{first}\ \text{when intervals overlap}$$
and the attributable contribution of a stage is its share of the critical path, measured from the timeline. *Assumptions*: one clock or a verified clock mapping; declared start and end boundaries for each stage (**D**, CLM-001).

where:
- $S$ is the set of declared stages; $d_s=b_s-a_s$ is the span (duration) of stage $s$, with start $a_s$ and end $b_s$.
- $t_{arrive}$ is when the request's bytes arrive; $t_{emit}$ is when the first output leaves; $T_{first}$ is the first-output time.
- The *critical path* is the chain of stages that the request was actually waiting on at each instant between $t_{arrive}$ and $t_{emit}$.

*Why the inequality holds.* $T_{first}$ is the length of the union of the busy intervals (plus any idle gaps). When two intervals overlap, the shared part is counted once in $T_{first}$ but twice in $\sum_s d_s$, so the sum can only overstate. In plain terms: summing a dashboard's per-stage times measures total work done, not how long the user waited.

**Worked Example (synthetic spans):**
*Input.* One request with one image, measured on a single clock: media decode 18 ms, processor 42 ms, queue 30 ms, encoder 95 ms, projector 2 ms, prefill 140 ms, first-token sampling and emission 8 ms. All stages serialize.
*Step 1.* $T_{first}=18+42+30+95+2+140+8=335$ ms.
*Step 2.* A second request carries two images. Preprocessing runs on one CPU worker at 42 ms per image; the encoder runs at 95 ms per image and can start an image once its preprocessing is done. Image 1: preprocess $[0,42]$, encode $[42,137]$. Image 2: preprocess $[42,84]$, encode starts at $\max(137,84)=137$ and ends at $232$.
*Result.* The media stages finish at 232 ms. Summing spans gives $2\times42+2\times95=274$ ms, which overstates the media path by 42 ms because preprocessing of image 2 is hidden under encoding of image 1.
*Check.* $274-232=42$ ms, exactly image 2's preprocessing span $[42,84]$, which lies inside image 1's encode interval $[42,137]$. The critical path is preprocess 1 (42) $\to$ encode 1 (95) $\to$ encode 2 (95) $=232$ ms.
*Interpretation and limits.* Adding a CPU worker would not shorten this path: image 2 already waits for the encoder, not for preprocessing. A faster encoder or a cache hit would. These values are synthetic; the method, not the numbers, transfers. Spans from different processes need a common clock before they are combined.

**Knowledge Check:**
1. A dashboard shows encoder time 95 ms and preprocessing 84 ms per request, but TTFT rose by only 95 ms after images were added. How is that possible?
2. Why is a 5 s audio clip not necessarily cheaper to encode than a 25 s clip under the pinned Whisper contract?

**Guided Practice:**
Instrument one image request and one audio request end to end. Record each declared stage on one monotonic clock, draw the timeline, and compute both the sum of spans and the critical path. Then add a second image and predict, before measuring, whether TTFT grows by the full per-image cost.

**Feedback Contract:**
- *Expected Evidence*: A ledger with declared boundaries per stage; a timeline figure; sum versus critical-path comparison; a written prediction and its check. KC1: overlap hides preprocessing under encoding. KC2: both are padded to the same 30 s mel length, so the encoder runs on the same input shape (CLM-008).
- *Common Failure*: Adding all spans from different threads and reporting the total as latency, or using an engine-side prefill timer as TTFT.
- *Diagnostic Hint*: Which stage was the request waiting on at every instant between arrival and first output?
- *Concept to Revisit*: Latency decomposition (Module 04, Lesson 4.1); honest timing (Module 02, Lesson 2.3).

**Learning Outcome:**
Instrument a multimodal request and attribute first-output time to stages on the critical path.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 21.2 — Encoders, Bridges, and Token Accounting by Architecture

**Engineering Question:**
Given an input image and a pinned model, how many positions does the decoder see, and how much logical KV do they occupy?

**Concepts & Definitions:**
- **Patch embedding**: the Vision Transformer splits an image into fixed-size patches, linearly embeds each flattened patch, prepends a class token, and adds position embeddings (**O**, CLM-002). That patch sequence is the *encoder's* input length, not a decoder-visible count.
- **Image processor**: code and configuration that turn decoded pixels into encoder input: resize, center crop, padding, tiling, normalization. The checkpoint's processor configuration, not the class defaults alone, determines the result.
- **Projector**: a small network mapping encoder width to decoder width. In pinned LLaVA it is Linear–GELU–Linear (**O**, CLM-003).
- **AnyRes tiling**: LLaVA-NeXT selects a grid resolution from the original aspect ratio, encodes a downscaled base image plus tiles, and inserts newline features per row (**O**, CLM-004).
- **Dynamic resolution with spatial merge**: Qwen2-VL resizes to multiples of patch size times merge size within pixel bounds and merges each $2\times2$ patch group into one decoder position (**O**, CLM-005).
- **Learned-query resampler**: BLIP-2's Q-Former uses a fixed set of learned queries, 32 of width 768 in its experiments, so the decoder-visible count does not grow with resolution (**O**, CLM-006).
- **Placeholder replacement**: the processor expands an image token into $N$ placeholder tokens; the model checks that placeholders and features match and scatters features into those positions (**O**, CLM-003).
- **Decoder-visible length versus other memory**: only positions processed by decoder self-attention occupy decoder KV under the Module 03 formula. Encoder activations and cross-attention memory are different resources. Resolution alone cannot determine decoder KV bytes (**D**, CLM-007).

**Mechanism Explanation — Pinned Transformers Trace (O, CLM-003, CLM-004, CLM-005, CLM-006):**
At `huggingface/transformers` commit `d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a` (static inspection, 2026-09-30):
1. `models/llava/processing_llava.py::LlavaProcessor.replace_image_token` computes `(height // patch_size) * (width // patch_size) + num_additional_image_tokens`, then subtracts 1 when `vision_feature_select_strategy == "default"`.
2. `models/llava/modeling_llava.py::LlavaModel.get_image_features` takes hidden states at `vision_feature_layer`, drops the CLS position under `"default"`, and calls `LlavaMultiModalProjector.forward` (Linear → activation → Linear). `LlavaModel.get_placeholder_mask` raises if placeholder count × hidden size differs from feature elements. `LlavaModel.forward` then calls `masked_scatter`.
3. `models/llava_next/processing_llava_next.py::LlavaNextProcessor._get_number_of_features` calls `select_best_resolution`, then `_get_unpadded_features` to remove aspect-ratio padding rows or columns, and adds newline features plus base features.
4. `models/llava_next/image_processing_llava_next.py::LlavaNextImageProcessor._get_image_patches` resizes and pads to the best resolution, divides it into crops, and prepends the resized original; `_pad_for_batching` pads the crop dimension to the largest crop count in the batch.
5. `models/qwen2_vl/image_processing_qwen2_vl.py::smart_resize` rounds height and width to multiples of `factor = patch_size * merge_size` and rescales into `[min_pixels, max_pixels]`. `Qwen2VLProcessor.replace_image_token` divides `image_grid_thw.prod()` by `merge_size**2`. `modeling_qwen2_vl.py::PatchMerger.forward` reshapes $4\times$ `context_dim` into one MLP input.
6. `models/blip_2/modeling_blip_2.py` expands `query_tokens` (`num_query_tokens`, default 32) and projects Q-Former output with `language_projection`; `Blip2Processor` prepends that fixed number of image tokens.

**Quantitative Model / Derivation:**
With patch size $p$, the pinned formulas are:
- LLaVA: $N=(H_c/p)(W_c/p)+e-\mathbb{1}[\text{default}]$, where $H_c\times W_c$ is the processed crop and $e$ the configured extra tokens.
- LLaVA-NeXT: $N=U+R+(b+e)-\mathbb{1}[\text{default}]$, with $U$ the unpadded tile features, $R$ the newline count (rows of $U$), and $b=(H_c/p)(W_c/p)$ the base features.
- Qwen2-VL: $N=(H'/p)(W'/p)/m^2$ with $(H',W')$ from `smart_resize`.
- BLIP-2: $N=Q$ learned queries.

Then, for a decoder-only fusion, logical decoder KV is $N\times$ bytes per position from Module 03 (**D**, CLM-007).

where:
- $N$: decoder-visible modality positions (placeholder tokens the decoder attends over).
- $p$: patch side in pixels (14 here). $H_c,W_c$: height and width of the processed crop that reaches the encoder; $H_c/p$ and $W_c/p$ are patches per column and per row.
- $e$: extra encoder tokens (`num_additional_image_tokens`, the CLS token here). $\mathbb{1}[\text{default}]$ is 1 when `vision_feature_select_strategy == "default"` (CLS dropped), else 0. With $e=1$ and the default strategy, $+e-1$ cancels: the decoder sees exactly the patch features.
- $U,R,b$ (LLaVA-NeXT): unpadded tile features, newline features, base-image features. $b$ here is a count, not bytes; bytes per position are written $b_{pos}$ in Lessons 21.5–21.6.
- $H',W'$ (Qwen2-VL): resized height and width, both multiples of $p\cdot m$. $m$: merge size (2), so $m^2=4$ patches become one position.
- $Q$: number of learned queries (32).

*Bytes per position (Module 03 formula), step by step* for the synthetic decoder below, with $L$ = layers (not a sequence length here), $H_{kv}$ = KV heads (not an image height):
$$b_{pos}=\underbrace{2}_{K,V}\times\underbrace{32}_{L}\times\underbrace{8}_{H_{kv}}\times\underbrace{128}_{d_h}\times\underbrace{2\text{ B}}_{\text{FP16}}$$
$2\times32=64$; $64\times8=512$; $512\times128=65{,}536$ elements; $\times2$ B $=131{,}072$ B $=131{,}072/1{,}024=128$ KiB. Then KV for $N$ positions is $N\times128$ KiB $=N/8$ MiB, e.g. $576/8=72.0$ MiB.

*Qwen2-VL upper bound from `max_pixels`.* Each decoder position covers $(p\cdot m)^2=28^2=784$ pixels after resizing, so $N\le$ `max_pixels`$/784$: $1{,}003{,}520/784=1{,}280$ and $12{,}845{,}056/784=16{,}384$. The 1920×1080 image needs fewer than the cap in both cases (1,222 and 2,691 below).

**Worked Example (pinned configurations; synthetic decoder):**
*Input.* One 1920×1080 image (width × height). Checkpoint processor configurations were read from the Hugging Face Hub on 2026-09-30: `llava-hf/llava-1.5-7b-hf` (revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`: crop 336, patch 14, `num_additional_image_tokens` 1, strategy `default`), `llava-hf/llava-v1.6-mistral-7b-hf` (revision `2424fdd47412fccc66d91719126b420e9fbd7065`: same, plus five grid pinpoints), and `Qwen/Qwen2-VL-7B-Instruct` (revision `eed13092ef92e448dd6875b2a00151bd3f7db0ac`: `max_pixels` 12,845,056). To compare KV, assume a *synthetic* decoder with 32 layers, 8 KV heads, $d_h=128$, FP16: $2\times32\times8\times128\times2=131{,}072$ B $=128$ KiB per position.

| Pipeline | Steps | Decoder-visible $N$ | Logical KV |
|---|---|---:|---:|
| LLaVA-1.5 | shortest edge → 336, center crop 336×336 (sides discarded); $24\times24+1-1$ | 576 | 72.0 MiB |
| LLaVA-NeXT | best grid 672×672 → base + 4 tiles; unpadded $28\times48=1{,}344$; newlines 28; base $576+1$; minus 1 | 1,948 | 243.5 MiB |
| Qwen2-VL, class default `max_pixels` $=1{,}003{,}520$ | resize 1080×1920 → 728×1316; $52\times94=4{,}888$ patches; $/4$ | 1,222 | 152.75 MiB |
| Qwen2-VL, checkpoint `max_pixels` $=12{,}845{,}056$ | resize → 1092×1932; $78\times138=10{,}764$ patches; $/4$ | 2,691 | 336.375 MiB |
| BLIP-2 (as if on the same synthetic decoder) | fixed queries | 32 | 4.0 MiB |

*Order note.* The input is stated as width × height (1920×1080), but the Qwen2-VL rows write height × width (1080×1920 → 728×1316), matching `smart_resize(height, width)`. LLaVA-NeXT grid sizes such as 336×1008 below are also height × width, as `select_best_resolution` returns them.

*Step-by-step derivations of the table rows* (Logical KV $=N\times128$ KiB $=N/8$ MiB):
1. *LLaVA-1.5.* Shortest edge 1080 → 336, so the image becomes about 597×336; the center crop keeps 336×336. Patches: $336/14=24$ per side, $24\times24=576$; $+e-1=+1-1$; $N=576$; $576/8=72.0$ MiB.
2. *LLaVA-NeXT.* Best grid 672×672 = $2\times2$ tiles of 336, so the tile feature map is $48\times48$ ($672/14=48$). The image is wider than the grid (16:9 versus 1:1), so after fitting its width into 48 columns its content is $1080\times48/1920=27$ rows tall; `_get_unpadded_features` removes $(48-27)//2=10$ padding rows top and bottom, leaving $48-20=28$ rows. $U=28\times48=1{,}344$; $R=28$ (one newline per row); base $b+e=576+1$; minus 1 for the default strategy: $1{,}344+28+577-1=1{,}948$; $1{,}948/8=243.5$ MiB.
3. *Qwen2-VL, `max_pixels` 1,003,520.* Rounding each side to a multiple of 28 gives 1092×1932 = 2,109,744 pixels, above the bound, so the image is scaled down by $\beta=\sqrt{1080\cdot1920/1{,}003{,}520}\approx1.437$: $1080/\beta\approx751.3$ and $1920/\beta\approx1{,}335.7$, which round down to multiples of 28 as $728=26\times28$ and $1{,}316=47\times28$ (rounding down reproduces the table; confirm the exact rule in `smart_resize` during the trace). Patches $728/14=52$, $1{,}316/14=94$, $52\times94=4{,}888$; $/m^2=4{,}888/4=1{,}222$; $1{,}222/8=152.75$ MiB.
4. *Qwen2-VL, `max_pixels` 12,845,056.* $1080/28\approx38.6\to39$, $1920/28\approx68.6\to69$, giving 1092×1932 = 2,109,744 pixels, inside the bound, so no rescale. Patches $78\times138=10{,}764$; $/4=2{,}691$; $2{,}691/8=336.375$ MiB.
5. *BLIP-2.* $N=Q=32$ for any input size; $32/8=4.0$ MiB.

*Interpretation and limits.* The same image spans an 84× range in decoder positions ($2{,}691/32\approx84.1$). Two rows use the same class with different `max_pixels`, so even a model name is insufficient: the processor configuration is part of the contract. LLaVA-1.5's crop discards the left and right of a wide image, a quality effect, not a cost effect. KV values are logical payloads only; block rounding and allocator overhead (Module 03) come on top. Encoder work follows a different count: LLaVA-NeXT encodes five 336×336 crops, Qwen2-VL encodes all patches before merging. A real BLIP-2 checkpoint uses its own language model, so its row only illustrates the fixed-count mechanism.

**Knowledge Check:**
1. Why does doubling the input width not change LLaVA-1.5's decoder-visible count, while it can change Qwen2-VL's?
2. In which of the four architectures does encoder work grow faster than decoder-visible length as resolution rises?
3. What error does `get_placeholder_mask` catch, and what mismatch would cause it in production?

**Guided Practice:**
At the pinned revision, call each processor's `_get_num_multimodal_tokens` (or count placeholders after `__call__`) for 336×336, 1000×1000, 400×1200, and 3840×2160 images. Compare with your hand calculation using the formulas above, and explain every difference. Then complete the source trace in Section 09.

**Feedback Contract:**
- *Expected Evidence*: Per-architecture counts from code and by hand, the processor configuration and revision used, logical KV with bytes per position stated, and trace citations by symbol. Reference values: LLaVA-NeXT 1000×1000 → 2,928; 400×1200 → 2,328 (best grid 336×1008); Qwen2-VL class default 400×1200 → 602. These reference sizes are height × width (height 400, width 1200); a 400-wide, 1200-tall portrait image instead selects grid 1008×336 and gives 2,376 for LLaVA-NeXT, while Qwen2-VL still gives 602 ($392\times1{,}204$ or $1{,}204\times392$ → $28\times86=2{,}408$ patches, $/4=602$). Check for 1000×1000: grid 672×672, no unpadding, $48\times48+48+577-1=2{,}928$. KC1: fixed crop versus pixel-bounded resize. KC2: Qwen2-VL (merge after encoding) and BLIP-2 (fixed output after a full encoder pass). KC3: processor and model disagree on feature count, for example a mismatched `vision_feature_select_strategy` or patch size.
- *Common Failure*: Using $(H/p)(W/p)$ on the raw image for every model, or reading class defaults instead of the checkpoint's processor configuration.
- *Diagnostic Hint*: What exact height and width reach the encoder, and what happens to features between the encoder and the decoder?
- *Concept to Revisit*: KV bytes per token (Module 03, Lesson 3.1); satellite `vision-language-models/03-vlm-architectures` for model-family depth.

**Learning Outcome:**
Compute decoder-visible modality positions and logical KV from a pinned processor and architecture, and trace the computation in source.

*(Effort: 50m instruction, 25m practice; source trace counted under `source_trace`)*

---

### Lesson 21.3 — Multimodal Batching, Padding, and Caching

**Engineering Question:**
When do mixed-shape multimodal batches and caches save work, and when do they waste it or return the wrong result?

**Concepts & Definitions:**
- **Multimodal shape dimensions**: images per request, tiles per image, patches per tile, audio frames, and decoder-visible positions. A dense batch pads each dimension to its maximum.
- **Padding efficiency**: for one padded dimension, useful positions over allocated positions, $\sum_i L_i/(B\max_i L_i)$. It counts positions, not measured FLOPs or time (**D**, CLM-011).
- **Fixed-length audio padding**: Whisper's 30 s contract pads short clips; the encoder still runs on 1,500 positions (**O**, CLM-008).
- **Shape bucketing**: grouping requests by modality, processed shape, decoder-visible length, and deadline. It helps only when saved work exceeds added queueing and lost batching (**H**, CLM-017). Scheduler mechanics stay in Module 04.
- **Encoder budget in a serving engine**: at the pinned vLLM revision, `Scheduler._try_schedule_encoder_inputs` schedules an encoder item only if its placeholders overlap this step's token window, it is not already cached, and the encoder budget and encoder cache can hold it. Otherwise it trims the step's decoder tokens to stop before that item (**O**, CLM-013). There `max_num_encoder_input_tokens` and `encoder_cache_size` are set from `max_num_batched_tokens`.
- **Three caches**:
  - *Processor cache*: processed media tensors. At the pinned vLLM revision it defaults to 4 GiB, LRU or shared-memory type, keyed by a framed hash of media and processor arguments; an oversize item is served uncached (**O**, CLM-014).
  - *Encoder-output cache*: projected embeddings for an item, reusable across steps or requests.
  - *Decoder prefix cache*: KV for a token prefix (Module 03, Lesson 3.6), whose key must include multimodal identifiers.
- **Cache identity**: a safe key binds media content, preprocessing parameters and code version, encoder/projector weights, model, adapters and position semantics, tenant policy, and output shape. A content hash alone proves only that the hashed bytes match (**D**, CLM-015).
- **Frontier**: ModServe reports stage-specific batching behavior on its evaluated systems (**O**, CLM-016). TCM-Serve's abstract reports modality grouping, dynamic priority, and aging to reduce multimodal head-of-line blocking (**O**, CLM-020; abstract only). VLCache's abstract reports approximate encoder/KV reuse with selective recomputation (**O**, CLM-021; abstract only).

**Mechanism Explanation:**
A mixed batch is padded per dimension, so one large item sets the shape for all. The encoder may run on padded tiles or frames unless the model removes them first; whether it does is an implementation fact to check in the trace, not to assume. In an iteration-level engine, a large encoder item can stall the decoder tokens behind it in the same request. A cache hit skips the stage it covers and nothing else: a processor-cache hit still pays encoding, and an encoder-cache hit still pays prefill unless the decoder prefix also hits.

**Quantitative Model / Derivation (D, CLM-011):**
$$\eta_{pad}=\frac{\sum_{i=1}^{B}L_i}{B\cdot\max_i L_i}$$
applied per padded dimension. For encoder budget $E$ and items with embed counts $n_1,n_2,\dots$ in placeholder order, the scheduled prefix is the longest prefix with $\sum n_j\le E$ (ignoring cache hits), and decoder tokens stop at the next item's offset.

where:
- $B$: number of items in the batch along this dimension (requests, images, or clips); $L_i$: useful length of item $i$ in that dimension (positions, crops, or frames). $L_i$ is a length, not the layer count $L$ of the KV formula.
- $B\cdot\max_i L_i$: slots allocated when every item is padded to the longest one. $\eta_{pad}=1$ means no padding; $1-\eta_{pad}$ is the padded share.
- $E$: encoder budget per scheduling step, in embeds; $n_j$: embeds produced by item $j$.

*Plain reading.* Walk the items in prompt order and keep adding them while the running total stays within $E$. The first item that does not fit is deferred, and the decoder tokens for this step stop just before its placeholders, because the decoder cannot process positions whose embeddings do not exist yet.

**Worked Example (synthetic batch; pinned token counts from Lesson 21.2):**
1. *Decoder-visible padding.* Three images with 576, 1,948, and 1,222 positions in one dense batch: $\eta=(576+1{,}948+1{,}222)/(3\times1{,}948)=3{,}746/5{,}844=0.641$. About 36% of allocated positions are padding.
2. *Tile padding.* LLaVA-NeXT images of 1920×1080, 400×1200, and 336×336 produce 5, 4, and 3 crops (base included). `_pad_for_batching` pads to 5: $\eta=12/15=0.80$. Here $B=3$, $L_i\in\{5,4,3\}$, $\sum L_i=12$, $B\max L_i=3\times5=15$.
3. *Audio padding.* A 5 s clip under the 30 s Whisper contract: the waveform becomes 480,000 samples and 3,000 mel frames; the encoder runs on 1,500 positions, of which about $5/30\times1{,}500=250$ come from real audio. $\eta\approx1/6$.
   Step by step: samples $=30\text{ s}\times16{,}000\text{ Hz}=480{,}000$; mel frames $=30\text{ s}/10\text{ ms hop}=3{,}000$; $3{,}000/1{,}500=2$ mel frames per encoder position, i.e. 20 ms of audio per position. Real audio: $5\text{ s}/10\text{ ms}=500$ frames $\to500/2=250$ positions; $\eta=250/1{,}500=1/6\approx0.17$. "About" because window edges can shift the real-frame count by one.
4. *Encoder budget.* Synthetic engine with $E=2{,}048$. One request has two images with 1,948 and 1,222 embeds, neither cached. Image 1 fits ($2{,}048-1{,}948=100$ left), image 2 does not, so this step's decoder tokens stop at image 2's placeholder offset; image 2 is encoded in a later step.
5. *Cache identity.* A team switches `max_pixels` from 1,003,520 to 12,845,056 but keys its encoder-output cache on image bytes only. A hit returns 1,222 embeds where the new contract expects 2,691: in the pinned code path this fails the placeholder check; in a system without such a check it silently returns the old representation.

*Interpretation and limits.* Position efficiency is not time efficiency: masked or packed kernels change measured cost, and the padded share of time must be measured. Example 4 follows the pinned code path, but whether it helps or hurts TTFT depends on load.

**Knowledge Check:**
1. Why can a processor-cache hit leave TTFT almost unchanged?
2. Which fields must a decoder prefix-cache key include beyond token IDs when a prompt contains an image?
3. Under what arrival pattern does shape bucketing increase P99 TTFT?

**Guided Practice:**
Take a 200-request synthetic mix (60% text, 30% single image, 10% five-image). Compute padded versus valid positions under three policies: one global batch, modality buckets, and modality-plus-length buckets. Then write the cache-key schema for processor, encoder-output, and prefix caches, and mark which field each invalidation event (new checkpoint, new processor config, new adapter, new tenant) changes.

**Feedback Contract:**
- *Expected Evidence*: Per-policy padding tables, a predicted queueing cost, and a key schema with every dependency from CLM-015. KC1: encoding and prefill still run. KC2: per-item media identity and offset (the pinned vLLM `_gen_mm_extra_hash_keys` adds `(identifier, offset)`), plus model/adapter/salt per Module 03. KC3: sparse arrivals where buckets wait to fill.
- *Common Failure*: Keying an encoder cache on file name or URL, or treating 64% position efficiency as 64% time efficiency.
- *Diagnostic Hint*: Which stage does each cache skip, and what would make two byte-identical images produce different embeddings?
- *Concept to Revisit*: Prefix-cache identity (Module 03, Lesson 3.6); continuous batching (Module 04, Lesson 4.2).

**Learning Outcome:**
Quantify padding across multimodal dimensions, design cache keys that preserve semantic equivalence, and predict when bucketing or caching changes latency.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 21.4 — Cross-Stream Timing for Audio, Video, and Text

**Engineering Question:**
How do we align audio, video, and text that arrive on different clocks, and how is a timing error told apart from a transport delay?

**Concepts & Definitions:**
- **Media clock**: the sampling clock of one stream, for example 48 kHz audio or a 90 kHz video timestamp clock. RTP timestamps record sampling instants on such a clock; different media normally have independent rates and random offsets (**O**, CLM-010).
- **Reference clock mapping**: an RTCP sender report pairs a media timestamp with a shared wall-clock timestamp, so a receiver can convert each stream to reference time (**O**, CLM-010).
- **Skew**: the difference in reference time between content presented together. **Arrival jitter**: variation in network arrival; it is not media skew.
- **Model-side time alignment**: Qwen2.5-Omni reports 16 kHz audio with 25 ms windows and 10 ms hops, audio representations at about 40 ms each, temporal position IDs tied to absolute time, audio and video interleaved in 2 s chunks, and block-wise encoders for streaming prefill (**O**, CLM-009). This is one reported architecture, not a general contract; streaming speech depth is in `speech-ai/04-streaming-speech`.

**Mechanism Explanation:**
Each stream is converted to reference time with its own clock rate and mapping. The pipeline then groups content into windows (encoder blocks, interleaving chunks) and may wait for the slowest stream to fill a window. Waiting adds latency equal to the window lookahead plus the slowest stream's lag. Emitting before a window is complete reduces latency but risks responding to partial context. Native models that encode time in positions still depend on the pipeline delivering correctly timed input.

**Quantitative Model / Derivation:**
For a stream with clock rate $f$ and sender-report pair $(rtp_{sr}, t_{sr})$:
$$t_{ref}=t_{sr}+\frac{rtp-rtp_{sr}}{f}$$
using modular 32-bit arithmetic when timestamps wrap. Skew between two items is $t_{ref,A}-t_{ref,B}$. A window of length $W$ cannot be complete before its last sample's reference time plus that stream's processing lag.

where:
- $rtp$: the item's RTP timestamp, in ticks of its own media clock; $f$: that clock's rate in ticks per second (48,000 for this audio, 90,000 for video).
- $(rtp_{sr},t_{sr})$: from the latest RTCP sender report, a media timestamp and the reference (wall-clock) time it corresponds to.
- $t_{ref}$: the item's sampling instant on the shared reference clock, in seconds. $W$ is a window duration in seconds, not an image width.

*Step by step.* $(rtp-rtp_{sr})$ counts ticks since the sender report; dividing by $f$ converts ticks to seconds; adding $t_{sr}$ places that offset on the reference clock. With wrap, compute the tick difference as $\Delta=\big((rtp-rtp_{sr}+2^{31})\bmod 2^{32}\big)-2^{31}$, so a small step across the wrap stays small instead of becoming a jump of about $-2^{32}$ ticks. Wrap period $=2^{32}/f$: about 47,722 s (13.3 h) at 90 kHz and about 89,478 s (24.9 h) at 48 kHz.

**Worked Example (synthetic timestamps):**
*Input.* Audio clock 48 kHz with sender report (RTP 1,000,000 ↔ 100.000 s). Video clock 90 kHz with sender report (RTP 5,000,000 ↔ 100.020 s). An audio packet carries RTP 1,048,000 and arrives at 101.050 s; a video frame carries RTP 5,088,200 and arrives at 101.180 s.
*Step 1.* Audio: $100.000+48{,}000/48{,}000=101.000$ s (tick difference $1{,}048{,}000-1{,}000{,}000=48{,}000$, i.e. 1 s at 48 kHz).
*Step 2.* Video: $100.020+88{,}200/90{,}000=100.020+0.980=101.000$ s (tick difference $5{,}088{,}200-5{,}000{,}000=88{,}200$).
*Result.* Media skew is 0 ms. The 130 ms arrival difference is transport and buffering, not misalignment.
*Step 3.* Under a 2 s interleaving window like Qwen2.5-Omni's, the window $[100,102)$ s cannot close before the video frame at 101.180 s arrival plus later frames up to 102 s, so first output for that window waits on the slowest stream.
*Interpretation and limits.* Comparing raw RTP values (1,048,000 versus 5,088,200) or arrival times would have suggested a large lag. The sender-report mapping assumes both senders' reference clocks are synchronized; if they are not, the mapping carries their offset.

**Knowledge Check:**
1. Why is "audio arrived 130 ms before video" not evidence of lip-sync error?
2. After roughly how long does a 90 kHz 32-bit RTP timestamp wrap, and what breaks if the receiver ignores it?
3. What does a block-wise audio encoder trade compared with full attention over the whole clip?

**Guided Practice:**
An audio stream at 16 kHz has sender report (RTP 0 ↔ 50.000 s) and a video stream at 90 kHz has (RTP 0 ↔ 50.010 s). Audio RTP 32,000 and video RTP 180,900 are presented together. Compute the skew, state its sign convention, and decide whether a 2 s window starting at 50.000 s contains both.

**Feedback Contract:**
- *Expected Evidence*: Audio 52.000 s and video $50.010+2.010=52.020$ s, so video lags by 20 ms under "video minus audio"; neither falls inside the half-open window $[50,52)$ s: audio sits exactly on its end boundary and video 20 ms past it, so both belong to the next window. State the window convention, because a closed window would admit the audio item. KC2: $2^{32}/90{,}000\approx47{,}722$ s $\approx13.3$ h; unwrapped arithmetic produces a huge negative jump. KC3: lower latency and bounded memory in exchange for attention limited to blocks.
- *Common Failure*: Comparing timestamps from different media directly, or treating arrival time as sampling time.
- *Diagnostic Hint*: Which clock produced each number, and what maps it to the shared reference?
- *Concept to Revisit*: RTP timestamp semantics (RFC 3550 §5.1, §6.4.1); critical path (Lesson 21.1).

**Learning Outcome:**
Convert multi-stream media timestamps to a shared clock, separate skew from arrival jitter, and compute the latency cost of window alignment.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 21.5 — Memory and Latency Boundaries: Competing Diagnoses

**Engineering Question:**
When a multimodal workload OOMs or misses its TTFT target, which measurements distinguish encoder work, padding, capacity, fragmentation, and host preprocessing?

**Concepts & Definitions:**
- **Competing hypotheses** for multimodal OOM or TTFT regressions (**H**, CLM-012):
  1. *Encoder work*: more processed patches or tiles lengthen encoder spans and compete with prefill/decode.
  2. *Padding*: mixed shapes inflate padded positions in encoder or decoder batches.
  3. *Capacity*: decoder-visible length times bytes per position (CLM-007) exceeds configured KV or encoder-cache capacity; the engine refuses allocation or trims work.
  4. *Engine allocation*: block-level waste, such as tail slack per sequence, reduces usable capacity (Module 03, Lesson 3.2).
  5. *Framework-allocator fragmentation*: the caching allocator holds reserved memory in segments that cannot serve a large request.
  6. *Host preprocessing*: CPU decode/resize/tile time leaves the device idle while requests wait.
- **Allocator evidence**: PyTorch reports `memory_allocated()` for live tensors and `memory_reserved()` for memory its caching allocator manages; cached free memory appears used to device tools. Snapshots and `memory_stats()` show segments and inactive split blocks, which is what a fragmentation claim needs; the reserved-minus-allocated gap alone does not establish fragmentation (**O**, CLM-022).
- **Necessary but insufficient signals**: high device memory occupancy, large images, high SM activity, and an OOM message each fit several hypotheses.

**Mechanism Explanation:**
Each hypothesis predicts a different response to one controlled change:
| Intervention | Encoder work | Padding | Capacity | Engine allocation | Allocator fragmentation | Host |
|---|---|---|---|---|---|---|
| Halve processed patches, same decoder length | encoder span falls | little change | little change | little change | little change | resize time may fall |
| Same total tokens, uniform shapes | little change | padded positions fall | little change | little change | may change | little change |
| Raise KV/encoder-cache capacity | little change | little change | refusals stop | waste share unchanged | may worsen | little change |
| Fresh process, same load | none | none | none | none | OOM disappears or occurs later | none |
| Add CPU workers / move preprocessing | none | none | none | none | none | device idle gaps shrink |

The diagnosis ranks hypotheses by which predicted signals covary with the symptom. Interactions are expected: for example, capacity pressure and allocator fragmentation can co-occur.

**Quantitative Model / Derivation:**
Logical decoder KV demand for a request: $KV=\sum_{items}N_{item}\cdot b_{pos}+N_{text}\cdot b_{pos}$ (**D**, CLM-007). The engine's physical demand adds rounding and metadata per Module 03. The framework reports $M_{reserved}-M_{allocated}$ as cached memory; fragmentation is supported only if snapshots show free memory split into pieces smaller than the failing request.

where:
- $N_{item}$: decoder-visible positions of one modality item (Lesson 21.2); $N_{text}$: text token positions in the same request.
- $b_{pos}$: logical KV bytes per decoder position (128 KiB for the synthetic decoder). Factoring out gives $KV=(\sum N_{item}+N_{text})\,b_{pos}$.
- $M_{allocated}$: bytes held by live tensors (`memory_allocated()`); $M_{reserved}$: bytes the caching allocator holds from the device (`memory_reserved()`). The gap is free-but-cached memory; its total size says nothing about whether it is in pieces large enough for the next request.

**Worked Example (synthetic telemetry; pinned token counts):**
*Input.* A document-QA service with the synthetic 128 KiB/position decoder sends 8 page images per request. After a configuration change, OOMs and a TTFT rise appear. Observations:
- Processor config diff: Qwen2-VL `max_pixels` moved from 1,003,520 to 12,845,056.
- Per-image decoder positions for 1920×1080 pages: 1,222 before, 2,691 after (Lesson 21.2).
- Engine logs show allocation refusals at peak; preemptions rose.
- `memory_snapshot()` at failure: reserved minus allocated ≈ 0.4 GiB; the largest free block is larger than the failing request.
- Host spans unchanged; GPU idle gaps unchanged.
- Encoder spans rose roughly in proportion to patches.

*Step 1: capacity.* Logical image KV per request: $8\times1{,}222\times128$ KiB $\approx1.19$ GiB before; $8\times2{,}691\times128$ KiB $\approx2.63$ GiB after, a 2.2× increase. Refusals at peak fit this.
Arithmetic: $8\times1{,}222=9{,}776$ positions $\times128$ KiB $=1{,}251{,}328$ KiB $=1{,}222$ MiB $\approx1.19$ GiB; $8\times2{,}691=21{,}528$ positions $\to2{,}691$ MiB $\approx2.63$ GiB; ratio $2{,}691/1{,}222\approx2.20$. (With 8 images and 128 KiB per position, KV in MiB equals the per-image position count.) Text positions are excluded here; they add $N_{text}\cdot b_{pos}$ to both.
*Step 2: fragmentation.* Free memory is not split below the failing size, so allocator fragmentation is not supported by this evidence.
*Step 3: host.* Unchanged host spans and idle gaps rule out preprocessing for this regression.
*Step 4: encoder work.* Encoder spans grew with patches, so encoder work explains part of the TTFT rise but not the OOM.
*Diagnosis.* Capacity pressure from a larger decoder-visible length (ranked first for OOM) plus encoder work (contributing to TTFT). Padding is untested: check padded/valid ratios before excluding it.
*Interpretation and limits.* All telemetry here is synthetic. In a real incident, the first remeasurement is the same load with the old `max_pixels`.

**Knowledge Check:**
1. Why does high `nvidia-smi` memory use not show fragmentation in a PyTorch process?
2. Which single experiment separates capacity from allocator fragmentation?
3. Give a case where host preprocessing and encoder work both rise after one change.

**Guided Practice:**
For the outline's original claim, "visual tokens fragment KV memory," write one prediction per hypothesis in the table, the metric that would confirm it, and the observation that would falsify it. Then state which three measurements you would collect first and why.

**Feedback Contract:**
- *Expected Evidence*: Six predictions with signals and falsifiers; a prioritized measurement plan beginning with processor config diff, decoder-visible counts, engine refusal counters, and an allocator snapshot. KC1: caching allocator reserves memory that device tools show as used (CLM-022). KC2: restart or snapshot comparison at the same load and capacity. KC3: raising resolution increases both resize time and patches.
- *Common Failure*: Accepting "fragmentation" because occupancy was high and images were large.
- *Diagnostic Hint*: What would you expect to see if your preferred hypothesis were false?
- *Concept to Revisit*: Fragmentation hierarchy (Module 03, Lesson 3.2); allocated versus reserved (Module 19, Lesson 19.2).

**Learning Outcome:**
Rank competing explanations for a multimodal OOM or latency regression using discriminating measurements, without presupposing fragmentation.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 21.6 — The Quality–Latency–Memory Boundary

**Engineering Question:**
How far can representation size be reduced before required quality is lost, and how is that boundary established rather than assumed?

**Concepts & Definitions:**
- **Representation controls**: resolution and pixel bounds, tile grid, frame sampling rate, audio window, learned-query count, merge factor, and token pruning.
- **Quality boundary**: the set of control settings whose paired quality stays within declared margins on every required slice. Compression lowers some latency and memory terms but may fail slices that need small, temporal, or late-used evidence (**H**, CLM-018).
- **Fixed-query bridges**: BLIP-2's fixed 32 queries cap decoder-visible length regardless of resolution (**O**, CLM-006); the cost is capacity for detail, which must be measured.
- **Frontier evidence**: the 2026 DSTP paper reports that static visual-token pruning can fail on complex reasoning because relevant visual information shifts during decoding (**O**, CLM-019; abstract only). VLCache reports approximate reuse with accumulated error controlled by recomputation (**O**, CLM-021; abstract only). Neither result is reproduced here.

**Mechanism Explanation:**
Reducing positions shortens encoder and prefill work and logical KV, but the effect on accuracy is task-dependent. A pruning policy chosen on aggregate accuracy can pass while a slice such as small-print OCR fails. The boundary is therefore drawn per slice with paired comparisons against the uncompressed baseline (Module 15).

**Quantitative Model / Trade-off Comparison:**
For each setting $k$: decoder positions $N_k$, logical KV $N_k b_{pos}$, measured stage latencies, and paired quality deltas $\Delta_{k,s}$ per slice $s$. A setting is admissible if the lower confidence bound of $\Delta_{k,s}\ge -m_s$ for all required $s$. Among admissible settings, choose the cheapest by the declared cost metric.

where:
- $k$: one control setting (here a `max_pixels` value); $s$: one required evaluation slice.
- $\Delta_{k,s}$: paired quality difference on slice $s$, setting $k$ minus baseline, in points (negative means worse).
- $m_s$: the non-inferiority margin for slice $s$, declared before testing. $m_s$ is a margin, unrelated to the merge size $m$ of Lesson 21.2.

*Plain reading.* A setting passes only if, on every required slice, even the pessimistic end of its confidence interval loses no more than the declared margin. A single failing slice rejects the setting, however good the aggregate looks.

**Worked Example (pinned token counts; synthetic quality):**
*Input.* Qwen2-VL on 1920×1080 pages. Settings by `max_pixels`: 200,704 → 252 positions; 401,408 → 480; 1,003,520 → 1,222; 12,845,056 → 2,691 (baseline). Margins declared before testing: 2 points on each of three slices. Synthetic paired results (lower 95% bound of delta versus baseline, points):

| `max_pixels` | $N$ | Logical KV | General QA | Small-print OCR | Chart reading | Admissible |
|---:|---:|---:|---:|---:|---:|---|
| 200,704 | 252 | 31.5 MiB | −1.1 | −14.0 | −6.2 | no |
| 401,408 | 480 | 60.0 MiB | −0.8 | −5.5 | −2.4 | no |
| 1,003,520 | 1,222 | 152.75 MiB | −0.4 | −1.6 | −0.9 | yes |
| 12,845,056 | 2,691 | 336.375 MiB | 0 | 0 | 0 | baseline |

*Admissibility check* ($-m_s=-2$ on every slice): 200,704 fails OCR ($-14.0<-2$) and charts ($-6.2<-2$); 401,408 fails OCR ($-5.5$) and charts ($-2.4$); 1,003,520 passes all three ($-0.4,-1.6,-0.9\ge-2$). Logical KV column $=N\times128$ KiB $=N/8$ MiB, e.g. $252/8=31.5$, $480/8=60.0$.
*Result.* The cheapest admissible setting is 1,003,520 pixels: 1,222 positions, a 54.6% reduction in logical KV from the baseline ($1-1{,}222/2{,}691$).
*Interpretation and limits.* The general-QA column alone would have admitted the 252-position setting. The decision holds only for these slices, images, and margins. Passing a bounded suite does not prove no other task degrades.

**Knowledge Check:**
1. Why can aggregate accuracy approve a setting that a slice rejects?
2. What does BLIP-2's fixed query count buy and cost?
3. Why must a pruning method be evaluated on tasks where needed evidence changes during decoding?

**Guided Practice:**
Pick one control (frame rate for video or pixel bound for images). Define three required slices, margins, and suite sizes before running. Sweep four settings, compute paired deltas with intervals, and report the admissible set and the cheapest member.

**Feedback Contract:**
- *Expected Evidence*: Pre-registered slices and margins, per-setting positions and logical KV, paired intervals, and an admissibility table. KC1: slices with small shares can fail without moving the aggregate. KC2: bounded decoder length at the cost of detail capacity. KC3: DSTP-type information shift (CLM-019).
- *Common Failure*: Choosing a compression rate from a paper's aggregate result, or declaring margins after seeing results.
- *Diagnostic Hint*: Which slice needs the detail your control removes?
- *Concept to Revisit*: Paired evaluation and margins (Module 15, Lesson 15.4); satellite `vision-language-models/06-vlm-evaluation`.

**Learning Outcome:**
Select representation budgets inside a measured, slice-level quality–latency–memory boundary.

*(Effort: 40m instruction, 15m practice)*

---

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020 (arXiv v2). *Read*: §3.1. *Scope*: patch embedding (CLM-002); not decoder-visible counts.
- [BLIP-2](https://arxiv.org/abs/2301.12597) — Li et al., 2023 (arXiv v3). *Read*: §3.1. *Scope*: fixed learned-query bridge (CLM-006).
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Radford et al., 2022. *Read*: §2.2. *Scope*: 30 s log-mel contract (CLM-008).
- [RFC 3550: RTP](https://www.rfc-editor.org/rfc/rfc3550) — Schulzrinne et al., 2003. *Read*: §5.1, §6.4.1. *Scope*: timestamp and sender-report mapping (CLM-010).

**CURRENT DEFAULT** (scoped; one library's default is not an industry default)
- Transformers processors and models for LLaVA, LLaVA-NeXT, Qwen2-VL, BLIP-2, and Whisper at the pinned revision (CLM-003, CLM-004, CLM-005, CLM-008): the defaults of that snapshot and the cited checkpoint configurations.
- vLLM V1 encoder scheduling and multimodal processor cache at the pinned revision (CLM-013, CLM-014), and the [vLLM Optimization and Tuning guide](https://docs.vllm.ai/en/stable/configuration/optimization/) (multimodal caching section), as one runtime's behavior.
- [PyTorch CUDA semantics: Memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management), version 2.14 (CLM-022).
- Recommended engineering baselines from this module: stage ledgers (CLM-001), architecture-scoped token accounting (CLM-007), and cache identity contracts (CLM-015).

**WORKLOAD-DEPENDENT**
- AnyRes tiling (CLM-004; [LLaVA-OneVision](https://arxiv.org/abs/2408.03326), 2024, v3, §3.2), dynamic resolution with merge ([Qwen2-VL](https://arxiv.org/abs/2409.12191), 2024, v2, §2.1; CLM-005), shape bucketing (CLM-017), and representation compression (CLM-018).

**FRONTIER**
- [ModServe](https://arxiv.org/abs/2502.00937) (2025, v3; §§2–3 read) — stage-aware disaggregation (CLM-016). Placement detail is Module 20.
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) (2025; §§2.2, 2.4 read) — time-aligned positions and block-wise streaming encoders (CLM-009).
- [TCM-Serve](https://arxiv.org/abs/2603.26498v2) (Papaioannou and Doudali, 2026, v2; v3 retitled *Rocks, Pebbles and Sand* (RPS-Serve); abstract only) — modality-aware scheduling (CLM-020).
- [VLCache](https://arxiv.org/abs/2512.12977) (Qin et al., 2025; abstract only) — approximate encoder/KV reuse (CLM-021).
- [Why and When Visual Token Pruning Fails?](https://arxiv.org/abs/2604.12358) (Kim et al., 2026; abstract only) — information shift during decoding (CLM-019).
- [FastVLM](https://arxiv.org/abs/2412.13303) (2024, v2; introduction read) — hybrid encoder reducing tokens and encoding latency; discovery only, no claim.

**LEGACY / INSUFFICIENT**
- Deriving visual tokens or KV from image resolution for all models.
- Diagnosing "KV fragmentation" from occupancy or image size.
- Keying caches on file names or URLs.
- Comparing media timestamps from different clocks directly.
- Choosing compression rates from aggregate accuracy.

**PRODUCTION SOURCE TRACE**
- Repository: `huggingface/transformers`. Revision: `d6c1e71bd717bf092f8293f0c3c9bd4a5ac5401a` (committed 2026-09-30). Verified 2026-09-30, static inspection only.
  - `src/transformers/models/llava/processing_llava.py::{LlavaProcessor.replace_image_token, LlavaProcessor._get_num_multimodal_tokens}`
  - `src/transformers/models/llava/modeling_llava.py::{LlavaMultiModalProjector.forward, LlavaModel.get_image_features, LlavaModel.get_placeholder_mask, LlavaModel.forward}`
  - `src/transformers/models/llava_next/processing_llava_next.py::{LlavaNextProcessor._get_number_of_features, LlavaNextProcessor._get_unpadded_features}`; `image_processing_llava_next.py::{LlavaNextImageProcessor._get_image_patches, LlavaNextImageProcessor._pad_for_batching}`; `src/transformers/image_processing_utils.py::select_best_resolution`
  - `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py::{smart_resize, Qwen2VLImageProcessor.patchify}`; `processing_qwen2_vl.py::Qwen2VLProcessor.replace_image_token`; `modeling_qwen2_vl.py::{PatchEmbed.forward, PatchMerger.forward}`
  - `src/transformers/models/whisper/feature_extraction_whisper.py::WhisperFeatureExtractor`; `modeling_whisper.py::WhisperEncoder.forward`
- Repository: `vllm-project/vllm`. Revision: `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad` (committed 2026-09-25; the same revision Module 04 pins). Verified 2026-09-30, static inspection only.
  - `vllm/v1/core/sched/scheduler.py::Scheduler._try_schedule_encoder_inputs`; `vllm/config/scheduler.py::SchedulerConfig.__post_init__`; `vllm/config/multimodal.py::MultiModalConfig`; `vllm/multimodal/hasher.py::MultiModalHasher.hash_kwargs`; `vllm/v1/core/kv_cache_utils.py::_gen_mm_extra_hash_keys`

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY` and record: Transformers and (if used) vLLM revisions, checkpoint processor configuration and its Hub revision, media fixtures with hashes, hardware, and whether each number is executed, static, or synthetic.

### LAB A — Stage Ledger and Token Accountant
- **Objective**: Build a stage ledger for image and audio requests and a token accountant that reproduces pinned processor counts for LLaVA, LLaVA-NeXT, Qwen2-VL (two `max_pixels`), BLIP-2, and Whisper.
- **Pre-Registered Hypothesis**: Hand formulas from Lesson 21.2 match processor placeholder counts exactly for 20 image sizes; the Lesson 21.1 critical path predicts measured first-output time within a declared tolerance.
- **Independent Variables**: Architecture, processor configuration, image size and aspect ratio, number of images, audio length.
- **Dependent Variables**: Decoder-visible positions, encoder input size (crops, patches, frames), logical KV, stage spans, critical path, sum of spans.
- **Break & Falsify**: Feed a 199:1 and a 201:1 aspect-ratio image (Qwen2-VL's `smart_resize` rejects ratios above 200), a 1-pixel-high image, and a 31 s clip. Any count mismatch falsifies the hand model until explained in source.
- **Alignment**: Lessons 21.1–21.2.
- **Effort Estimate**: 3h total.

### LAB B — Padding, Buckets, Encoder Budget, and Cache Identity
- **Objective**: Measure padded versus valid positions across shape dimensions, compare batching policies, trace the pinned vLLM encoder-budget path, and break a cache key.
- **Pre-Registered Hypothesis**: CLM-017, with predeclared arrival regimes, a minimum padding reduction, and a maximum acceptable throughput loss. Also: an encoder-output cache keyed on bytes only returns a stale representation after a processor change.
- **Independent Variables**: Batching policy, request mix, arrival rate, encoder budget, cache key schema, processor configuration.
- **Dependent Variables**: $\eta_{pad}$ per dimension, measured stage time, queue residence, TTFT P50/P95/P99, throughput, cache hit rate, stale-hit count.
- **Break & Falsify**: Use sparse arrivals where buckets wait; use one 10-image request in a text-heavy mix; change `max_pixels` without changing the key. Bucketing is falsified for a regime if it does not improve the declared utility frontier. Confirm by test (or, if vLLM cannot be run, by trace) whether padded LLaVA-NeXT tiles reach the vision tower.
- **Alignment**: Lesson 21.3; the source trace step is counted under `source_trace`.
- **Effort Estimate**: 3.5h total (source trace counted separately, 2h).

### LAB C — Stream Alignment and the Quality Boundary
- **Objective**: Align synthetic audio and video streams with independent clocks, measure window-induced latency, and map an admissible representation boundary for one image or video control.
- **Pre-Registered Hypothesis**: Reference-time mapping removes apparent skew created by arrival jitter; at least one compression setting that passes aggregate quality fails a required slice (CLM-018).
- **Independent Variables**: Clock offsets and rates, jitter, wraparound, window length; pixel bound or frame rate; pruning on or off.
- **Dependent Variables**: Computed skew, window completion time, decoder positions, stage latency, paired quality per slice with intervals.
- **Break & Falsify**: Inject a 32-bit RTP wrap, a sender-report offset of 40 ms, and a mid-stream clock-rate error. For compression, include a slice where needed evidence appears only late in the answer (CLM-019). The quality hypothesis is falsified if a compressed setting stays within margins on every slice and repeated run.
- **Alignment**: Lessons 21.4 and 21.6.
- **Effort Estimate**: 3h total.

### LAB D — Discriminating Diagnosis of a Multimodal OOM
- **Objective**: Reproduce a multimodal OOM or TTFT regression on a small VLM (or a simulator with the pinned token counts) and run the six-hypothesis protocol of Lesson 21.5.
- **Pre-Registered Hypothesis**: CLM-012: each hypothesis's predicted signal covaries with its intervention and not with others'.
- **Independent Variables**: Pixel bound, images per request, shape mix, KV/encoder-cache capacity, process restarts, CPU preprocessing workers, allocator configuration.
- **Dependent Variables**: Encoder spans, padded/valid positions, engine refusals or preemptions, block-level waste, `memory_allocated`, `memory_reserved`, snapshot segment sizes, host spans, device idle gaps, TTFT.
- **Break & Falsify**: Construct one case for each hypothesis that produces a similar top-level symptom (OOM or TTFT rise), then show that the discriminating table separates them. A hypothesis whose prediction does not covary with its intervention is rejected for that case.
- **Alignment**: Lesson 21.5 and Incident 21.1.
- **Effort Estimate**: 3.5h total.

---

## 07 Break / Incident Scenarios

### Incident 21.1 — "The Visual Tokens Fragmented Our KV Cache"

- **Incident Symptoms**:
  - A document assistant upgraded its model checkpoint and serving image on the same day.
  - P95 TTFT rose from 1.4 s to 3.9 s; OOM restarts appeared during the daily peak of scanned-contract uploads.
  - `nvidia-smi` shows 97% memory used on all GPUs; an engineer posts "visual tokens fragment the KV cache" and proposes lowering block size.
  - Text-only requests also slowed at peak.
  - CPU utilization on API servers rose from 35% to 80%.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: encoder work from a larger processed image; padding from mixed page counts; capacity pressure from more decoder-visible positions; engine block waste; framework-allocator fragmentation; host preprocessing saturation; a processor/checkpoint mismatch; a cache invalidated by the upgrade (cold processor/encoder caches).
  2. *Rank Initial Plausibility*: Use the change list and timing, but assign no cause without a test. Note that the 97% figure fits every hypothesis, and that CPU growth and text-request slowdown point at shared resources.
  3. *Identify Missing Evidence*: processor configuration diff and decoder-visible counts per image before and after; per-stage ledger; padded/valid ratios; engine allocation refusals and preemptions; allocator snapshot at failure; API-server CPU profile and queueing; cache hit rates before and after; checkpoint and serving revisions.
  4. *Design Discriminating Tests*: replay a captured peak with (a) old and new pixel bounds, (b) shapes bucketed versus mixed, (c) increased capacity, (d) a fresh process at the failure load, (e) doubled API-server workers, (f) warmed versus cold caches. State the predicted outcome of each test under each hypothesis before running it.
  5. *Execute Causal Diagnosis*: Rank causes from test results; report interactions (for example, larger images raise both host preprocessing and capacity pressure) and anything unexplained.
  6. *Prescribe Mitigation and Prevention*: Only for supported causes. Candidates: restore or tune the pixel bound inside a measured quality boundary; scale API-server preprocessing; bucket by shape; size capacity from decoder-visible counts; include processor configuration in cache keys and warm caches before peak. Reject the block-size change unless engine-waste evidence supports it.
  7. *Remeasure*: Same peak replay: TTFT distribution, OOM count, refusals, encoder and host spans, quality on required slices, and cache hit rates, compared with the pre-upgrade baseline.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Insurance Claims Intake with Photos, Calls, and Video

A claims platform accepts, per claim, up to 6 damage photos, an optional voice note, and an optional 60 s inspection video. It must return a structured damage summary with a first output under 2.5 s at P95 and must not regress on small-print reading (policy numbers in photos) or on temporal questions in videos.

**Workload Fixture (SYNTHETIC — exercise assumptions, not measurements):**

| Input | Fixture value |
|---|---|
| Model | Qwen2-VL-style pipeline at the pinned Transformers revision; patch 14, merge 2; candidate `max_pixels` in {401,408; 1,003,520; 12,845,056} |
| Decoder | 32 layers, 8 KV heads, $d_h=128$, FP16 → 128 KiB per decoder position |
| Photos | 1920×1080 or 4032×3024; mix 70/30 |
| Voice note | 16 kHz, 5–40 s; a Whisper-style 30 s frontend is used for transcription |
| Video | 2 fps sampling before processor; temporal patch 2 |
| Arrivals | 3 claims/s mean, bursts of 9 claims/s for 60 s |
| Measured stage costs at 1,003,520 | processor 35 ms/photo (one CPU worker), encoder 70 ms/photo, prefill 0.11 ms/decoder position, decode first token 25 ms |
| KV budget | 40 GiB logical per replica after weights and non-KV memory |
| Unknown by design | Video encoder cost per frame, quality deltas, tail distributions; measure or carry as symbols |

**Required Deliverables**:
1. A stage ledger and critical-path model for a 6-photo claim with and without a second CPU worker, with the latency budget it implies.
2. Decoder-visible positions and logical KV per photo for each candidate `max_pixels` and both photo sizes, computed from the pinned processor code path, plus the voice-note encoder input under the 30 s contract.
3. A batching and caching design: padded/valid analysis, bucket rules, and cache keys for processor, encoder-output, and prefix caches with every dependency.
4. A stream-alignment plan for voice note plus video: clocks, mapping, windowing, and its latency cost.
5. A representation-budget decision with pre-registered slices, margins, and suite sizes, and the admissibility rule.
6. A capacity check against the 40 GiB budget under the burst, using resident concurrency from Module 04 Lesson 4.4, with conditions stated.
7. A diagnosis of Incident 21.1 following all seven steps.
8. The pinned production source trace (Section 09).
9. Falsification and rollback: which observations would invalidate the design and which metrics trigger rollback.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Trace, at the pinned revisions:
1. Image preprocessing to placeholder count for LLaVA, LLaVA-NeXT, and Qwen2-VL (`replace_image_token`, `_get_number_of_features`, `smart_resize`).
2. Feature extraction, CLS selection, projection or merge, and the placeholder-count check (`get_image_features`, `LlavaMultiModalProjector.forward`, `PatchMerger.forward`, `get_placeholder_mask`).
3. Whisper's pad-to-30 s and fixed-length check (`WhisperFeatureExtractor`, `WhisperEncoder.forward`).
4. vLLM encoder scheduling and cache-key construction (`Scheduler._try_schedule_encoder_inputs`, `MultiModalHasher.hash_kwargs`, `_gen_mm_extra_hash_keys`).
5. State what was statically inspected and what you executed; record any behavior you could not verify as `TODO_VERIFY`.

### Reference Checks for Deliverables 1–2 (fixture inputs only)
- 1920×1080 photo at 1,003,520 → 1,222 positions, 152.75 MiB; at 401,408 → 480, 60.0 MiB; at 12,845,056 → 2,691, 336.375 MiB.
- 6 photos at 1,003,520: 7,332 positions; prefill $\approx7{,}332\times0.11=806.5$ ms (fixture rate, text positions excluded).
- One CPU worker, serialized pipeline per photo: processor 35 ms and encoder 70 ms per photo; with overlap the media path ends at $35+6\times70=455$ ms, versus $6\times(35+70)=630$ ms if summed. A second worker does not shorten it: encoding is the bottleneck.
  (Why: photo $i$ finishes preprocessing at $35i$ ms, but the encoder frees up only every 70 ms, so after the first 35 ms the encoder is never idle: $35+6\times70$. Positions: $6\times1{,}222=7{,}332$.)
- Voice note: always 1,500 encoder positions under the 30 s contract; notes longer than 30 s need chunking.

### Rubric Dimensions
- **Token and Memory Accounting** (Deliverables 2, 6):
  - *Insufficient*: derives tokens or KV from raw resolution, or uses class defaults without the checkpoint configuration.
  - *Competent*: computes counts from the pinned code path and logical KV with stated exclusions.
  - *Strong*: also separates encoder work from decoder-visible length and gives sensitivity to the pixel bound and photo mix.
- **Latency Reasoning** (Deliverable 1):
  - *Insufficient*: sums spans across concurrent stages.
  - *Competent*: builds a critical path with declared boundaries.
  - *Strong*: identifies the bottleneck stage and predicts which intervention moves TTFT, with a measurement plan.
- **Batching and Caching** (Deliverable 3):
  - *Insufficient*: one global batch; cache keyed on content or URL only.
  - *Competent*: per-dimension padding analysis; keys bind media, processor, and model.
  - *Strong*: predicts when bucketing fails and covers invalidation, tenant policy, and output-shape checks.
- **Stream Timing** (Deliverable 4):
  - *Insufficient*: compares arrival times or raw timestamps.
  - *Competent*: maps each stream to a reference clock and computes window latency.
  - *Strong*: handles wrap, sender offset, and partial-window policy with explicit latency/quality trade-offs.
- **Quality Boundary** (Deliverable 5):
  - *Insufficient*: chooses compression from aggregate or published results.
  - *Competent*: pre-registers slices and margins with paired intervals.
  - *Strong*: sizes suites to resolve the margins and states what the boundary cannot certify.
- **Failure Diagnosis** (Deliverable 7):
  - *Insufficient*: accepts the fragmentation explanation or another single cause from symptoms.
  - *Competent*: forms the six hypotheses and discriminating tests.
  - *Strong*: predicts each test's outcome per hypothesis, handles interactions, and remeasures against baseline.
- **Source Reasoning** (Deliverable 8):
  - *Insufficient*: cites file names without revision or symbols.
  - *Competent*: traces each required path at the pinned revision.
  - *Strong*: connects code behavior (placeholder checks, encoder-budget trimming) to observed system behavior.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Multimodal stage ledger and critical path | 21.1 | LAB A | Deliverable 1; Incident step 3 | Ledger, timeline, sum vs critical path |
| Architecture-scoped token and KV accounting | 21.2 | LAB A | Deliverables 2, 6 | Counts from code and by hand with revision and config |
| Production source trace | 21.2 | LAB B (trace step) | Deliverable 8; Section 09 items 1–5 | Pinned trace artifact |
| Padding, bucketing, encoder budget | 21.3 | LAB B | Deliverable 3; Incident step 4 | Padding tables, bucket experiment, frontier plot |
| Cache identity | 21.3 | LAB B | Deliverable 3; Incident step 6 | Key schema; stale-hit test |
| Cross-stream timing | 21.4 | LAB C | Deliverable 4 | Skew computations; window-latency measurements |
| Competing-hypothesis diagnosis | 21.5 | LAB D | Deliverable 7; Incident steps 1–7 | Discrimination table with predictions and results |
| Quality–latency–memory boundary | 21.6 | LAB C | Deliverable 5; Deliverable 9 | Admissibility table with paired intervals |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 21 must be able to:
1. Instrument a multimodal request and attribute first-output time to critical-path stages.
2. Compute decoder-visible modality positions and logical KV from a pinned processor and architecture, and trace that path in source.
3. Quantify padding across multimodal dimensions and design batching and cache keys whose reuse is semantically safe.
4. Align audio, video, and text on a reference clock and compute the latency of window alignment.
5. Discriminate encoder work, padding, capacity, engine allocation, allocator fragmentation, and host preprocessing with measurements.
6. Choose representation budgets inside a pre-registered, slice-level quality boundary.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: Resolution is an input, not a cost. Cost follows from the pinned processor and architecture: which pixels reach the encoder, how many features survive the bridge, and whether they enter decoder self-attention.
- **The Path**: media bytes → modality contract → encoder → bridge → decoder-visible length → KV and prefill → batching and caches → stream alignment → output. Each arrow has its own counter and its own failure modes.
- When a multimodal system regresses, write the stage ledger and token counts first, list competing hypotheses, and run the experiment that separates them. "Fragmentation" is one hypothesis, not a default.
- Next: Module 22 prices this work; Module 23 turns the ledger into telemetry; the satellite tracks go deeper on each model family.

### Cost & Telemetry Hook
- [Lesson 22.1](../22-ai-economics/README.md): price the stage ledger per offered multimodal task; failed preprocessing and timed-out encodes stay in the denominator.
- [Lesson 22.4](../22-ai-economics/README.md): value encoder and processor caches from measured hit rate on content-hash keys, net of storage and invalidation cost.
- [Lesson 23.1](../23-observability-reliability/README.md): emit one span per stage (preprocess, encode, bridge, prefill, decode) with decoder-visible token counts as attributes, not media size alone.
- [Lesson 23.5](../23-observability-reliability/README.md): monitor representation-budget changes with a calibrated proxy per slice; delayed labels gate any compression rollout.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
