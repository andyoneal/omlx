# Gemma 4 ANE/GPU Prefill (Experimental)

This source-build experiment extends the Qwen ANE prompt-processing runtime
to dense Gemma 4. For prompt chunks that exactly match the configured fixed
shape, each layer's fused `gate_proj`/`up_proj` pair runs as a hybrid split:
an INT8 channel prefix on the ANE and the quantized affine GPU suffix on
Metal. The merge kernel applies GeGLU while combining the two halves, so the
full gate/up result is never materialized.

Accelerated per layer:

- The `gate_proj`/`up_proj` pair, at the configured fraction of channels.

The `down_proj`, the attention core, the embedding and LM head, decode and
speculative verification keep the existing GPU path. The bank ladder,
fixed-shape eligibility, memory admission, split-bank recovery and teardown
are shared with the Qwen implementation; the activation and the dispatch
targets are the only differences.

`down_proj` is not accelerated, and the reason is cost rather than legality.
Its exact triple — contraction 21,504, output 5,376, int4 with a
per-output-channel scale — compiles at chunk widths 8, 72, 96, 128 and 160,
and fails at 16, 24, 32, 40, 45, 48, 56, 64 and 80. Legality is not monotone
in the width, and no product or alignment rule predicts it. The contraction
lowers onto the W axis, whose declared cap is 16,384, but that cap does not
act as a plain limit: a 20,480 contraction also compiles at several widths.
The compiler tiles W as a function of the chunk width, and the tiling either
succeeds or it does not — so the shape has to be compiled to know.

Compiling is not executing, and this projection demonstrates both halves. Of
the five compile-legal widths, 96 compiles **and loads** and then fails at
dispatch, so the legality set above does not license a width on its own. The
other four dispatch and return correct values.

More decisively, **none of them streams**. Being inside the streaming budget
turns out to be necessary and not sufficient at this contraction: width 8 uses
a fifth of the budget and still reads the 55.1 MiB weight 3.6 times over, and
the other widths read it 4.1 to 9.1 times. So the earlier reading — that width
8 was the one streaming width and the problem was de-amortizing a single fetch
across 16 dispatches — was wrong in the direction that matters. The single
`conv` never streams at any legal width, which is a stronger objection than
poor amortization. The likely cause is that a 21,504 contraction exceeds the
16,384 cap on W, so the compiler splits it internally and re-reads the weight
per split; that is inference from the traffic and not measured directly.

Splitting the contraction explicitly and summing the partials on the GPU is
therefore the design, and it is measured rather than projected. It partitions
the weight instead of duplicating it, so total bytes are unchanged, and each
partial lands on a contraction that does stream: a 3-way split gives 7,168 and
a 4-way split gives 5,376, which is `gate_proj`/`up_proj`'s own contraction.
Dispatched at 7,168 by 5,376 the partial reads 1.05 to 1.14 times its 18.4 MiB
blob across widths 45 to 136 — it streams, and it shows the same 128 step with
traffic flat across it.

Head to head at a 128-token block, for identical total weight bytes: three
split dispatches cost 3.37 ms and read 60.9 MiB, against 10.11 ms and 226.2 MiB
for the unsplit projection. **The split is 3.0x faster on a third of the
traffic**, and the reason is not that it is smaller but that the unsplit form
never streams while the split always does.

What remains unmeasured is the part that is ours rather than the engine's: the
GPU reduction over the partials, both its cost and whether summing in fp16
disturbs numerics. The figures above count ANE dispatches only, so they are an
upper bound on the gain rather than a prediction of it.

The gate/up weights stream compressed only while the padded chunk width stays
inside the activation tile. Width pads to a multiple of 8 before it is
measured, and a quantised weight streams from DRAM while `8 * ceil(T / 8) * C`
is within 983,040 elements; at the 31B contraction of 5,376 the last streaming
width is **176**. Above the tile the compiler expands the weights to dense
fp16 before dispatch, which makes the resident cost several times the INT8
bank's nominal size — sizing a deployment from the INT8 figure alone then
under-counts it.

The tile is not what sets the default. A separate step at 128 gives up most of
the INT8 advantage from 129 upwards, on both dtypes and unchanged when the
output width is halved, so it is neither the streaming tile nor an
output-surface effect. DRAM traffic settles it directly: read volume is flat
across the step — measured at this contraction, 56.7 MiB at 128 against
57.1 MiB at 136 — while wall time steps by about 1.8x. The weight is still
streaming on the far side; only the time changes. Widths 129 to 176 are
therefore legal and still streaming, yet slower than 128, so the width above
the default is available and worse. The mechanism behind the step is open.
This is why the default is 128 rather than a value chosen for throughput; see
the block-width note under Settings.

Two ceilings sit on this contraction and the tighter one binds: 176 is where
the weight stops streaming, 128 is where the dispatch stops being cheap. The
default sits on the second. Neither is a rule about `T` itself — widths that
are neither powers of two nor multiples of 32 stream identically at 40, 72, 88
and 120, so the power-of-two rule that governs an `einsum` lowering does not
reach the `conv` this path compiles.

The step is one riser of a staircase, not a cliff, and the period is 128. Cost
is a fixed term plus `ceil(T / 128)` passes over the weight, flat inside each
group and stepping between them. At a contraction of 1,536, where the budget
allows widths up to 640 and so admits several multiples of 128, three groups
are visible with equal risers: 0.61 to 0.66 ms for widths through 128, 1.00 to
1.04 for 136 through 256, and 1.42 to 1.48 for 264 through 384. `ceil(T / 64)`
does not fit — its 3-to-4 and 5-to-6 transitions cost nothing.

That makes the cheapest width **the largest multiple of 128 inside the
streaming budget**, because each group adds one weight pass while the tokens it
covers double. At the 31B's contraction of 5,376 the budget stops at 176, so
128 is the only multiple of 128 that fits, and the default is right here for
that reason — not because widening never pays. It is why the step looks like a
ceiling at this contraction rather than like the period it is. At 2,816 the
budget reaches 344, 256 fits, and 256 measures 1.14x better per token than 128.
So the settable range above the default is a real lever on other geometries
rather than a footgun, and it should not be clamped.

The penalty scales with the weight rather than with the width, at 0.0201,
0.0195 and 0.0202 ms per MB across three contractions — constant to 4%, and
equivalent to one extra pass over the weight at roughly 50 GB/s. But DRAM
traffic does not rise across a riser, so whatever is re-read does not come from
memory. That is the whole of what is known; the mechanism is open.

A third filter is neither of those, and it is not predictable: compile legality
is non-monotone in the width for `gate_proj`/`up_proj` as well as for
`down_proj`. At the 31B geometry the pair compiles at 45, 72, 80, 96, 128 and
160 — identically at 4 and 8 bits, so legality at this triple does not depend
on the weight dtype — and fails at 32, 40, 48, 56 and 64. The failures form a
band of *small* widths below the 128 floor that both the tuner and the settings
route enforce, so none of them is reachable by configuration. Above 160 nothing
has been compiled, and a bank that fails to compile latches its own module and
leaves that layer on the GPU with a warning, so the failure mode there is a
quiet loss of acceleration rather than an error; `gemma4_ane_prefill_status` is
where it shows up.

Among the legal widths the default is chosen by amortization. At this triple
128 is **2.80x cheaper per token than 45** — 45.0 tokens/ms against 16.1 — with
DRAM read flat across the whole range at 111.3 to 111.9 MiB, so the gain is
nothing but a fixed weight fetch spread over more tokens. 160 is legal and
still streaming and is nonetheless *worse* per token than 128, at 30.0
tokens/ms, because it crosses the 128 step and the step costs more than the
extra width buys. So 128 is the largest width that clears both the legality and
the step filters, and it is the only one of those that is also a multiple of 32
and so needs no strided staging.

The multiple-of-32 rule is a property of this implementation's staging rather
than of the engine, and DRAM traffic cannot see it: a misaligned width streams
at 1.0x. The `[1, C, 1, T]` fp16 IOSurface pads its W axis to a multiple of
32; the engine computes the declared `T` columns and leaves the padding zero,
so a misaligned width is correct on the engine and wrong only to a contiguous
host read, which interleaves real values with that zero padding. This path
reads contiguously, so `ANETuningRequest` rejects a width that is not a
multiple of 32. Reaching 176 would mean staging strided, not relaxing the
check.

CPU fp16 sharing and the fused SwiGLU/down path are also unsupported here:
only the two non-CPU merge kernels are instantiated for GeGLU, and enabling
either combination raises rather than silently computing SwiGLU.

MoE checkpoints (26B-A4B) are rejected at enable time. Their routed experts
run through `SwitchGLU`, which this path does not wrap — and wrapping them
would be a new backend rather than a wider gate. `SwitchLinear` selects a
different expert weight per token through `mx.gather_mm`, which a 1x1 `conv`
cannot express, since a convolution applies one shared weight across every
position. Every ANE program this path compiles is a `conv`. Accelerating
experts therefore means either computing all experts densely and discarding
most of the result, one dispatch per expert over its own token run, or a
batched lowering with its own width rules. The attention projections are
ordinary `nn.Linear`, so they carry none of that. The gate reads
`enable_moe_block` off the language-model config as well as the outermost one,
because a Gemma 4 checkpoint whose top-level `model_type` is `gemma4` loads as
the multimodal wrapper even when its weights are text-only, and that wrapper's
`ModelArgs` does not carry the flag.

The dense per-layer `self.mlp` does survive beside the experts and shares the
same shape, so wrapping it is cheap — but it does not pay. On
`gemma-4-26b-a4b-it-oQ4e`, one 1024-token forward pass of the transformer body
takes 1150.7 ms with the GPU alone, against 1533.5 / 1332.6 / 1309.9 ms with
the dense MLP offloaded at widths 128 / 256 / 320: **0.75x to 0.88x, a
regression at every legal width**. All three widths stream: this checkpoint's
contraction is 2,816, which puts the tile edge at 344. The excess is roughly
1.5 ms per ANE block and does not move with width, which rules out arithmetic:
bank weights are `[O, C, 1, 1]` and cost the same bytes at every width. The
excess is host-side, not the engine. The ANE itself takes about 0.35 ms at this
bank shape, measured twice over — once by this runtime's own per-dispatch
counters and once by an unrelated harness on the same silicon — against a bank
of only 2,048 rows by 2,816, or 5.77 MB at INT8.

Most of the remainder is one Metal command-buffer round trip per dispatch. The
activation is staged into the ANE input surface by a kernel whose own cost is
about 0.02 ms, and the submit-to-complete round trip that reports it finished
costs about 0.20 ms on an M1 Max — measured against a bare commit-and-wait on
the same submission path, which is 0.203 ms with no work in it at all. The
kernel is not the cost; learning that it finished is. Because the engine cannot
start until that staging lands, and in this implementation the host is what
starts it, the round trip sits on the critical path however the host is
notified.

That round trip cannot be removed by restructuring command buffers, and this is
measured rather than assumed. A shared event's signalled value becomes visible
to a host poller only once the queue's pending work has drained, so the
lateness is a property of the queue state and not of where the signal sits: a
signal at the end of a buffer containing nothing else is just as late as one
encoded midway with further encoders behind it — 2.29 ms against 2.23 ms with
equal trailing work — while the same dedicated buffer with nothing queued
behind it reports in 0.54 ms, the duration of its own work. Since the GPU
suffix is always queued after the staging, no arrangement of buffers or signal
positions lets the engine start early.

The host can be taken out of the loop, though, and that is where the remaining
cost lives. The engine accepts a wait event built directly on a Metal shared
event — `+[_ANESharedWaitEvent waitEventWithValue:sharedEvent:]` at the default
event type, bundled by
`+[_ANESharedEvents sharedEventsWithSignalEvents:waitEvents:]` and passed to
the `...procedureIndex:sharedEvents:` request variant — with no bridging,
because Metal's shared event is already an `IOSurfaceSharedEvent`. The wait is
honoured. Against GPU signals delayed 100 to 1500 ms, completion lands 0.43 to
0.60 ms after the signal at every delay, tracking the signal rather than the
dispatch, while a request carrying no events completes some 400 ms *before* the
same signal and an event that is never signalled never releases the request at
all. Signal-to-completion is indistinguishable from an unfenced warm dispatch,
so the fence itself costs nothing measurable. The reverse direction works too:
a request carrying an `_ANESharedSignalEvent` raises a Metal shared event's
value, so the engine can report completion to the GPU without the host either.

Two properties of that path are easy to mistake for hardware faults. Attaching
shared events makes the dispatch **asynchronous** — the evaluation call returns
at submission in about 0.2 ms and completion is reported only through
`request.completionHandler` — so timing the call measures the submit and makes
a working fence look like one that completed before its own signal. And the
completion path invokes that handler unconditionally, so leaving it nil is a
segmentation fault on the framework's own dispatch thread rather than a silent
no-op, on a thread outside any handler this path could install.

The property the trade turns on holds: concurrent GPU work proceeds while a
fenced request is pending. A 32-64 MB blit on its own queue and command buffer,
sized near the engine dispatch so that inflation would show, runs within 4% of
its unfenced time while a fenced request sits pending, against the ~90% a
serializing fence predicts, and `agentMask` does not change it. Coarsely, an
8x64 MB blit completes in 3.897 ms while a fenced request waits 400 ms on an
unsignalled event.

Device-side ordering for the single-engine dispatch sits behind
`OMLX_QWEN35_ANE_FENCE`, settable at runtime so both orderings can be
interleaved in one process. It is worth less than the round trip's headline
cost, and it depends on the width. Wall time per operation at the 26B dense
geometry, arms interleaved on an idle engine, 280 samples each:

| width | host-ordered | device-ordered | ratio |
|---:|---:|---:|---:|
| 128 | 1.002 | 0.797 | 1.257x |
| 256 | 1.144 | 0.938 | 1.220x |
| 320 | 1.290 | 1.436 | 0.898x |

Two counters confirm the arms differ in the intended way rather than one
silently falling back: the host's staging wait collapses from 0.227 ms to
0.011 ms, and the engine bracket from 0.319 ms to 0.076 ms — the latter being
the asynchronous submission, not a faster engine, which is why it cannot be
compared across the two arms. Outputs are bitwise identical across 192 fenced
dispatches at both widths and at four queue depths, so the wait is honoured
under exactly the condition that would expose an engine reading early.

Engine dispatch size decides the sign, independently of the width. Holding the
width at 320 and the suffix identical, and varying only the engine's output
width:

| engine rows | int8 bank | host-ordered | device-ordered | ratio |
|---:|---:|---:|---:|---:|
| 256 | 0.72 MB | 1.416 | 1.287 | 1.100x |
| 512 | 1.44 MB | 1.408 | 1.272 | 1.108x |
| 1,024 | 2.88 MB | 1.387 | 1.355 | 1.023x |
| 2,048 | 5.77 MB | 1.369 | 1.502 | 0.911x |

So a width that loses at one bank size wins at a smaller one, and the useful
statement is about the engine's share of the operation rather than about width
at all. Empirically that favours anything which shortens the engine — a narrower
split, or a sparser or more heavily quantized bank.

Why the sign turns is not established, and two candidate accounts are already
ruled out. It is not the engine holding memory bandwidth: the GPU suffix's
slowdown at width 320 is the same at every bank size above, +0.149, +0.102,
+0.150 and +0.154 ms across an eightfold range of engine bytes. And it is not
the driver being slow to release a waiting request: against GPU signals delayed
100 to 1500 ms, signal-to-completion is 0.43 to 0.60 ms where an unfenced warm
dispatch of the same shape is 0.51 to 0.57, so there is no scheduling latency
left to explain anything.

The finished-last counters cannot arbitrate it, for the same reason the engine
bracket cannot: the two arms learn of engine completion through different paths,
a synchronous return against an asynchronous callback, so the fenced arm records
the engine finishing later partly as an artefact of how it is told. What is
real, on a counter measured identically in both arms, is that the GPU suffix's
own completion moves out by 0.114 ms at width 320 while barely moving at 128.
The suffix genuinely slows; nothing measured here says why.

That the mechanism is open is an argument for choosing this per configuration
from measured wall time rather than from a rule, which is what the tuner already
does for the ANE/GPU split itself.

The 31B dense geometry has a far larger dispatch than anything measured here and
is untested; its sign should not be assumed from these numbers.

The round trip's cost is a property of an idle queue, not a floor. Removing it
saves 0.21 ms with nothing else in flight and essentially nothing under load:
with three chained 2048x2048 matmuls queued ahead of the pack, the operation
costs 6.980 ms host-ordered against 6.908 ms device-ordered, a 1.010x wash at
every width. The host's staging wait does show the queue drain it is blamed for,
rising to 5.74 ms — but the wait moves into the launch-to-join bracket rather
than disappearing, because the pack kernel is queued behind that work on MLX's
own stream and no device-side ordering lets it jump the queue. The round trip is
worth removing on an idle engine; under load the serialization is the queue's,
and this does not address it.

The two-engine dispatch paths opt out at the ticket and keep the host wait: one
pack buffer releasing two engines needs two wait events and two tickets retired
from a single pack failure.

On a real forward pass at the share the tuner selects, the ordering is worth a
few percent. Prefill throughput in tokens per second, measured through the
server with the tuner's own GPU-only arm as a per-run control the ordering
cannot affect:

| model | share | ordering | GPU only | hybrid | hybrid samples | hybrid/GPU |
|---|---:|---|---:|---:|---|---:|
| 31B dense | 0.30 | host | 129.80 | 144.5 | 141.6, 144.5, 144.8 | 1.1133 |
| 31B dense | 0.30 | device | 129.65 | 148.0 | 146.2, 148.0, 148.2 | 1.1415 |
| 12B unified | 0.53 | host | 337.25 | 387.3 | 383.2, 387.3, 390.3 | 1.1484 |
| 12B unified | 0.53 | device | 339.10 | 391.2 | 389.4, 391.2, 393.5 | 1.1536 |

So +2.4% on the 31B and +1.0% on the 12B, against control drift of 0.12% and
0.55%. The 31B figure is the sounder of the two: its fenced samples do not
overlap its unfenced ones at all, where the 12B's do. Put against what the
offload itself earns, the ordering supplies about a fifth of the 31B's total
14.2% gain over GPU-only and about a thirtieth of the 12B's 15.4%.

That the 31B gains more is consistent with the crossover above rather than in
tension with it. The 31B runs a 5,376 contraction at share 0.30 and the 12B a
3,840 at 0.53, so the two sit at different points of the engine-share
relationship, and the synthetic sweep is not sensitive enough at these sizes to
say which term dominates. Both are far below the 1.20-1.26x the single-projection
loop reports, which is the expected direction: a round trip removed once per
dispatch amortizes against a whole forward pass, and the loop measures the
dispatch alone.

Two candidates are ruled out by measurement rather than argument, and are
recorded so they are not proposed again. Per-dispatch thread creation is not
the cost: this path spawns an evaluation thread per dispatch, and the interval
from submission to that thread entering the engine call is 0.022 ms. Neither is
the dequantization node in the bank program: it is real, but at this bank size
it is a few hundredths of a millisecond. That block carries only gate/up of the dense MLP — 11.6% of active
projection FLOPs, 5.8% at `fraction` 0.50 — which is too little work to repay a
dispatch. The 26B case needs the attention projections, not this.

> This arm is a **microbenchmark, not throughput**: one cache-less forward pass
> of `language_model.model`, so it excludes the LM head, the paged KV cache,
> chunking and all server work, and it runs at a short context where the
> attention-score term is still small. Its tok/s therefore reads far above the
> throughput benchmark — 892 here, against 630 for the same model once the LM
> head is included at a 4096-token context — and the two must not be compared.
> Only the ratio between the two arms carries, and on a fuller denominator the
> same absolute penalty is a proportionally smaller regression.

The E-series variants remain out of scope. The unified 12B does not -- it
carries the same MLP class as the dense text path, so the existing dispatch
tuple already accelerates it. `omlx/patches/gemma4_ane_prefill.py` records each
variant and its support state in `VARIANT_SUPPORT`.

## Requirements

- A Gemma 4 checkpoint quantized affine at 4, 5, 6 or 8 bits with group size
  64 or 128 on `gate_proj`/`up_proj`. A uniform 3-bit quant silently disables
  the feature: no `qwen35_q3_affine_qmm_t` GPU suffix exists to pair with the
  ANE prefix.
- A source build with the native extension
  (`OMLX_WITH_CUSTOM_KERNEL=1 pip install -e .`). A prebuilt extension
  predating the GeGLU merge fails `qwen35_ane_fused_geglu_available()` and
  the path stays off rather than erroring.
- `OMLX_QWEN35_ANE_PREFILL=0` disables the shared ANE runtime, this path
  included.

## Settings

```json
{
  "gemma4_ane_prefill_enabled": true,
  "gemma4_ane_prefill_sequence_length": 128,
  "gemma4_ane_prefill_tail_padding_min_tokens": 0,
  "gemma4_ane_prefill_fraction": 0.50,
  "gemma4_ane_prefill_max_layers": 60,
  "gemma4_ane_prefill_dual_ane": true
}
```

The controls are exposed in the web per-model settings editor for detected
Gemma 4 models. There are no GDN keys (Gemma 4 has none), no down or
fused-down keys, and no CPU keys.

`gemma4_ane_prefill_dual_ane` defaults on and means *allow* dual, not *force*
it. `enable_gemma4_ane_prefill` detects the physical ANE count and falls back
to a single procedure bank on anything that is not an Ultra part, detection
failures included; only Ultra parts present two instances. The fallback is
what makes the default safe, because the ANE stack reads no instance-pinning
hint: two banks on a one-engine machine would both land on `ane0`, where two
submitting threads measure 0.82x to 0.89x the aggregate throughput of one.
Total ANE bytes are unchanged either way, since each bank holds half the
slice.

`gemma4_ane_prefill_fraction` is a starting point, not a target: the split
tuner searches down from a memory-derived ceiling. 1.0 is not a valid
"let the tuner decide" value — it leaves no GPU suffix, and a layer with no
suffix is rejected, so every layer would be skipped.

Enabling the feature realigns the paged cache block size to the fixed ANE
shape, which rebuilds this model's SSD cache once.

## Memory

`gate_proj` plus `up_proj` across all 60 layers of the 31B is 13.87 GB at
INT8, so the offload costs `13.87 * fraction` GB on top of the checkpoint,
the KV cache and the boundary snapshots. Bank weights arrive as a
file-backed copy-on-write mapping rather than dirty anonymous memory, so
they compete for physical RAM but sit outside the Metal working set — which
is why the tuner derives its fraction ceiling from the bank compiler's
`phys_footprint` headroom gate rather than from the working-set budget.

Each individual bank must also stay inside the roughly 3.76 GB device-virtual
aperture, so any fraction above about 27% engages the split-bank ladder even
in single-ANE mode.

## Measured results

M1 Max (32 GB, one ANE), native source build, on the real 31B MLP
geometry — hidden 5,376, intermediate 21,504, affine q4 group 64 — at a
2,048-token block and a 0.50 offload fraction.

### Reading engine timings at all

A fresh shape's first engine timings read high, by enough to invert a
conclusion. At a 2,816 contraction by 2,048 rows the first three readings of a
series run 0.846, 0.584 and 0.369 ms, a span of 2.3x, and settle at 0.364 to
0.372 ms thereafter. The GPU arm shows nothing comparable. So an absolute engine
time is only meaningful after that settling, and the first readings of every
shape have to be discarded — including after switching back to a shape measured
earlier, since changing banks is not free.

Interleaving is the arrangement that survives this. Compile every arm up front,
dispatch them round-robin in short blocks inside one process, discard the first
few dispatches after each switch, and report each arm's first-half and
second-half medians. When the per-arm medians move while the ratios between arms
hold, whatever is moving is common-mode and the comparison survives it. A run
worth quoting shows within-run variation of 0.98 to 1.00 and between-arm ratios
agreeing to about 0.02 across the halves; repeating the first arm last is a
cheap return control. Every ratio below comes from a run that passes those
checks.

Dispatch cadence also moves an engine timing, though mildly on this path. With
the gap between dispatches imposed by a busy-wait, the median at a 2,816
contraction is flat from back-to-back through 300 us — 0.337, 0.335, 0.347 ms —
then 0.365 ms at 600 us and 0.421 ms at 1 ms, so 1.09x and 1.26x. That bounds a
real concern: `gap_before_ns` reads about 94 us in an isolated loop but about
574 us with unrelated work in flight, which is closer to a forward pass, and the
penalty there is around 9% rather than anything that would overturn a ratio.

A sweep that imposes the gap with `usleep` instead reports 2.91x at that same
1 ms, and the difference is the sleep rather than the engine. Holding the idle
constant and changing only whether the dispatching thread sleeps or stays hot
moves the median from 0.925 ms to 0.360 ms, because a thread that must wake
before it can submit puts its own wake inside a bracket that begins at
submission. This path's figures say the same without needing that control: one
operation takes about 1.02 ms of wall time against 0.34 ms of engine time, so
even at an imposed gap of zero the engine already idles some 680 us between
dispatches, and the median there is 0.337 ms rather than the 0.834 ms such a
curve predicts. An engine idle 680 us dispatching at plateau cost is
incompatible with a penalty that begins at 500.

One instrumentation trap belongs with that. `qwen35_ane_profile_reset()` clears
`g_previous_ane_done_ns` along with the counters, and `gap_before_ns` is only
recorded when that timestamp is non-zero. So resetting around every dispatch —
the natural way to collect per-dispatch samples rather than totals — reads the
gap as zero every time, silently. Collect gaps over an unreset window and divide.

Two further cautions apply to absolute times specifically, and neither affects a
ratio measured this way. The engine is a single shared resource, so another
process dispatching to it inflates readings arbitrarily — a sweep run beside an
unrelated ANE benchmark on this machine read 2.75x high and was attributed to a
drift in the engine before the concurrent user was accounted for. And this
path's medians sit 11 to 16% above its own minima, widening with the pass count,
so a median and a minimum are not interchangeable here: the minima agree with an
independently fitted cost model to within 6% while the medians do not.

### Which geometries pay

These are single-projection figures and they do not predict throughput. Measured
end to end through the server at the share the tuner selects, the 12B unified
runs 1.148x GPU-only where this table's best 12B row says 1.57x. The 26B is not
reachable at all, because an explicit guard skips MoE checkpoints. Read the rows
below as a statement about one dispatch's cost against contraction and width,
which is what they measure.

The 26B row also understates that geometry, because it prices the offload by
bytes and prefill is paid in arithmetic. Its dense shared MLP is intermediate
2112 where each routed expert is 704, and `top_k_experts` is 8, so the shared
projection carries 3 of every 11 units of per-token MLP arithmetic — 27%, or 18%
counting only the gate/up pair this path splits. The banks for it are 357 MB of
INT8 across 30 layers, against 2.77 GB for the 31B at a 0.20 share. That is
nearly the same arithmetic share for an eighth of the residency, on a projection
every token passes through, so its shapes are fixed and no routing is involved.
Since residency is the constraint that actually bound this path on the 31B, the
MoE checkpoint is the most attractive of the unsupported variants rather than
the least.

Per token, one 1x1 `conv` prefix plus its GPU suffix and merge against a single
quantized matmul over the whole projection, at a 0.50 fraction, interleaved:

| geometry | contraction | width | hybrid | GPU only | ratio |
|---|---:|---:|---:|---:|---:|
| 26B-A4B dense MLP | 2,816 | 128 | 0.00775 | 0.00486 | 0.63x |
| 26B-A4B dense MLP | 2,816 | 256 | 0.00448 | 0.00386 | 0.86x |
| 26B-A4B dense MLP | 2,816 | 320 | 0.00410 | 0.00373 | 0.91x |
| 12B unified | 3,840 | 128 | 0.01346 | 0.01681 | 1.25x |
| 12B unified | 3,840 | 256 | 0.01008 | 0.01585 | 1.57x |
| 31B dense | 5,376 | 128 | 0.02127 | 0.03130 | 1.47x |
| 31B dense | 5,376 | 160 | 0.02253 | 0.03107 | 1.38x |

What sets that boundary is this path's fixed per-dispatch cost, not the engine's
rate. The engine is competitive at the losing contraction: 0.318 ms for 2,048
rows at width 128 against 0.225 ms for the same rows on the GPU, a matched-shape
ratio of 1.41 narrowing to 1.21 at width 256, both arms interleaved in one
process on an otherwise idle engine over 495 dispatches per width. Those are
medians, which is the like-for-like statistic here: each engine sample is one
dispatch while each GPU sample is a mean over 32 matmuls, so averaging
suppresses the GPU arm's spread to 1.05x against the engine's 2.73x and a
minimum-against-minimum ratio would compare that arm's mean to this one's low
tail. The ratio is not sensitive to the choice — 1.32x at the minima, 1.37x at
the tenth percentile, 1.41x at the median — but the statistic has to be named,
because a single engine minimum is not converged. And the 26B's deficit is small — 0.118 ms
per dispatch at width 320, 1.312 ms against 1.194 — against a staging round trip
worth 0.21 ms on an idle engine at widths 128 and 256.

That the deficit is smaller than an addressable overhead does not by itself
close the gap, and at width 320 the two do not even have the same sign: the
device-side ordering costs 0.13 ms there rather than saving it. Whether the
offload pays at this geometry is a question for the end-to-end prefill A/B
rather than for either microbenchmark, since both draw the hybrid as a single
operation paying a full command-buffer round trip while a GPU-only arm measured
as a burst amortizes that round trip away — an asymmetry that biases the
comparison against the hybrid by construction.

So an offload pays where the projection is large enough for a fixed overhead of
about 0.7 ms to disappear into it. The 26B is the case where the whole GPU-only
projection costs less than that overhead, which is why it loses at every width
and why the loss shrinks as the width grows and amortizes the overhead over more
tokens. Taking the host out of the dispatch is a throughput gain on the
geometries that already pay, and at widths 128 and 256 it is of the same order
as the 26B's deficit — but it is not on its own enough to move this boundary,
and at width 320 it moves it the wrong way.

The finished-last counters say the same from the other side, and remain the
cheapest single test of whether a projection is worth moving: at the 31B
geometry the GPU suffix finishes last in every dispatch, so the engine is fully
hidden, while at the 26B the engine finishes last in most of them.

Width is a separate lever from geometry and the two do not have the same
optimum, and two separate rules act on it. The 31B is best at 128 because its
streaming budget stops at 176, so 128 is the largest multiple of 128 that fits;
160 also streams and is still worse, because it buys 1.25x the tokens for two
weight passes instead of one. That is the staircase acting alone, with no
folding involved. Folding is the other rule and it is much more expensive: at a
7,168 contraction, widths 136 and 144 carry identical weight bytes and both cost
two passes, yet 136 streams at 1.99 ms while 144 exceeds the budget and folds at
2.82 ms. The staircase explains why both cost about twice their width-128
figure; only the budget explains the 0.8 ms between them. The 12B reaches 256
and gains 28% by using it. The 26B improves monotonically with
width and still does not reach parity at the widest legal setting, so no width
rescues it.

Engine cost is not flat in the width either. A dispatch reads its weight set
once per `ceil(T/128)`, so the cost is a fixed term plus one pass per 128
columns, and the extra passes cost time without adding DRAM traffic. On a quiet
engine at the 26B bank the medians are 0.318 ms at width 128 and 0.451 ms at
256, implying a 0.133 ms pass over a 0.185 ms intercept; the minima, 0.294 and
0.428, give the same pass over a lower intercept. An independent fit
across a tenfold range of bank sizes and a fourfold range of output widths gives

    0.204 + ceil(T/128) * (0.028 + weight_bytes / 44.6 GB/s)
          + O * T * 2 / 44.6 GB/s

with the output term paid once rather than per pass, and predicts 0.373 ms where
this path measures 0.296 — so the shape of the model is established and its
constants are not yet reconciled. A cost model that omits the width will
mis-predict every width but the one it was fitted at, whichever constants it
carries.

### Numerics

Against the stock GPU path, `down_proj(geglu(gate_proj(x), up_proj(x)))`:

| quantity | single ANE | dual ANE |
|---|---:|---:|
| output cosine | 0.999992 | 0.999992 |
| relative RMS vs the GeGLU reference | 0.40% | 0.40% |
| relative RMS vs a SwiGLU reference | 12.77% | 12.77% |

The SwiGLU row is the control that matters: a merge that applied the wrong
activation would land near it, and the GeGLU reference is 32x closer. The
0.40% residual is the INT8 quantization error of the ANE prefix over half the
channels, and it is the whole cost of the approximation — the GPU suffix is
exact. No output was non-finite, so the signed fixed-point accumulator does
not saturate over the 5,376-wide contraction at real activation magnitudes.

Two identity controls hold: a decode-shaped input passes through the wrapper
bit-identically to the GPU path, and `release_gemma4_ane_prefill()` restores
bit-identical GPU output.

### Isolated MLP body

Four real-shaped layers, blocked arms, N=11 after 10 warm-up rounds:

| arm | median | best | per layer (best) |
|---|---:|---:|---:|
| ANE + GPU | 594.6 ms | 457.6 ms | 114.4 ms |
| GPU only | 846.5 ms | 718.3 ms | 179.6 ms |

**1.42x median, 1.57x best** on the offloaded operation. This is an upper
bound on the end-to-end gain, not a prediction of it: the microbench times
only the MLP, so attention, the embedding and the LM head are absent from the
denominator. Bank compilation costs 2.5 s to 3.3 s for four layers.

The arms are blocked rather than interleaved, which is the opposite of the
usual advice and deliberate. Interleaving cancels thermal drift, but it also
leaves the ANE idle for the whole of each GPU arm, and calls after any gap run
well above steady state. Interleaved, the same measurement is bimodal — 0.86x
median against 1.52x best — and that dispersion is larger than the drift the
interleaving removes. The ANE itself does not thermally throttle.

### End-to-end prefill throughput

The only end-to-end numbers are the tuner's own arms in the ordering table
above: 31B +11.3% at share 0.30 with the default ordering. The benchmark is
not yet run. Its method is the built-in throughput benchmark
(`code_python` context, TG=128, greedy, fresh server and cleared SSD cache per
configuration, `Full · 2048` warm-up, ANE prompt alignment on) at 4K / 16K /
32K, reporting median and dispersion over N >= 5.

Three controls gate any number reported there, because output divergence is
not a valid oracle — the Qwen path recorded 16K and 32K output hashes matching
the GPU path exactly while the ANE was demonstrably running:

1. `OMLX_ANE_PROFILE=1` must show 60 MLP operations and 0 GDN per prompt.
2. `[benchmark-prefill]` must show both arms taking the same chunk sequence
   and cache block size. A nominal 4K request prefills 4,095 tokens, one short
   of the ANE shape, so a gain there would come from block realignment
   collapsing prefill into a single chunk rather than from ANE execution. 4K
   is structurally a wash; the ANE should contribute from 16K up.
3. `vm_stat` swap-in/swap-out deltas must be zero across each arm. At 18 GB of
   checkpoint plus up to about 6 GB of bank on a 32 GB machine this is live,
   and one swap event makes the arms incomparable.

Like the Qwen variant this is an approximate path, not bit-exact inference.
