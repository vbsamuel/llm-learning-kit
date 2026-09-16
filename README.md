# AI Systems Lab

Small, measured experiments in inference serving and accelerator behavior.

I use these notebooks to answer systems questions with controlled runs rather than architecture claims. Each experiment keeps the failure case, the measurements, and the limit of what the data can support.

## Experiments

### 01 · KV cache tiering under long-context pressure

**Question:** when reusable KV state no longer fits in device memory, when is reloading it from a slower memory tier cheaper than recomputing the prefix?

Recorded run:
- warm mean TTFT: **9.315 s → 0.336 s**
- warm improvement: **27.7×**
- forced-eviction 30K-context case: **2.871 s → 0.355 s (8.1×)**
- cold path regressed: **9.377 s → 13.263 s**

The cold regression matters: cache tiering pays only when reuse amortizes the cost of persisting state.

[Notebook](notebooks/01_kv-cache-tiering-under-pressure.ipynb) · [recorded data](data/kv_cache_recorded_run.json)

---

### 02 · Dynamic batching under concurrent inference

**Question:** how should a batcher trade queueing delay against accelerator utilization?

The first configuration failed:

```text
naive                         1,438.9 QPS
max_batch=32                  1,106.6 QPS   ← regression
best measured: max_batch=8    2,923.3 QPS
```

The useful result is not “batching is faster.” It is that configured capacity above offered concurrency can add waiting without creating a larger realized batch. The notebook implements the batcher, load generator, synchronization, and instrumentation directly.

[Notebook](notebooks/02_dynamic-batching-under-concurrency.ipynb) · [recorded data](data/dynamic_batching_recorded_run.json)

---

### 03 · Precision × batch frontier

**Question:** which `(batch size, numerical precision)` operating points maximize throughput while staying inside memory and numerical-error constraints?

Recorded run:
- batch sweep: **3,271.6 → 44,494.9 samples/s**
- fp16: **1.88×** fp32 throughput at the fixed comparison batch
- bf16: **2.26×** fp32 throughput at the fixed comparison batch
- selected recorded point: **bf16, batch=32, 29,140 samples/s**

The tested range did **not** establish a saturation knee: throughput was still increasing at the largest measured batch. That boundary is part of the result.

[Notebook](notebooks/03_precision-batch-frontier.ipynb) · [recorded data](data/precision_batch_recorded_run.json)

---

### 04 · Adaptive optimization under baseline drift

**Question:** when warm-up changes the baseline, what evidence is required before an automated optimizer is allowed to promote a candidate?

Recorded measurements:

```text
cold baseline      1375.208 tokens/s/device
hot baseline       1441.083 tokens/s/device
candidate          1454.510 tokens/s/device
```

The candidate looks **5.77%** better against the cold baseline but only **0.93%** better against the hot baseline. The notebook therefore does not promote it: repeats, quality evidence, and tail-latency evidence are missing.

[Notebook](notebooks/04_adaptive-optimization-under-baseline-drift.ipynb) · [recorded data](data/optimizer_recorded_run.json)

---

## How I use the lab

For every experiment I try to keep the same discipline:

```text
question
→ falsifiable hypothesis
→ controlled workload
→ instrumentation
→ measured result
→ counter-result / failure
→ interpretation
→ claim boundary
→ rerun path
```

The important part is the boundary. A point estimate is not a benchmark distribution. A throughput win is not automatically a latency win. A warm-cache result says nothing about the cold path. A larger tested batch is not automatically an optimum. If the data cannot support a claim, the notebook should say so.

## Repository layout

```text
notebooks/   experiment records + rerun harnesses
data/        machine-readable measurements used by the analysis
```

`EVIDENCE_MANIFEST.json` records the artifacts that make up this snapshot.
