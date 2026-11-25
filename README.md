<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png"><img src="assets/basalt-lockup-light.png" alt="Basalt" width="480"></picture></p>

Basalt is a local inference engine for large language models: **Qwen3.8-Flash-Next**, **DeepSeek-V4.1-Flash**, **Qwen3.8-27B**, **Hy4-flash** and **Inkling 2 Mini**. It runs
on an NVIDIA Blackwell GPU (sm_120) on Linux and Windows x64, and on the DGX Spark (GB10, Arm64), with what does not fit in VRAM
(the experts of a mixture-of-experts model) in system RAM, and optionally on more GPUs. The server speaks the OpenAI and Anthropic APIs. Website: [basalt.jgulfo.com](https://basalt.jgulfo.com).

<p align="center"><img src="docs/img/frontend.gif" alt="The Basalt web UI while a reply streams" width="720"></p>

**Help wanted:** spare Blackwell compute for testing? See [Contributing](CONTRIBUTING.md#other-ways-to-help).

## This fork: Ada (sm_89) support

This is a fork of Basalt that adds an **Ada (sm_89) build** so **Inkling 2 Mini** runs on an RTX 4090. Upstream Basalt is
Blackwell (sm_120) only; this fork keeps that path unchanged and adds a second one. Only Inkling 2 Mini is wired for sm_89.

What differs from upstream:

- The build takes `-DCMAKE_CUDA_ARCHITECTURES=89` and defines `BASALT_ADA`. The Blackwell-only prefill products (the FP4
  block-scaled tensor-core path and the FP8 path) are left out; prefill runs through the portable MMQ path instead.
- The sm_120a-only instructions get portable fallbacks: the FP4/FP8 to bf16 conversions are done in software, the
  `cp.async.bulk.prefetch.L2` hint compiles out, and the cluster attention, GDN and gating kernels fall back to their
  non-cluster split forms.
- The DSpark drafter works, so drafting is on by default as upstream.

This fork is **not upstreamable**: [CONTRIBUTING.md](CONTRIBUTING.md#scope) states the project is sm_120 only and refuses
`#if` branches for older architectures. It is meant to be built and run locally.

### Build

Windows x64, from an x64 VS 2022 prompt, with CUDA 13 and Ninja:

```sh
cmake -S . -B build_sm89 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 -DBASALT_MODELS=inkling2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_sm89 -j 8
```

`-DBASALT_MODELS=inkling2` builds only Inkling 2 Mini. Drop it to build every model, but only Inkling 2 Mini runs on sm_89.
The binaries land in `bin/` (`basalt`, `basalt-bench`, `basalt-probe`, `basalt-vision`).

### Get the pack

```sh
pip install -r requirements.txt
hf download jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-DQ8.basalt --local-dir ".\models\Inkling-2-Mini-Basalt"

hf download jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-MX8.basalt --local-dir ".\models\Inkling-2-Mini-Basalt"

hf download jesdga/Qwen3.8-27B-Basalt Qwen3.8-27B-QUASAR-NVFP4.basalt --local-dir ".\models\Qwen3.8-27B-Basalt"
```

The pack is one file, `Inkling-2-Mini-NVFP4-DQ8.basalt` (19.7 GB). It needs about 16.2 GiB of VRAM on the card, so it fits an
RTX 4090 (24 GB) with room for the context.

**Both packs run on sm_89.** The MX8 pack (`Inkling-2-Mini-NVFP4-MX8.basalt`, 19.6 GB) stores its experts as MXFP8. On
Blackwell that goes through the block-scaled tensor-core kernel; on Ada the same weights go through the portable MMQ
path, which now has an MXFP8 tile: the E4M3 codes are quantized to int8 per 32-value sub-block and the UE8M0 scale is
folded into the float the epilogue multiplies by. The MX8 pack loads and generates on an RTX 4090.

### Run

The repo ships `basalt_config.json`, which names the pack, the card, the context and the KV format:

```json
{
  "version": 2,
  "models": [
    {
      "model": "models/Inkling-2-Mini-Basalt/Inkling-2-Mini-NVFP4-DQ8.basalt",
      "gpu": "0",
      "max-context": 589824,
      "kv": "fp8"
    }
  ],
  "engine": { "vram-reserve-mib": 700, "ram-reserve-mib": 1024, "fit-max-tokens": true, "resident-budget-gib": 8.5 },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

```sh
bin/basalt serve --config basalt_config.json --python .venv/Scripts/python.exe
```

The repo also ships `basalt_config_mx8.json`, the same settings pointed at the MX8 pack. It runs on both Blackwell and
Ada (see [Get the pack](#get-the-pack)).

```sh
bin/basalt serve --config basalt_config_mx8.json --python .venv/Scripts/python.exe
```

Those two keep a long context (589,824 tokens) and spill the experts the card cannot hold into RAM, computing them on
the CPU. For the fastest decode, `basalt_config_dq8_fullgpu.json` and `basalt_config_mx8_fullgpu.json` instead keep
**every expert on the card** and no RAM copy: a 131,072-token context, which is the largest that fits beside all
14.7 GiB of experts on a 24 GB card. Decode runs about 150 tok/s (DQ8) and 156 tok/s (MX8) with a 100% expert-cache
hit and no PCIe traffic, against the RAM-spill configs' slower CPU-computed experts. Use them when the context fits;
use the long-context configs when it does not.

```sh
bin/basalt serve --config basalt_config_dq8_fullgpu.json --python .venv/Scripts/python.exe
bin/basalt serve --config basalt_config_mx8_fullgpu.json --python .venv/Scripts/python.exe
```

Model paths in the file are relative to the file itself. The server listens on 127.0.0.1:8080 and speaks the OpenAI and
Anthropic APIs. The `--python` flag points at a Python with `requirements.txt` installed (a venv is fine); the server uses it
for tokenizing and the web UI. Then:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Inkling 2 Mini","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64}'
```

The same run without a config file:

```sh
bin/basalt serve ./models/Inkling-2-Mini-Basalt/Inkling-2-Mini-NVFP4-DQ8.basalt --python .venv/Scripts/python.exe --max-context 589824 --kv fp8
```

Useful flags: `--max-context N` (1,024 to 1,048,576, default 8,192), `--kv fp16|fp8`, `--spec N` (drafting window, default 8;
`--spec 1` turns drafting off), `--parallel N` (sessions). Every setting: [configuration](docs/configuration/index.md).

**Context on a 24 GB card with 32 GB RAM.** The weights take 16.2 GiB of the card, so the context has what is left, and
experts spill to RAM to make room. `kv fp8` costs 16 KiB per position against `fp16`'s 32 KiB. On this machine the ceiling is
**589824** positions: it loads with 6.8 GiB on the card and 9.4 GiB of experts in page-locked RAM, and serves at about
150 tok/s. Above that the load fails on RAM, not VRAM — 655360 wants 11.6 GiB of pinned experts and the machine cannot spare
it, even with `ram-reserve-mib` lowered. The ceiling moves with the free RAM at start: close other programs to raise it, or
lower `max-context` to leave headroom.

**`resident-budget-gib` keeps long prompts alive.** Left auto, the engine pins as many experts as fit (10.7 GiB here) and
leaves only about 2 GiB free — right at the 2048 MiB RAM guard. A long prompt then needs host buffers for its tokens, RAM
drops below the guard, and the request is stopped with `RAM low ... lower --resident-budget-gib`. The shipped config caps the
pinned experts at **8.5 GiB** (1859 of 3072), which frees ~1.3 GiB and lets an 84K-token prompt finish. Lower it further if
you still see the warning; raise it toward 10.5 for a little more decode speed if your prompts stay short.

**`ram-reserve-mib` sets the guard itself.** The guard warns when free RAM falls below `ram-reserve-mib` (default 2048) and
stops a request when it stays below for 10 s. With the experts pinned, free RAM settles near 2.0 GiB, so the default guard
sits right on the line and prints `RAM low` warnings during long prompts. The shipped config lowers it to **1024**, which
puts the guard below the settled free RAM: the warnings stop and requests are not killed, while the fixed 1 GiB hard floor
still protects the system. `--ram-reserve-mib 0` turns the guard off entirely.

`vision` is off in the shipped config; `"vision": "dynamic"` adds no card memory but stages the encoder in RAM.

On an RTX 4090 this fork decodes at roughly 120 to 190 tok/s on a short chat with drafting on, and reads prompts at roughly
100 to 140 tok/s. The model's own guide, with its architecture, memory and quality notes, is [Inkling 2 Mini](docs/models/inkling2.md);
its numbers there are for an RTX 5090 and do not apply to sm_89.

### Tests

```sh
cmake -S . -B build_sm89 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 -DBASALT_MODELS=inkling2 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build_sm89 -j 8
ctest --test-dir build_sm89 --output-on-failure
```

The C++/CUDA suite passes on sm_89 (the FP8/FP4-only tests are skipped, since those products are not built).

<p align="center">
  <a href="https://github.com/sponsors/jesdga95"><img src="assets/donate/github.svg" height="44" alt="Sponsor on GitHub"></a>
  <a href="https://ko-fi.com/jesdga"><img src="assets/donate/kofi.svg" height="44" alt="Support on Ko-fi"></a>
  <a href="https://paypal.me/jesdga"><img src="assets/donate/paypal.svg" height="44" alt="Donate on PayPal"></a>
</p>

## Requirements

- Linux x86-64 with AVX2, or Windows 10 1903+ x64, with an NVIDIA Blackwell GPU (compute capability 12.0) ([platforms](docs/platforms.md)).
- Or a DGX Spark or another GB10 machine: Linux Arm64 with its integrated Blackwell GPU (12.1) ([Arm64](docs/arm64.md)).
- CUDA 13.
- CMake 3.24+, GCC 13+ (Linux), Python 3.10+, and enough system RAM for the experts that do not fit in VRAM.

## Quick start

Prebuilt Linux (x64 and Arm64) and Windows archives are attached to each [release](https://github.com/jesdga95/basalt/releases) ([what they need](docs/platforms.md#release-archives)); otherwise build from source:

```sh
git clone https://github.com/jesdga95/basalt && cd basalt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j                  # bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision
                                        # Windows: add -G Ninja to the first line, from an x64 VS 2022 prompt ([platforms](docs/platforms.md#building-on-windows))
                                        # one model only: -DBASALT_MODELS=qwen4exp, deepseek41, qwen35, hy4flash or inkling2 ([building](docs/building.md#building-only-some-models))
pip install -r requirements.txt
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-NVFP4-DQ6_K.basalt --local-dir models
```

Put the settings in `basalt_config.json`:

```json
{
  "version": 2,
  "models": [{ "model": "models/Qwen3.8-FN-NVFP4-DQ6_K.basalt", "gpu": "0,1", "gpu-mode": "peer", "kv": "int8" }],
  "engine": { "resident-experts": true },
  "server": { "port": 8080 }
}
```

```sh
bin/basalt serve --config basalt_config.json
```

The server listens on 127.0.0.1. Every setting: [configuration](docs/configuration/index.md). More examples:
[example configurations](docs/configuration/examples.md).

## Models

| | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash | Qwen3.8-27B | Hy4-flash | Inkling 2 Mini |
|---|---|---|---|---|---|
| Packs | default `Qwen3.8-FN-NVFP4-DQ6_K`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) | default `DeepSeek-V4.1-Flash-1017-UD-IQ1_M`, [Hugging Face](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt) | `Qwen3.8-27B-QUASAR-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-27B-Basalt) | `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt) | `Inkling-2-Mini-NVFP4-MX8`, [Hugging Face](https://huggingface.co/jesdga/Inkling-2-Mini-Basalt) |
| Context | up to 262,144 (longer with YaRN) | up to 1,048,576 | up to 262,144 | up to 1,048,576 | up to 1,048,576 |
| Drafting | MTP, on by default | DSpark, on by default | DFlash 2, on by default | DFly2, on by default | DSpark, on by default |
| Images | yes | yes | yes | yes | yes |
| Documents (PDF, text) | yes | yes | yes | yes | yes |
| Guide | [guide](docs/models/qwen38-flash-next.md) | [guide](docs/models/deepseek41.md) | [guide](docs/models/qwen38-27b.md) | [guide](docs/models/hy4-flash.md) | [guide](docs/models/inkling2.md) |
| Precision | [precision](docs/precision/qwen38-flash-next.md) | [precision](docs/precision/deepseek41.md) | [precision](docs/precision/qwen38-27b.md) | [precision](docs/precision/hy4-flash.md) | [precision](docs/precision/inkling2.md) |

All models compared, with each one's context limit, flags and cards: [models](docs/models/index.md).

## Video

Qwen3.8-Flash-Next and Qwen3.8-27B read video; DeepSeek, Hy4-flash and Inkling 2 Mini do not. A video is named like an image in a chat message, and needs
`--vision` and FFmpeg on the machine. Flash-Next decodes frames on the GPU with NVDEC; the 27B decodes them on the CPU.
How a request names a video, and the limits: [video](docs/running.md#video).

## Documentation

- **Run:** [running](docs/running.md), [configuration](docs/configuration/index.md), [models and the router](docs/configuration/models.md),
  [sampling](docs/configuration/sampling.md), [image sources](docs/configuration/image-sources.md),
  [documents](docs/configuration/documents.md),
  [diagnostics](docs/configuration/diagnostics.md), [basalt-probe](docs/probe.md), [platforms](docs/platforms.md).
- **Models:** [models](docs/models/index.md), [Flash-Next](docs/models/qwen38-flash-next.md), [DeepSeek](docs/models/deepseek41.md),
  [Qwen3.8-27B](docs/models/qwen38-27b.md), [Hy4-flash](docs/models/hy4-flash.md), [Inkling 2 Mini](docs/models/inkling2.md),
  [the `.basalt` file](docs/model-file.md), [file format changes](docs/basalt-file-changelog.md).
- **Quality:** [precision](docs/precision/index.md), [method](docs/precision/quality.md).
- **Build and develop:** [building](docs/building.md), [adding a model](docs/adding-a-model.md), [community runs](docs/community.md),
  [changelog](docs/changelog/index.md), [contributing](CONTRIBUTING.md).

## Credits

- **Code:** [Strata](https://github.com/Niko1221/Strata), the project Basalt forks (MIT); [Infernix](https://github.com/Wallawalla47/Infernix)
  (Ian Ranson and contributors, Apache-2.0), whose kernels are adapted; [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp)
  (MIT); and ideas from [NInfer](https://github.com/Neroued/ninfer). Licences: [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- **Models:** [Qwen](https://huggingface.co/Qwen), [DeepSeek](https://huggingface.co/deepseek-ai), [Tencent](https://huggingface.co/tencent) (Hy4-flash) and Thinking Machines Lab (Inkling 2 Mini), under the licences on their cards;
  [Unsloth](https://huggingface.co/unsloth)'s GGUFs; [NVIDIA](https://huggingface.co/nvidia)'s NVFP4 checkpoints;
  [QUASAR-QAT](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)'s NVFP4 27B (Apache-2.0); [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)'s DFlash 2 drafter.

Thank you to everyone who has [contributed](https://github.com/jesdga95/basalt/graphs/contributors) or
[sponsored](https://github.com/sponsors/jesdga95) the project.

## License

MIT, see [LICENSE](LICENSE).
