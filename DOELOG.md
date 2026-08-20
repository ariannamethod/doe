# DOELOG

The running engineering log of DOE — Democracy of Experts. Every fix, every
verified change, every bug-class closed — dated, with commit and proof. `README.md`
is the spec and the manifesto; **this is the work**.

Convention: small fixes (bug fixes, hardening, single-op work, docstring touch-ups)
are recorded **here**. Large changes (a new backend, a new physics term, an
architecture shift) get a section in the README too. When in doubt: it goes here
first. Entries are **de-personalized and technical** — what changed, where
(`file:line`), and the reproducible proof. No signatures in the log.

Newest entries on top.

---

## 2026-08-20 — int8 dynamic-activation-quant path becomes the default

`doe_mv_impl` selected the int8 matvec only when `DOE_INT8=1` was set in the
environment (`doe.c:1946`), so every default run took the exact dequant-inline route.
On aarch64 that route is roughly four times slower, and nothing in the output paid for
the difference.

Measured on a Galaxy A56 (Exynos 1580, aarch64, Ubuntu chroot, OpenBLAS, `taskset -c 4-7
--threads 4`), Qwen2.5-0.5B Q4_K_M, 25 decoded tokens, `DOE_DEBUG_TIMING=1`
(`doe.c:4485`):

| path | decode |
|------|--------|
| exact dequant-inline (`DOE_INT8=0`) | 4.68 s — **5.34 t/s** |
| int8 SDOT (default after this change) | 1.11 s — **22.44 t/s** |

Output equality was checked before flipping the default, greedy (`--temp 0.0`,
`--max-new 20`) on three prompts covering arithmetic, recall and translation: *"What is
17 plus 25?"* → `42`, *"Name three prime numbers."* → `Three prime numbers are 2, 3, and
5.`, *"Translate to French: the cat sleeps."* → `le chat dort.` — token-identical on both
paths.

The selector now reads the variable as an opt-out: `DOE_INT8=0` restores the exact path,
anything else keeps int8 (`doe.c:1946`). The path remains approximate by construction —
activations are quantized per matvec — so bit-faithful runs must ask for it explicitly.
`make run-exact` added alongside `run-int8` for that. The int8 route already threads
against `--threads` since the earlier threading fix, so the default no longer pins
matvecs to one core either.

Proof (phone-1): `make test` → **113/113 passed, 0 failed**; `make openblas` clean;
default run 22.44 t/s, `DOE_INT8=0` 5.34 t/s, `DOE_INT8=1` 23.31 t/s, all three producing
the same sentence.

---

## 2026-07-20 — integer-overflow hardening: alloc/copy sizes widened to `size_t`

CodeQL (default setup, threat model `remote`) flagged 16
`cpp/integer-multiplication-cast-to-long` alerts in `doe.c`: `int * int` products
used as allocation or copy sizes that overflow in `int` before the implicit widen to
`size_t`. A crafted or oversized host profile could wrap the product to a small value
and under-allocate a LoRA / KV / RoPE buffer, then write past its end.

Each flagged size expression now casts its leading operand to `size_t`, so the product
computes in the wide type:

- **LoRA experts** — `init_lora_expert` (`doe.c:1749`), `mycelium_save` (`:2662`),
  `mycelium_load` (`:2722`, `:2728`): `dim*rank` / `rank*dim` in `calloc`, `fwrite`, and
  `fread` — the readback comparison `(size_t)(dim*rank)` was itself an int product cast
  after the fact and is now `(size_t)dim * rank`.
- **inference state** — `alloc_infer` (`doe.c:2776`–`2804`): `host_heads*host_head_dim`,
  `host_heads*max_seq`, `host_n_layers*max_seq*kd`, `max_seq*half` in `calloc`.
- **KV reset** — `chat` (`doe.c:3639`): `host_n_layers*max_seq*kd*4` in `memset` — now
  matches the already-cast `kv_bytes` and the sibling reset at `:4050`.

Numerically inert for in-range sizes; C cast precedence makes `(size_t)A*B` compute as
`((size_t)A)*B`. Proof (neo): `make test` → **113/113**; `cc -O2 -Wall -Wextra -c doe.c`
clean apart from the one pre-existing `unused 'hs'` warning. Canon fix — the
`yent-inference` vendored copy was already patched by Codex; this closes the source.
