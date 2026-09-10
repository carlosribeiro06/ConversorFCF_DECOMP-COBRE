# conversor-fcf

Converts a Cobre case — its JSON inputs plus the FlatBuffers policy checkpoint — into a
DECOMP-format `mapcut` / `cortdeco` binary pair, so DESSEM can couple to a Cobre-produced future
cost function. It also emits CSV mirrors of both the cuts it read and the records it wrote, because
the DECOMP artifacts are opaque binaries.

## Implementation mode

This project runs in **Rigoroso** mode: every identifier, comment, docstring and document is in
English, comments are minimal, and the style is terse and publication-grade.

## Tech stack

Python `>=3.12` (developed against 3.14.4). Runtime: `numpy`, `pandas`, `flatbuffers`, `rich`.
Test-only: `idecomp` (read-back oracle), `pytest`. Dev: `ruff`, `mypy --strict`.

Deliberately **no `pyarrow` and no `polars`**: because mapcut records 4-17 are zero-filled and
`n_utv = 0`, the converter reads only JSON and FlatBuffers, never Cobre Parquet.

Tooling lives in the project virtualenv: `.venv/bin/{python,pytest,ruff,mypy,conversor-fcf}`.

## Hard-won facts

Each of these was established empirically against the reference artifacts. Do not re-derive them,
and do not "fix" code that looks wrong because it contradicts an intuition listed here.

- `idecomp`'s `Mapcut.write` raises `NotImplementedError`, so all serialization in this project is
  native; `idecomp` is used only as a read-back test oracle.
- `Cortdeco.read` expects `numero_total_cortes` as **cuts per node**, not the total cut count;
  passing the total yields thousands of all-zero padding rows.
- `Cortdeco.cortes` has a no-op setter (`dados = df` rebinds a local), so it cannot be used to
  mutate a loaded file.
- `StageCuts.state_dimension` is study-global and must **never** size the coefficient array; size
  by `EntityManifestLength()` / `CoefficientsLength()` instead. In the reference case it reads 2211
  everywhere while trunk pools carry only 183 slots.
- `EntitySlot.subindex` is **0-based** for `AnticipatedThermalState` ring slots.
- In `mapcut`, reg 9 is written once per **node** (273 in the reference deck) while reg 10 is
  written once per **stage** (7); the two interleave only across the first seven pairs.
- Reg 9's trailing float64 block, at offset 28, is **populated in the reference and its axis is not
  settled**. The reference holds `ngnl*npat` values — the GNL lag month's hours per load block,
  three per submarket summing to exactly 730.5 = `365.25*24/12`, the same day-count basis as the
  discount series — with its first zero at index `ngnl*npat`. But `idecomp`'s `__le_nono_registro`
  consumes `ngnl*n_estagios` while its own `dados_gnl` strides by `sum(patamares)`; the reader
  contradicts itself and the file agrees with the block half. This project emits zeros there under
  premise P12. Do not populate it without settling the axis: a stage-indexed vector laid into a
  block-indexed slot produces a wrong file that still reads cleanly. Note that examining only
  offsets 0-28 makes the block look absent — the int32 head read as float64 yields denormals around
  6.4e-314, which is not the block.
- **`cortdeco`'s `pi_gnl` coefficients are emitted as zeros under premise P14, and the reduction
  behind them is an open TODO** — the divergence most like P12, and deferred to `ticket-016`
  (`plans/conversor-fcf/epic-08-documentation-deferred/`), which holds the full evidence record.
  Only the *values* are zero: `n_sbm_gnl`, `codigos_submercados_gnl`, the
  `n_sbm_gnl*n_estagios*n_patamares` term of `NCOEF`, the 42-slot span at positions 170-211, the 289
  records and the 7,796,064 bytes are all unchanged. `write_cortdeco` enforces this and
  `assert_gnl_span_is_zero` verifies it on the written file. The `ticket-005` ECO CSVs still carry
  the **real** Cobre GNL values, which is what makes the zeroing auditable rather than opaque.
  Consequently premises **P4 (GNL sign) and P5 (load-block weighting) are DORMANT**: `negate_gnl`,
  `negate_gnl_array` and `gnl_block_weights` remain tested and reachable for `ticket-016`, but no
  real coefficient reaches them. The open problem, in one line: the reference populates only 6 of 42
  GNL slots, and that is consistent both with a *collapse* (DECOMP's GNL state is one declaration per
  `(submarket, block)` and the stage axis is vestigial) and with an *addressing* reading (the axis is
  genuine and that deck's horizon is simply one month) — the two are mutually exclusive, and nothing
  in either artifact separates them. Do not pick one: all three candidate reductions write a
  structurally valid file DESSEM reads without complaint, so a wrong one corrupts the cut's GNL slope
  invisibly. Related: negating a zero yields `-0.0`, whose byte pattern is `0x…80`; the reference
  holds 36 true-zero GNL slots and no negative zero, so `_normalize_signed_zero` removes it.
- `NCOEF` must be computed from the formula
  `1 + n_uhes + n_utv * max_lag + n_sbm_gnl * n_estagios * n_patamares`, never inferred from the
  last non-zero coefficient — the GNL block is dimensioned by stage count but only lag 1 is
  populated.
- `codigos_uhes_jusante` is `int32` on disk but `idecomp` reads it as `float32`; a read-back oracle
  needs `.view(np.int32)`.
- `codigos_submercados_gnl` is `int32` on disk too, despite `idecomp` surfacing `[1.0, 3.0]`. A
  byte-level read of reg 9 shows the clean triple as int32 and denormals around 6.4e-314 as float64;
  the float appearance is `dados_gnl`'s stride bug upcasting the column during `pd.concat`. Reg 4 is
  the genuine on-disk-versus-read-type quirk, not this one.
- `numero_cortes = numero_iteracoes × n_cut_building_nodes` (the reference: 438 = 73 × 6), and
  `mapcut` reg 1's cut heads descend as `head(j) = numero_cortes - j`, 1-based, zero for every
  non-cut-building node (the reference: `[438, 437, 436, 435, 434, 433]`).
- **`write_mapcut` never validates `numero_cortes` against anything**, and `mapcut_record_count` does
  not depend on it. That is why a `mapcut` declaring one cut count beside a `cortdeco` holding
  another is constructible at all, and it is the whole reason the pipeline cross-checks the pair
  after writing both. It also settles how to reproduce that failure: patching the header
  `write_mapcut` receives works, while tampering `write_cortdeco`'s `numero_cortes` argument does
  not — that fails inside `write_cortdeco`'s own "supplied cuts != numero_cortes" guard before
  `cortdeco` is ever created.
- `cortdeco`'s chain pointer at offset 0 is **1-based**, points to the same node's previous cut with
  a stride equal to the number of cut-building nodes, and 0 terminates. In the reference, exactly 6
  of 439 records carry 0 and the other 433 all sit at `own_1based - 6`, with no exceptions.
- `n_records = numero_cortes + 1`. That extra record is the **last** one and is **not** a
  terminator: its pointer is the last cut-building node's head, so it is the next backward-pass cut
  (iteration 74 in the reference), written but not yet exposed in the head table. Nothing points to
  it, because chains only step backwards from the six heads.
- The `pi_gnl` block is submarket-major:
  `1 + n_uhes + n_utv*max_lag + sbm*(n_estagios*n_patamares) + stage*n_patamares + block`. Only
  stage 0 is ever populated, so the reference fills positions 176-178 and 197-199 and nothing
  between — which is why its last non-zero coefficient sits at 199 against a declared `NCOEF` of
  218.
- Cost units: `Cobre = DECOMP x 1000` (DECOMP works in 10^3 R$), verified to 1 ulp on a matched
  pair.
- The discount day-count basis is **365.25**, not 365. Solving the basis from the reference series'
  per-stage ratio `0.997830417741` yields exactly 365.250000, and
  `(1+r)^(-cumulative_days/365.25)` reproduces all seven reference factors to 3.7e-13. With 365 the
  error reaches 9e-6. Cumulative days to the start of stage `k` is `7k` in this deck, so stage 6's
  600-hour span never enters its own factor: the basis was the whole discrepancy.
- The anticipated-thermal (GNL) slots are a **rotating ring buffer**. `subindex` is the physical ring
  position and `delivery_date` is its meaning, and the ring rotates between stages: pool 0 maps
  subindex 0 to 2026-04-01 and 1-5 to 2026-05-01, while pool 5 maps 0-4 to 2026-07-01 and 5 to
  2026-05-01. Keying a DECOMP stage axis off `subindex` scrambles it differently in every stage; key
  off `delivery_date`.
- The **GNL lag month is not derivable** from the case, and premise P15 says so. Both anticipated
  thermals declare `lead_time_hours = 1608.0` while the reference deck declares `lag_meses_gnl =
  (2, 2)`, and 1608 / 730.5 = 2.2013 — so `floor` and `round` both fit that single data point and
  first diverge at 2.5 (any fractional part of 0.5 or more). With one point the mapping from lead
  time to lag month is undetermined, exactly like P14's reduction. `assert_gnl_lead_time_is_evidenced`
  therefore refuses any other lead time instead of converting it. Do not "fix" this by deriving a
  formula: a wrong lag month writes a structurally valid file that DESSEM reads without complaint.
- Load-block hours differ **per stage**, not just in total. Stage 0 is 24/65/79, stages 1-4 are
  15/64/89, stage 5 is 12/61/95 and stage 6 is 51/226/323 (600 h, against 168 h elsewhere). The
  frequently repeated "24/65/79" is stage 0's split alone.
- Cobre GNL coefficients span seven orders of magnitude (`-548523.58` to `-0.0888`, with structural
  zeros), while the oracle's `pi_gnl` spans a 15% band. No scalar or affine map connects them, and
  the two artifacts come from independent runs, so their magnitudes say nothing about the unit
  convention. Never gate a validation check on cross-run GNL magnitude agreement.

## Git conventions

Work on a feature branch; never commit to `main`, never force-push. Confirm with the user before
every commit and every push. Conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
`chore:`).
