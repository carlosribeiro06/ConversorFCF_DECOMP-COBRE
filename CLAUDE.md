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
- `NCOEF` must be computed from the formula
  `1 + n_uhes + n_utv * max_lag + n_sbm_gnl * n_estagios * n_patamares`, never inferred from the
  last non-zero coefficient — the GNL block is dimensioned by stage count but only lag 1 is
  populated.
- `codigos_uhes_jusante` is `int32` on disk but `idecomp` reads it as `float32`; a read-back oracle
  needs `.view(np.int32)`.
- Cost units: `Cobre = DECOMP x 1000` (DECOMP works in 10^3 R$), verified to 1 ulp on a matched
  pair.

## Git conventions

Work on a feature branch; never commit to `main`, never force-push. Confirm with the user before
every commit and every push. Conventional commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
`chore:`).
