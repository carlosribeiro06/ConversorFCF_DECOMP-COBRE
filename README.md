# conversor-fcf

Converts a Cobre case into a DECOMP-format `mapcut` / `cortdeco` pair, so DESSEM can couple to a
future cost function produced by Cobre. The converter reads the case's JSON inputs and its
FlatBuffers policy checkpoint, and writes the two DECOMP binaries alongside CSV mirrors of both
what it read and what it wrote.

## Installation

Requires Python `>=3.12` (developed against 3.14.4).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test,dev]"
```

## Usage

```bash
conversor-fcf /path/to/DEC_ONS_052026_RV0_VE_CONVERTIDO
```

That is the whole command for the common case. The PMO revision is **derived from the case
directory name** — which carries it upper case, as `..._RV0_...` — and normalized to the lower-case
form the DECOMP filenames use, giving `mapcut.rv0` and `cortdeco.rv0`.

| Option | Meaning |
|--------|---------|
| `--revision rv0` | Declare the revision explicitly. When the case name also carries one, the two **must agree**: a disagreement stops the run rather than writing `rv0` artifacts from an `rv1` case. Required when the name carries none. |
| `--output DIR` | Override the output root. |
| `--settings FILE` | Settings file to read (default `settings.json`). |
| `--include-terminal-pool` / `--no-include-terminal-pool` | Override `conversion.include_terminal_pool`. Unset leaves the settings value alone. |
| `--force` | Replace existing artifacts. Without it, an existing `mapcut.<rev>`, `cortdeco.<rev>` or `run_manifest.json` stops the run **before any work**, naming every file you must move. |
| `--version` | Print the version and exit. |

Outputs land in `<case>/output/decomp_fcf/` by default, so the artifacts stay with the case that
produced them. A relative `output.directory` in `settings.json` resolves against the **case**, not
the working directory; an absolute one is used as given. The audit log follows the same rule against
the output root, so a run's log sits with the artifacts it describes.

### Exit codes

An official wrapper can branch on the failure class without parsing messages. Every failure is also
written to the audit log and to stderr.

| Code | Meaning |
|------|---------|
| `0` | Success |
| `2` | Usage error (argparse: unknown flag, missing argument, mutually exclusive flags) |
| `3` | `ConfigError` — the settings file is missing, malformed, or a key is absent or mistyped |
| `4` | `InputReadError` / `PolicyFormatError` — a Cobre input or the policy checkpoint could not be read |
| `5` | `MappingError` — a Cobre-to-DECOMP mapping rule refused its input |
| `6` | `LayoutError` / `ReadError` — a DECOMP binary failed its own layout invariant, or a read-back found the file on disk did not match what its own layout declares |
| `7` | `PathError` — the revision or an output path could not be resolved, the case is not a converted Cobre case, or an artifact already exists |

## Outputs

| Artifact | Description |
|----------|-------------|
| `mapcut.<rev>` | DECOMP cut-header binary, fixed 48020-byte records |
| `cortdeco.<rev>` | DECOMP cut binary, chained fixed-size records |
| `eco/eco_cuts_pool_<pool>.csv[.gz]` | The Cobre cuts exactly as read, before any transformation |
| `content/mapcut_content.csv` | Every `mapcut` record, read back from the written binary, long format |
| `content/cortdeco_content.csv` | Every `cortdeco` cut, read back from the written binary, one row per record |
| `run_manifest.json` | Provenance: tool and library versions, inputs, settings snapshot, premises, `status` (`"ok"`/`"failed"`) and, on failure, the `failed_step` label |

The content CSVs are produced by **reading the two binaries back**, not by re-serializing the records
still held in memory, so they witness what actually reached disk rather than what the writer was
given. `run_manifest.json` is written on both the success and the failure path (a failed run still
leaves an audit-grade record naming which step raised), and both it and the two binaries are written
atomically: `.partial`, fsynced, then renamed into place.

After writing both binaries the converter **cross-checks the pair**: they must agree on
`numero_cortes` and on the cut-head table, which are derived from one source, so a disagreement means
a wiring defect rather than bad input. A disagreeing pair must not stand as the converted result, so
both files are moved aside to `mapcut.<rev>.rejected` and `cortdeco.<rev>.rejected` and the run exits
`6`. Nothing is published under the names DECOMP reads, and the bytes are kept for diagnosis rather
than deleted; `run_manifest.json` records `status: "failed"` with `failed_step: "cross-check the
pair"`. A failure in a **later** step — writing the content CSVs, say — leaves the pair published,
because by then the cross-check has already validated it.

The window that triggers this is any failure from the start of the `mapcut` write until the
cross-check passes, not just the cross-check itself — a `--force` re-run that fails while writing
`cortdeco` also rejects both files, including a stale `cortdeco` left over from a previous run, which
would otherwise sit beside the fresh `mapcut` as an uncross-checked mixed pair. One consequence of
that wider window: a `--force` re-run that fails **inside** the `mapcut` write itself, before it
replaces anything, also moves the *previous, already-validated* pair aside to `.rejected` — this is
harmless (the bytes are preserved, not deleted, and `--force` already means "about to replace this
pair"), but it means a failed `--force` re-run always leaves the previous outputs at `.rejected`
rather than at their published names, even when that previous pair was perfectly good.

`<rev>` is the PMO revision, for example `rv0`, derived from the case directory name or declared with `--revision`. It appears **only** in the two binary filenames, because those are the names DECOMP reads.

## Configuration

All paths and tunables live in `settings.json` at the repository root. Every key is required —
a missing or mistyped key stops the run with a message naming the offending dotted key.

| Key | Meaning | Default |
|-----|---------|---------|
| `logging.level_console` | Console verbosity | `INFO` |
| `logging.level_file` | Audit-log verbosity | `DEBUG` |
| `logging.file_path` | Audit-log destination | `logs/conversor-fcf.log` |
| `logging.max_bytes` | Rotation threshold | `10485760` |
| `logging.backup_count` | Rotated files kept | `5` |
| `output.directory` | Output root, relative to the case | `output/decomp_fcf` |
| `output.eco_subdirectory` | ECO CSV subdirectory | `eco` |
| `output.content_subdirectory` | Content CSV subdirectory | `content` |
| `conversion.hydro_codes_path` | Cobre-to-DECOMP plant code map, resolved like `--settings`'s own default: as given, against the working directory | `decomp_hydro_codes.json` |
| `conversion.include_terminal_pool` | Also read the 267-node terminal pool (006) and write its ECO CSV; it never reaches `mapcut`/`cortdeco` (premise P7) | `false` |

## Structure

| Path | Contents |
|------|----------|
| `src/conversor_fcf/config.py` | `settings.json` loader; every key validated, no defaults substituted |
| `src/conversor_fcf/logging_setup.py` | Console and audit-file handlers, and the `log_step` timing idiom |
| `src/conversor_fcf/pipeline.py` | `run_conversion`, the single seam behind the CLI: ingest, map, write, cross-check, report |
| `src/conversor_fcf/run_manifest.py` | Provenance record; `PREMISES` is the single source of truth for the v1 premises |
| `src/conversor_fcf/cobre/` | Readers for the Cobre JSON inputs and the FlatBuffers policy checkpoint |
| `src/conversor_fcf/mapping/` | Cobre-to-DECOMP mapping rules (codes, units, signs, submarkets) |
| `src/conversor_fcf/decomp/` | Native `mapcut` / `cortdeco` serializers, plus the native read-back reader |
| `src/conversor_fcf/reporting/` | ECO and content CSV emitters |
| `src/Cobre/` | Vendored generated FlatBuffers readers, verbatim and not linted |
| `src/Cobre/PROVENANCE.md` | Upstream schema, namespace, file identifier and the 12 module digests |

## Logs and auditing

Logging is audit-grade: a `rich` console handler for the operator and a rotating file handler for
the record, both configured from `settings.json`. The audit log records the start and end of every
significant step with elapsed time, and carries the WARNING lines that name the premises applied to
a given run.

## Known premises

The subsections below quote `run_manifest.PREMISES` verbatim, one heading per premise, each
inside a fenced block so nothing in it needs escaping. `tests/unit/test_readme_premises.py`
checks that every one of the 15 strings is still a substring of this file after whitespace
normalization, so this section and the manifest cannot silently disagree. A `ticket-NNN`
reference below points into `plans/conversor-fcf/`.

### P1

```
P1: mapcut record indices 4-17 emitted as zeros; WARNING logged; deferred to ticket-015
```

Enforced by: `assert_mapcut_layout`'s byte scan of records 4-17, run before `write_mapcut`'s
rename; `_physical_span_nonzero` in the read-back reader; a WARNING logged by `write_mapcut` on
every run.

> **Accepted DESSEM-compatibility risk.** Record index 4 (`max_storage_hm3`) and index 5
> (`min_storage_hm3`) carry real per-plant physical data in a genuine DECOMP file, and both are
> already provably derivable from the case's own `system/hydros.json`. This converter emits the
> whole span as zeros anyway. Whether DESSEM requires records 4-17 is unknown; populating them is
> `ticket-015`'s deferred follow-up.

### P2

```
P2: intercept and every coefficient divided by 1000 (DECOMP is in 10^3 R$); verified to 1 ulp on
a matched pair; not configurable
```

Enforced by: `to_decomp_cost` / `to_decomp_costs`, `DECOMP_COST_DIVISOR = 1000.0`; witnessed at
full scale by `tests/integration/test_division_witness.py`.

### P3

```
P3: n_utv = 0, so no mapcut regs 7/8 and NCOEF omits the pi_qdefp block; Cobre carried no
HydroTransitBucket state
```

Enforced by: `assert_no_travel_time`, called from `assemble_mapcut_header`, raising `LayoutError`
before any byte is written.

### P4

```
P4: GNL coefficients negated (Cobre negative, DECOMP positive), isolated in one named, tested,
documented function. DORMANT while P14 holds: cortdeco emits its whole pi_gnl block as zeros, so
no real coefficient reaches this rule and it changes no byte of either output file. negate_gnl
and negate_gnl_array are retained, tested and reachable because ticket-016 needs them the moment
the reduction is settled
```

Enforced by: `negate_gnl`, `negate_gnl_array` — **DORMANT**: tested and reachable, but no real
coefficient reaches either while P14 holds.

### P5

```
P5: GNL disaggregated across the 3 load blocks weighted by each stage's own hours. The weights
are per-stage, not one triple for the study: stage 0 is 24/65/79, stages 1-4 are 15/64/89, stage
5 is 12/61/95 and stage 6 is 51/226/323 over 600 hours rather than 168. DORMANT while P14 holds:
with the pi_gnl block emitted as zeros there is no coefficient to spread across load blocks, so
this rule changes no byte of either output file. gnl_block_weights is retained and tested for
ticket-016
```

Enforced by: `gnl_block_weights` — **DORMANT**, for the same reason.

> **P4 and P5 stay DORMANT while P14 holds.** Neither is exercised by any real path, because the
> whole `pi_gnl` block is zero; `negate_gnl`, `negate_gnl_array` and `gnl_block_weights` stay in
> the code, tested, because `ticket-016` needs them the moment the reduction is settled. The run
> manifest still declares both rather than omitting them: a manifest asserting a premise with no
> effect on the output pair is an auditability defect, and so is one that silently drops a rule
> the code still carries — naming the dormancy is what distinguishes the two.

### P6

```
P6: submarket = bus_id + 1; bus 5 (IV) excluded; n_submercados = 5
```

Enforced by: `submarket_for_bus`, `SUBMARKET_COUNT = 5`, `EXCLUDED_BUS_ID = 5`.

### P7

```
P7: only trunk pools 0-5 converted; pool 6 (267-node terminal fan, 10 000 warm-start cuts)
excluded
```

Enforced by: `load_pools`, `cut_building_pools`, `select_eco_pools`.

### P8

```
P8: inflow-lag coefficients dropped with a counted audit report, never a silent truncation
```

Enforced by: `inflow_lag_drop_audit`, a counted audit report, never a silent truncation.

### P9

```
P9: discount rate duration-proportional (1+r)^(-cumulative_days/365.25); the day-count basis is
365.25, solved from the reference series and exact to 3.7e-13, not 365
```

Enforced by: `discount_factors`, `DAYS_PER_YEAR = 365.25`.

### P10

```
P10: all populated cuts emitted; is_active and active_cut_indices recorded as ECO columns, not
used as filters
```

Enforced by: the `is_active` / `in_active_cut_indices` ECO CSV columns;
`tests/unit/test_eco_csv.py::test_active_flags_are_recorded_and_never_filter_rows`.

### P11

```
P11: mapcut reg 10 emits zeros for parcela_custo_geracao_termica_minima,
parcela_custo_contrato_importacao_minimo, parcela_custo_contrato_exportacao_minimo,
geracao_termica_minima_sinalizada_gnl and geracao_termica_minima_gerada_gnl; they are DECOMP
operational quantities from the deck's own data, non-zero in the reference, and are not
derivable from a Cobre policy checkpoint, whose cut intercept already embeds the constant term.
Only taxa_desconto is filled.
```

Enforced by: `_cost_record` writes the five fields as zeros; a WARNING logged by `write_mapcut`.
**No byte-level check on the written file.**

### P12

```
P12: mapcut reg 9's trailing float64 block emitted as zeros. The reference populates it with the
GNL lag month's hours split across load blocks, three values per submarket summing to 730.5 =
365.25*24/12, one average month on the same day-count basis as P9. Two independent reasons, the
first decisive: the values are not derivable from a Cobre case, because that split is a monthly
load-block structure belonging to the DECOMP deck while Cobre supplies weekly stage blocks, and
no aggregation of this case's own blocks reproduces the reference triple (the closest, stages
0-5, is off by 0.0054 in proportion). Second, the block's axis is itself unsettled: the
populated width is ngnl*npat while idecomp's reader consumes ngnl*n_estagios. Populating this
block needs a DECOMP-side input, not better Cobre parsing.
```

Enforced by: `_gnl_record` writes the trailing block as zeros; a WARNING logged by `write_mapcut`;
`tests/integration/test_mapcut_assembly.py::test_the_reference_populates_reg_nine_where_premise_p12_emits_zeros`
proves the reference *does* populate what this converter zeros — not that zero is right. **No
byte-level check on the written file.**

### P13

```
P13: cortdeco holds numero_cortes + 1 records, and the extra one duplicates the last
cut-building node's last cut. The reference deck holds the next backward-pass cut there, written
but not yet exposed in the head table, which a checkpoint with a whole number of completed
iterations cannot supply. Nothing points to that record, since the chains only step backwards
from the heads. A duplicate is mathematically inert, because a repeated hyperplane adds nothing
to an FCF, while a zero-filled record would fabricate theta >= 0.
```

Enforced by: `cortdeco_record_count`, `assert_cortdeco_layout` (the extra record's pointer must
equal the last head).

### P14

```
P14: cortdeco's pi_gnl coefficients emitted as zeros. Only the values are zero: the block stays
dimensioned by the NCOEF formula and occupies its full n_sbm_gnl*n_estagios*n_patamares span,
and n_sbm_gnl, codigos_submercados_gnl (int32 on disk despite idecomp surfacing floats), NCOEF,
the record size and the file size are all unchanged. Reducing Cobre's anticipated-thermal ring
positions to DECOMP's (submarket, stage, block) address is unresolved: ring positions sharing a
delivery month carry different coefficients, so they are distinct state variables rather than
copies, and the reference deck comes from an independent run whose GNL magnitudes span seven
orders of magnitude against the oracle's 15% band and therefore cannot arbitrate between
candidate reductions. Every candidate writes a structurally valid file that DESSEM reads without
complaint, so a wrong one would corrupt the cut's GNL slope invisibly - unlike P1, P11 and P12,
whose divergence is a declared zero. Deferred to ticket-016; P4 and P5 are DORMANT while this
premise holds
```

Enforced by: `zeroed_gnl_block`, `_assert_gnl_input_is_zero`, `_normalize_signed_zero`, and
`assert_gnl_span_is_zero` at two independent call sites (`write_cortdeco` and `cross_check_pair`).

> **P14 is different in kind from P1, P11 and P12.** Those three are declared zeros an auditor
> can see and compare against the reference. P14's zeros hide an unresolved question: every
> candidate reduction of Cobre's anticipated-thermal ring positions to DECOMP's
> `(submarket, stage, block)` address writes a structurally valid file that DESSEM reads without
> complaint, so a wrong one would corrupt the cut's GNL slope invisibly. The full evidence record
> lives in
> [`ticket-016`](plans/conversor-fcf/epic-08-documentation-deferred/ticket-016-place-gnl-coefficients.md).

### P15

```
P15: mapcut reg 9's lag_meses_gnl emitted as 2 for every GNL plant. Evidenced only for the
reference case's own lead_time_hours of 1608.0 (1608.0/730.5 = 2.2012): floor and round both map
that single known point to 2, and the two formulas first diverge at 2.5 (any fractional part of
0.5 or more), so one evidenced point cannot decide between them. Guessing either would repeat
the error premise P14 exists to avoid. assert_gnl_lead_time_is_evidenced refuses any GNL plant
whose lead_time_hours differs from 1608.0 rather than guess at an unevidenced case.
```

Enforced by: `assert_gnl_lead_time_is_evidenced`, `EVIDENCED_GNL_LEAD_TIME_HOURS = 1608.0`,
`EVIDENCED_GNL_LAG_MESES = 2`; refuses any other lead time.

> **P15 fails closed rather than guessing.** `1608.0 / 730.5 = 2.2012` against a declared
> `lag_meses_gnl` of 2: `floor` and `round` both fit that single evidenced point, and the two
> formulas first diverge at 2.5 (any fractional part of 0.5 or more), so one evidenced point
> cannot decide between them. Picking either would repeat the exact error premise P14 exists to
> avoid, so `assert_gnl_lead_time_is_evidenced` refuses any GNL plant whose `lead_time_hours`
> differs from 1608.0 instead.

## Known limitations

Three properties with no known correction, stated as limitations rather than as bugs with a
pending fix:

- **The CSV writes are not atomic.** `eco_csv.write_eco_csv`, `write_eco_csv_gzip`,
  `content_csv.write_mapcut_content_csv` and `write_cortdeco_content_csv` all open their
  destination with `"w"`, truncating in place, so a crash mid-write leaves a short CSV at its
  final name with no length declaration by which a reader could detect the truncation. In a
  default run over the reference case that is **eight** files — six ECO CSVs (pools 000-005, one
  per converted trunk pool, so the count is case-dependent) and the two content CSVs — and
  `--include-terminal-pool` adds a ninth, gzipped. Both binaries and `run_manifest.json` are
  written atomically (`.partial`, fsync, `os.replace`, directory fsync), the manifest
  deliberately so, on the argument that a truncated audit record is unreadable. The
  inconsistency spans all eight, so fixing only the newer pair would be the wrong shape; this is
  a policy decision the project has not taken.
- **`codigos_submercados_gnl` holds one entry per anticipated thermal, not per distinct
  submarket.** `pipeline.assemble_mapcut_header` builds it as
  `tuple(submarket_for_bus(t.bus_id) for t in gnl)`. This case has two GNL plants in two distinct
  submarkets, so nothing is duplicated; two plants in one submarket would duplicate a submarket
  declaration and inflate `NCOEF`'s GNL term by a whole `n_estagios * n_patamares` block. No deck
  seen so far exhibits it, and DECOMP's semantics for a duplicated declaration are unknown, so a
  de-duplication could itself be the wrong fix.
- **`cost_scale_factor` is decoded and never asserted.** `policy_reader` decodes it onto both
  `PolicyManifest` and `StageCutPool` (1,000,000.0 in this case), and no code path consults it.
  Premise P2's divisor is a fixed, non-configurable 1000.0, established empirically to 1 ulp on a
  matched pair — *not* derived from `cost_scale_factor`. The two are different quantities:
  Cobre's internal cost scaling and DECOMP's 10^3 R$ unit. A case whose checkpoint declared a
  different `cost_scale_factor` would be converted with the same divisor and nothing would
  notice.

## Development

```bash
.venv/bin/ruff check
.venv/bin/ruff format --check .
.venv/bin/mypy --strict src/conversor_fcf tests
.venv/bin/pytest --cov=conversor_fcf --cov-report=term-missing
```

All four are expected to be clean on a fresh checkout. The vendored FlatBuffers readers under
`src/Cobre/` are generated code: they are excluded from `ruff` and `mypy` rather than edited to satisfy
them.

### Running against the reference artifacts

Eleven modules under `tests/` skip by name when the reference Cobre case and the two reference DECOMP
binaries (`mapcut.rv0`, `cortdeco.rv0`) are absent — safe per test, but `pytest`'s own exit status does
not distinguish "every reference-gated test skipped" from "ran and passed". `tests/conftest.py` adds a
session-level check on top of that, unrelated to the converter CLI's own exit code `6` for
`LayoutError`/`ReadError` documented above.

| Variable | Meaning |
|----------|---------|
| `CONVERSOR_FCF_REFERENCE_ROOT` | Directory holding the reference case and the two reference binaries. `~` expanded, then resolved. Defaults to `/home/carlosribeiro/git`. |
| `CONVERSOR_FCF_ALLOW_MISSING_REFERENCE` | `1` to accept an unverified session and keep exit `0`. Never silences the terminal summary — only the exit status is negotiable. |

When any of the three artifacts is absent at the resolved root — or the root is overridden while
those eleven modules still hardcode the default one (`ticket-022`) — `pytest -q` prints a summary
naming the resolved root, every absent artifact and both variables above, then exits **6**:

```bash
$ CONVERSOR_FCF_REFERENCE_ROOT=/srv/decks pytest -q
...
1 reference artifact(s) absent at /srv/decks: /srv/decks/cortdeco.rv0.
This session verified nothing reference-dependent: every reference-gated test skipped, and
pytest's own exit status does not distinguish that from having run.
Set CONVERSOR_FCF_ALLOW_MISSING_REFERENCE=1 to accept this and keep exit 0, or point
CONVERSOR_FCF_REFERENCE_ROOT at a directory holding all three artifacts.
$ echo $?
6
```

Git conventions: work on a feature branch, never commit to `main`, never force-push, and **confirm
with the user before every commit and every push**. Use conventional commits (`feat:`, `fix:`,
`refactor:`, `test:`, `docs:`, `chore:`).

## Verifying a conversion

Four independent checks bound what a converted pair can be trusted to mean. None of them proves
byte-for-byte correctness: this Cobre case completed 48 iterations while the CEPEL reference deck
completed 73, so byte comparison is impossible by construction. Acceptance here is **structural
and range conformance, never byte equality**.

- **`tests/integration/test_idecomp_cortdeco_oracle.py`** decodes the emitted `cortdeco` through
  `idecomp.decomp.cortdeco.Cortdeco`. Every `Cortdeco.read` argument comes from the `Mapcut`
  object `idecomp` itself read back, never from this project's own `decomp/reader.py`.
  ```bash
  .venv/bin/pytest tests/integration/test_idecomp_cortdeco_oracle.py -v
  ```
  Proves the file is readable by a third party's decoder with the layout this project believes
  it has. It cannot say anything about whether the *values* are right: it still reads the same
  file this project wrote.
- **`tests/integration/test_range_envelopes.py`** bounds every emitted `rhs` and `pi_varm`
  coefficient by the CEPEL reference deck's own magnitude envelopes.
  ```bash
  .venv/bin/pytest tests/integration/test_range_envelopes.py -v
  ```
  An executed mutation shows the `rhs` envelope catches a missed ÷1000 (multiplying the emitted
  `rhs` by 1000 escapes its upper bound) but not a doubled one (dividing `rhs` by 1000 again
  stays inside); the `pi_varm` envelope catches **neither** direction on this case.
- **`tests/integration/test_division_witness.py`** is the only end-to-end witness of premise P2
  at full scale, and only in one direction.
  ```bash
  .venv/bin/pytest tests/integration/test_division_witness.py -v
  ```
  `content == eco / 1000.0` exactly, over 288 matched records and 48,672 `pi_varm` comparisons.
  The inverted `content * 1000.0 == eco` fails for 643 of those 48,672, because a float64
  division is not exactly invertible.
- **`pipeline.cross_check_pair`** runs in-pipeline, on every conversion, not as a separate opt-in
  check.
  ```bash
  .venv/bin/pytest tests/unit/test_pipeline.py -v
  ```
  Checks that `numero_cortes` and the head table agree across the written pair, plus
  `assert_gnl_span_is_zero` on independently re-derived offsets over the *published* file. It
  cannot say that zero is the right `pi_gnl` value; that is premise P14's open question.

The first three checks are reference-gated: absent the reference case and the two reference
binaries, they skip rather than fail — see **Development → Running against the reference
artifacts** above for `CONVERSOR_FCF_REFERENCE_ROOT`, `CONVERSOR_FCF_ALLOW_MISSING_REFERENCE` and
exit code `6`. The fourth, `tests/unit/test_pipeline.py`, runs unconditionally against a
synthetic pair, since `cross_check_pair`'s own logic needs no real Cobre case to exercise.
`ticket-022`, the sweep converting the eleven modules that still hardcode the default reference
root, remains open and unimplemented.

## License

MIT. See [LICENSE](LICENSE).
