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

## Known premises and limitations

Pending `ticket-014`, which documents the v1 premises by quoting `run_manifest.PREMISES` so
code and prose cannot drift.

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

Git conventions: work on a feature branch, never commit to `main`, never force-push, and **confirm
with the user before every commit and every push**. Use conventional commits (`feat:`, `fix:`,
`refactor:`, `test:`, `docs:`, `chore:`).

## License

MIT. See [LICENSE](LICENSE).
