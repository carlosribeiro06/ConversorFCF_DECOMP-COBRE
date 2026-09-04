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

Pending `ticket-011`, which defines the command-line surface. The console entry point is already
installed as `conversor-fcf`.

## Outputs

| Artifact | Description |
|----------|-------------|
| `mapcut.<rev>` | DECOMP cut-header binary, fixed 48020-byte records |
| `cortdeco.<rev>` | DECOMP cut binary, chained fixed-size records |
| ECO CSVs | The Cobre cuts exactly as read, before any transformation |
| Content CSVs | The records actually written into the two binaries, in readable form |
| `run_manifest.json` | Provenance: tool and library versions, inputs, settings snapshot, premises |

`<rev>` is the PMO revision passed on the command line, for example `rv0`.

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
| `conversion.hydro_codes_path` | Cobre-to-DECOMP plant code map | `decomp_hydro_codes.json` |
| `conversion.include_terminal_pool` | Include the 267-node terminal fan in the ECO CSVs | `false` |

## Structure

| Path | Contents |
|------|----------|
| `src/conversor_fcf/config.py` | `settings.json` loader; every key validated, no defaults substituted |
| `src/conversor_fcf/logging_setup.py` | Console and audit-file handlers, and the `log_step` timing idiom |
| `src/conversor_fcf/run_manifest.py` | Provenance record; `PREMISES` is the single source of truth for the v1 premises |
| `src/conversor_fcf/cobre/` | Readers for the Cobre JSON inputs and the FlatBuffers policy checkpoint |
| `src/conversor_fcf/mapping/` | Cobre-to-DECOMP mapping rules (codes, units, signs, submarkets) |
| `src/conversor_fcf/decomp/` | Native `mapcut` / `cortdeco` serializers |
| `src/conversor_fcf/reporting/` | ECO and content CSV emitters |
| `src/Cobre/` | Vendored generated FlatBuffers readers, verbatim and not linted |
| `src/Cobre/PROVENANCE.md` | Upstream schema, namespace, file identifier and the 12 module digests |

## Logs and auditing

Logging is audit-grade: a `rich` console handler for the operator and a rotating file handler for
the record, both configured from `settings.json`. The audit log records the start and end of every
significant step with elapsed time, and carries the WARNING lines that name the premises applied to
a given run.

## Known premises and limitations

Pending `ticket-014`, which documents the ten v1 premises by quoting `run_manifest.PREMISES` so
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
