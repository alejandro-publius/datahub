---
description: "SQL Profiling in DataHub collects table-level and column-level statistics for relational sources during ingestion."
---

# SQL Profiling

SQL Profiling collects table level and column level statistics.
The SQL-based profiler does not run alone, but rather can be enabled for other SQL-based sources.
Enabling profiling will slow down ingestion runs.

:::caution

Running profiling against many tables or over many rows can run up significant costs.
While we've done our best to limit the expensiveness of the queries the profiler runs, you
should be prudent about the set of tables profiling is enabled on or the frequency
of the profiling runs.

:::

## Capabilities

Extracts:

- Row and column counts for each table
- For each column, if applicable:
  - null counts and proportions
  - distinct counts and proportions
  - minimum, maximum, mean, median, standard deviation, some quantile values
  - histograms or frequencies of unique values

## Supported Sources

{{ inline /docs/generated/ingestion/sql_profiling_support_table.md.snippet }}

## Profiler Implementation

DataHub uses a SQLAlchemy-based profiler for all SQL sources. It runs profiling queries directly against your SQL source's existing SQLAlchemy connection and emits the table- and column-level statistics listed under [Capabilities](#capabilities). No additional dependencies are required beyond the SQL connector itself.

No configuration is required to use it — any SQL source with profiling enabled will use the SQLAlchemy profiler automatically:

```yaml
source:
  config:
    profiling:
      enabled: true
```

:::note

The legacy Great Expectations profiler (`profiling.method: ge`) has been removed. SQLAlchemy is now the only SQL profiler; the `profiling.method` option no longer has any effect and can be dropped from recipes.

:::

## Profiling connection isolation level

For MySQL and Postgres, the profiler runs each profiling `SELECT` under `AUTOCOMMIT` instead of holding one implicit transaction open across every profiling query for a table. The previous behavior pinned InnoDB read views (MySQL) and held Postgres idle-in-transaction, which blocked `VACUUM`. `AUTOCOMMIT` makes each statement self-contained and avoids the long-lived transaction.

### Cross-snapshot skew

Under `AUTOCOMMIT`, `min`, `max`, `COUNT(*)`, `uniqueCount`, quantiles, histograms, and sample values each come from a different snapshot, so a single profile can be internally inconsistent on a concurrently-written table (for example `uniqueCount` > `rowCount`). This is an accepted trade-off — analytical profiling tolerates minor inconsistency, and the long-transaction alternative is worse for the database. The profiler already clamps derived ratios (`nullProportion`, `uniqueProportion` via `min(1, ...)`) to prevent nonsensical values; raw counts are emitted as-is and may be inconsistent on busy tables.

### Turning AUTOCOMMIT off

Two knobs restore the prior per-table transactional behavior; both are no-ops on platforms whose adapter does not opt in (Snowflake, BigQuery, Athena, Trino, ClickHouse — their adapters override `setup_profiling` and create session-scoped temp resources that `AUTOCOMMIT` can corrupt):

- **Per source (recipe):** set `profiling.isolation_level: TRANSACTIONAL` on the source. A SQLAlchemy isolation level name is also accepted to force a specific level.
- **Fleet-wide (operator):** set `DATAHUB_PROFILING_FORCE_TRANSACTIONAL=true` on the ingestion executor. This boolean kill switch wins unconditionally — it discards the adapter default *and* any explicit recipe pin (including an unrelated one such as `isolation_level: READ COMMITTED` on Redshift), which is correct for an emergency override. Unrecognized values are warned about (via the `env_vars.py` logger) and treated as unset, so a typo cannot silently force transactional. The switch is read through `datahub.configuration.env_vars` so it is discoverable alongside every other DataHub env var.

Reverting to transactional behavior restores the pre-change profile output with no migration, state, or cache to clean up — the only difference is the isolation level on the profiling connection. The kill switch changes **subsequent** runs; it does not repair profiles already emitted, so a re-ingestion is still required after an incident.

### Failure modes

An invalid level name is fatal: it fails identically on every table, so it raises `ConfigurationError` (naming `profiling.isolation_level`) rather than degrading. `generate_profiles` submits every request to a `ThreadPoolExecutor` up front and the executor's `__exit__` waits for all submitted futures, so every table is attempted before the run unwinds.

A level the server rejects — e.g. a proxy or a restricted user disallowing the `AUTOCOMMIT` session setting (a raw DBAPI error that `execution_options` does not wrap) — is **not** fatal: the first refusal warns once, latches, and profiling continues without the level (under the pre-change transactional behavior). The run report carries `profiling_isolation_level_degraded` (bool) and `profiling_isolation_level_degraded_tables` (int) so the operator can see the scale; the warning itself fires exactly once because the refusal latches. A stale pooled connection hitting this path used to fail one table on master; making it fatal would kill the whole run and blame the operator's config, a resilience regression.
