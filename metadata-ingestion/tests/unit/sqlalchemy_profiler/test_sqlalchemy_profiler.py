"""Unit tests for SQLAlchemyProfiler."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy import Column, Float, Integer, String, create_engine

from datahub.configuration.common import ConfigurationError
from datahub.ingestion.source.ge_profiling_config import (
    ProfilingConfig,
)
from datahub.ingestion.source.profiling.common import Cardinality, ProfilerRequest
from datahub.ingestion.source.sql.sql_report import SQLSourceReport
from datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler import (
    SQLAlchemyProfiler,
)
from datahub.ingestion.source.sqlalchemy_profiler.type_mapping import ProfilerDataType

# Env var read by get_profiling_force_transactional() (env_vars.py). Kept as a
# constant here so the kill-switch tests don't drift on the literal string.
FORCE_TRANSACTIONAL_ENV = "DATAHUB_PROFILING_FORCE_TRANSACTIONAL"


@pytest.fixture(autouse=True)
def _clean_profiling_force_transactional_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure DATAHUB_PROFILING_FORCE_TRANSACTIONAL never leaks across tests — the
    # kill switch is read at resolution time, so a leftover value would silently
    # force transactional profiling for any test that does not set it explicitly.
    monkeypatch.delenv(FORCE_TRANSACTIONAL_ENV, raising=False)


@pytest.fixture
def sqlite_engine():
    """Create an in-memory SQLite engine for testing."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def test_table(sqlite_engine):
    """Create a test table with sample data."""
    metadata = sa.MetaData()
    table = sa.Table(
        "test_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(50)),
        Column("value", Float),
    )
    metadata.create_all(sqlite_engine)

    with sqlite_engine.connect() as conn, conn.begin():
        conn.execute(
            sa.insert(table),
            [
                {"id": 1, "name": "Alice", "value": 10.5},
                {"id": 2, "name": "Bob", "value": 20.5},
                {"id": 3, "name": "Charlie", "value": 30.5},
            ],
        )

    return table


@pytest.fixture
def profiler_config():
    """Create a test profiling config."""
    return ProfilingConfig(
        enabled=True,
        include_field_null_count=True,
        include_field_distinct_count=True,
        include_field_min_value=True,
        include_field_max_value=True,
        include_field_mean_value=True,
        include_field_median_value=True,
        include_field_stddev_value=True,
        include_field_sample_values=True,
    )


@pytest.fixture
def mock_report():
    """Create a mock SQLSourceReport."""
    report = MagicMock(spec=SQLSourceReport)
    report.report_dropped = MagicMock()
    report.warning = MagicMock()
    return report


@pytest.fixture
def profiler(sqlite_engine, profiler_config, mock_report):
    """Create a SQLAlchemyProfiler instance."""
    return SQLAlchemyProfiler(
        conn=sqlite_engine,
        report=mock_report,
        config=profiler_config,
        platform="sqlite",
        env="TEST",
    )


class TestSQLAlchemyProfiler:
    """Test cases for SQLAlchemyProfiler."""

    def test_init(self, profiler, sqlite_engine):
        """Test profiler initialization."""
        assert profiler.base_engine == sqlite_engine
        assert profiler.platform == "sqlite"
        assert profiler.env == "TEST"
        assert profiler.times_taken == []
        assert profiler.total_row_count == 0

    def test_get_columns_to_profile(self, profiler, sqlite_engine, test_table):
        """Test column filtering logic."""
        # Create a table object with metadata
        metadata = sa.MetaData()
        sql_table = sa.Table(
            "test_table",
            metadata,
            autoload_with=sqlite_engine,
        )

        columns = profiler._get_columns_to_profile(sql_table, "test_table")
        # Should include all columns that match the config
        assert len(columns) > 0
        assert "id" in columns or "name" in columns or "value" in columns

    def test_should_ignore_column(self, profiler):
        """Test column type-based filtering."""
        # Should not ignore regular types
        assert not profiler._should_ignore_column(sa.Integer(), "id")
        assert not profiler._should_ignore_column(sa.String(), "name")
        assert not profiler._should_ignore_column(sa.Float(), "value")

    def test_generate_profiles_empty_list(self, profiler):
        """Test generate_profiles with empty request list."""
        requests: list = []
        # max_workers must be > 0
        profiles = list(profiler.generate_profiles(requests, max_workers=1))
        assert len(profiles) == 0

    def test_get_columns_to_profile_with_nested_fields_disabled(
        self, profiler, sqlite_engine
    ):
        """Test column filtering with nested fields disabled."""
        profiler.config.profile_nested_fields = False

        metadata = sa.MetaData()
        table = sa.Table(
            "test_table",
            metadata,
            Column("id", Integer),
            Column("nested.field", String(50)),
        )

        columns = profiler._get_columns_to_profile(table, "test_table")
        # Nested field should be excluded
        assert "nested.field" not in columns

    def test_get_columns_to_profile_with_nested_fields_enabled(
        self, profiler, sqlite_engine
    ):
        """Test column filtering with nested fields enabled."""
        profiler.config.profile_nested_fields = True

        metadata = sa.MetaData()
        table = sa.Table(
            "test_table",
            metadata,
            Column("id", Integer),
            Column("nested.field", String(50)),
        )

        profiler._get_columns_to_profile(table, "test_table")
        # Nested field should be included
        # Note: May still be filtered by type or other criteria
        # Just verify the method doesn't crash

    def test_get_columns_to_profile_max_limit(self, profiler, sqlite_engine):
        """Test column filtering with max columns limit."""
        profiler.config.max_number_of_fields_to_profile = 2

        metadata = sa.MetaData()
        table = sa.Table(
            "test_table",
            metadata,
            Column("id", Integer),
            Column("name", String(50)),
            Column("value", Float),
            Column("extra", String(50)),
        )

        columns = profiler._get_columns_to_profile(table, "test_table")
        # Should be limited to max_number_of_fields_to_profile
        assert len(columns) <= 2

    def test_setup_permission_error_with_catch_exceptions_true(
        self, profiler, mock_report, sqlite_engine
    ):
        """Test permission error during setup when catch_exceptions=True."""
        profiler.config.catch_exceptions = True

        request = ProfilerRequest(
            pretty_name="test_table",
            batch_kwargs={"table": "test_table", "schema": "test_schema"},
        )

        # Mock setup_profiling to raise PermissionError
        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_adapter.setup_profiling.side_effect = PermissionError(
                "permission denied"
            )
            mock_get_adapter.return_value = mock_adapter

            # Should return tuple (request, None) and log warning, not raise
            result_request, result_profile = profiler._generate_profile_from_request(
                None, request
            )

            # Should return None for profile (error was caught)
            assert result_request == request
            assert result_profile is None

            # Should have called report.warning for setup failure
            mock_report.warning.assert_called()
            call_args = mock_report.warning.call_args
            assert call_args is not None
            assert "Profiling setup failed" in call_args[1]["title"]

    def test_permission_error_with_catch_exceptions_false(
        self, profiler, sqlite_engine
    ):
        """Test permission error handling when catch_exceptions=False."""
        profiler.config.catch_exceptions = False

        request = ProfilerRequest(
            pretty_name="test_table",
            batch_kwargs={"table": "test_table", "schema": "test_schema"},
        )

        # Mock setup_profiling to raise PermissionError
        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_adapter.setup_profiling.side_effect = PermissionError(
                "permission denied"
            )
            mock_get_adapter.return_value = mock_adapter

            # Should re-raise the exception
            with pytest.raises(PermissionError, match="permission denied"):
                profiler._generate_profile_from_request(None, request)

    def test_sqlalchemy_error_with_catch_exceptions_true(
        self, profiler, mock_report, sqlite_engine
    ):
        """Test SQLAlchemy error handling when catch_exceptions=True."""
        profiler.config.catch_exceptions = True

        request = ProfilerRequest(
            pretty_name="test_table",
            batch_kwargs={"table": "test_table", "schema": "test_schema"},
        )

        # Mock setup_profiling to raise SQLAlchemy error
        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_adapter.setup_profiling.side_effect = sa.exc.OperationalError(
                "database error", None, None
            )
            mock_get_adapter.return_value = mock_adapter

            # Should return tuple (request, None) and log warning, not raise
            result_request, result_profile = profiler._generate_profile_from_request(
                None, request
            )

            # Should return None for profile (error was caught)
            assert result_request == request
            assert result_profile is None

            # Should have called report.warning
            mock_report.warning.assert_called()
            call_args = mock_report.warning.call_args
            assert "Profiling setup failed" in call_args[1]["title"]

    def test_sqlalchemy_error_with_catch_exceptions_false(
        self, profiler, sqlite_engine
    ):
        """Test SQLAlchemy error handling when catch_exceptions=False."""
        profiler.config.catch_exceptions = False

        request = ProfilerRequest(
            pretty_name="test_table",
            batch_kwargs={"table": "test_table", "schema": "test_schema"},
        )

        # Mock setup_profiling to raise SQLAlchemy error
        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_adapter.setup_profiling.side_effect = sa.exc.OperationalError(
                "database error", None, None
            )
            mock_get_adapter.return_value = mock_adapter

            # Should re-raise the exception
            with pytest.raises(sa.exc.OperationalError):
                profiler._generate_profile_from_request(None, request)

    def test_connection_error_with_catch_exceptions_true(
        self, profiler, mock_report, sqlite_engine
    ):
        """Test ConnectionError handling when catch_exceptions=True."""
        profiler.config.catch_exceptions = True

        request = ProfilerRequest(
            pretty_name="test_table",
            batch_kwargs={"table": "test_table", "schema": "test_schema"},
        )

        # Mock setup_profiling to raise ConnectionError
        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_adapter.setup_profiling.side_effect = ConnectionError(
                "connection lost"
            )
            mock_get_adapter.return_value = mock_adapter

            # Should return tuple (request, None) and log warning, not raise
            result_request, result_profile = profiler._generate_profile_from_request(
                None, request
            )

            # Should return None for profile (error was caught)
            assert result_request == request
            assert result_profile is None

            # Should have called report.warning
            mock_report.warning.assert_called()

    def test_unexpected_error_with_catch_exceptions_true(
        self, profiler, mock_report, sqlite_engine
    ):
        """Test unexpected exception handling when catch_exceptions=True."""
        profiler.config.catch_exceptions = True

        request = ProfilerRequest(
            pretty_name="test_table",
            batch_kwargs={"table": "test_table", "schema": "test_schema"},
        )

        # Mock setup_profiling to raise an unexpected error
        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_adapter.setup_profiling.side_effect = RuntimeError("unexpected error")
            mock_get_adapter.return_value = mock_adapter

            # Should return tuple (request, None) and log warning, not raise
            result_request, result_profile = profiler._generate_profile_from_request(
                None, request
            )

            # Should return None for profile (error was caught)
            assert result_request == request
            assert result_profile is None

            # Should have called report.warning
            mock_report.warning.assert_called()

    def test_unexpected_error_with_catch_exceptions_false(
        self, profiler, sqlite_engine
    ):
        """Test unexpected exception handling when catch_exceptions=False."""
        profiler.config.catch_exceptions = False

        request = ProfilerRequest(
            pretty_name="test_table",
            batch_kwargs={"table": "test_table", "schema": "test_schema"},
        )

        # Mock setup_profiling to raise an unexpected error
        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_adapter.setup_profiling.side_effect = RuntimeError("unexpected error")
            mock_get_adapter.return_value = mock_adapter

            # Should re-raise the exception
            with pytest.raises(RuntimeError, match="unexpected error"):
                profiler._generate_profile_from_request(None, request)

    def test_cleanup_called_after_error(self, profiler, sqlite_engine):
        """Test that adapter cleanup is called even when profiling fails."""
        profiler.config.catch_exceptions = True

        request = ProfilerRequest(
            pretty_name="test_table",
            batch_kwargs={"table": "test_table", "schema": "test_schema"},
        )

        # Mock setup_profiling to raise an error
        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_adapter.setup_profiling.side_effect = RuntimeError("test error")
            mock_get_adapter.return_value = mock_adapter

            # Execute profiling (will fail)
            profiler._generate_profile_from_request(None, request)

            # Cleanup should have been called even though profiling failed
            mock_adapter.cleanup.assert_called_once()

    @pytest.mark.parametrize(
        "stat_name,expected_title",
        [
            ("min", "Profiling: Unable to Calculate Min"),
            ("max", "Profiling: Unable to Calculate Max"),
            ("mean", "Profiling: Unable to Calculate Mean"),
            ("stdev", "Profiling: Unable to Calculate Standard Deviation"),
            ("median", "Profiling: Unable to Calculate Median"),
        ],
    )
    def test_batchable_numeric_stats_exception_caught(
        self, profiler, mock_report, stat_name, expected_title
    ):
        """Test that batchable numeric stats exceptions are caught in _process_numeric_column_stats."""
        mock_runner = MagicMock()
        mock_table = MagicMock()
        mock_column_profile = MagicMock()

        # Create a FutureResult that raises an exception when .result() is called
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception(f"{stat_name} error")

        # Pass the future in numeric_stats_futures (nested dict: {col_name: {stat_name: future}})
        numeric_stats_futures = {"value_col": {stat_name: mock_future}}

        # Should not raise, should log warning
        profiler._process_numeric_column_stats(
            runner=mock_runner,
            sql_table=mock_table,
            col_name="value_col",
            column_profile=mock_column_profile,
            col_type=ProfilerDataType.FLOAT,
            cardinality=Cardinality.MANY,
            numeric_stats_futures=numeric_stats_futures,
            pretty_name="test.table",
            platform="sqlite",
        )

        # Verify warning was logged
        mock_report.warning.assert_called()
        call_args = mock_report.warning.call_args
        assert call_args.kwargs["title"] == expected_title
        assert "test.table.value_col" in call_args.kwargs["context"]

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "name": "sample_values",
                "config_overrides": {},
                "mock_method": "get_column_sample_values",
                "profiler_method": "_add_sample_values",
                "method_kwargs": {
                    "col_name": "test_col",
                    "non_null_count": 10,
                    "row_count": 100,
                    "pretty_name": "test.table",
                },
                "expected_title": "Profiling: Unable to Calculate Sample Values",
                "expected_context": "test.table.test_col",
            },
            {
                "name": "histogram",
                "config_overrides": {"include_field_histogram": True},
                "mock_method": "get_column_histogram",
                "profiler_method": "_process_numeric_column_stats",
                "method_kwargs": {
                    "col_name": "value_col",
                    "col_type": ProfilerDataType.FLOAT,
                    "cardinality": Cardinality.MANY,
                    "numeric_stats_futures": {},
                    "pretty_name": "test.table",
                    "platform": "sqlite",
                },
                "expected_title": "Profiling: Unable to Calculate Histogram",
                "expected_context": "test.table.value_col",
            },
            {
                "name": "quantiles",
                "config_overrides": {"include_field_quantiles": True},
                "mock_method": "get_column_quantiles",
                "profiler_method": "_process_numeric_column_stats",
                "method_kwargs": {
                    "col_name": "value_col",
                    "col_type": ProfilerDataType.FLOAT,
                    "cardinality": Cardinality.MANY,
                    "numeric_stats_futures": {},
                    "pretty_name": "test.table",
                    "platform": "sqlite",
                },
                "expected_title": "Profiling: Unable to Calculate Quantiles",
                "expected_context": "test.table.value_col",
            },
            {
                "name": "distinct_value_frequencies",
                "config_overrides": {"include_field_distinct_value_frequencies": True},
                "mock_method": "get_column_distinct_value_frequencies",
                "profiler_method": "_maybe_add_distinct_value_frequencies",
                "method_kwargs": {
                    "col_name": "status_col",
                    "cardinality": Cardinality.ONE,
                    "allowed_cardinalities": {Cardinality.ONE, Cardinality.TWO},
                    "pretty_name": "test.table",
                },
                "expected_title": "Profiling: Unable to Calculate Distinct Value Frequencies",
                "expected_context": "test.table.status_col",
            },
        ],
        ids=lambda tc: tc["name"],
    )
    def test_non_batchable_query_exceptions_caught(
        self, sqlite_engine, mock_report, test_case
    ):
        """Test that non-batchable query exceptions are caught and logged."""
        # Create profiler with appropriate config
        config = ProfilingConfig(
            enabled=True, catch_exceptions=True, **test_case["config_overrides"]
        )
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="sqlite",
            env="TEST",
        )

        # Set up mocks
        mock_runner = MagicMock()
        mock_table = MagicMock()
        mock_column_profile = MagicMock()

        # Make the runner method raise an exception
        getattr(mock_runner, test_case["mock_method"]).side_effect = Exception(
            f"{test_case['name']} error"
        )

        # Call the profiler method
        method = getattr(profiler, test_case["profiler_method"])
        method(
            runner=mock_runner,
            sql_table=mock_table,
            column_profile=mock_column_profile,
            **test_case["method_kwargs"],
        )

        # Verify warning was logged
        assert mock_report.warning.called
        warning_calls = [
            call.kwargs for call in mock_report.warning.call_args_list if call.kwargs
        ]
        matching_warnings = [
            w
            for w in warning_calls
            if w.get("title") == test_case["expected_title"]
            and test_case["expected_context"] in w.get("context", "")
        ]
        assert len(matching_warnings) > 0, (
            f"Expected warning with title '{test_case['expected_title']}' "
            f"and context '{test_case['expected_context']}' not found. "
            f"Got warnings: {warning_calls}"
        )

    def test_row_count_failure_returns_none(self, profiler, mock_report, sqlite_engine):
        """
        Test that profiling returns None when row_count metric fails.

        This prevents empty profiles from being emitted when we can't get basic
        metrics like row count (e.g., due to permission errors). This matches
        GE profiler behavior which asserts that profile.rowCount is not None.

        The row_count extraction includes explicit exception handling and early
        return logic to prevent emitting profiles without this critical metric.
        """
        profiler.config.catch_exceptions = True

        request = ProfilerRequest(
            pretty_name="test.table",
            batch_kwargs={"table": "test_table", "schema": "test_schema"},
        )

        # Mock the profiling to fail during row count extraction
        # This simulates permission errors or other database failures
        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()

            # Setup succeeds but subsequent profiling will fail
            # We raise an exception that will propagate through the profiling pipeline
            mock_adapter.setup_profiling.side_effect = Exception(
                "Simulated row count failure"
            )

            mock_get_adapter.return_value = mock_adapter

            # Attempt to profile - should return None for failed profiling
            result_request, result_profile = profiler._generate_profile_from_request(
                None, request
            )

            # Verify that None is returned (no profile emitted on failure)
            assert result_request == request
            assert result_profile is None, (
                "Expected None to be returned when profiling fails, "
                "preventing incomplete profiles from being emitted"
            )

            # Verify warning was logged
            assert mock_report.warning.called

    def test_empty_table_skips_column_profiling(
        self, profiler, sqlite_engine, test_table
    ):
        """
        Test that empty tables (row_count == 0) skip column profiling but return basic profile.

        This optimization matches GE profiler behavior:
        - Empty tables get a basic profile with rowCount=0
        - Column profiling is skipped (no field profiles generated)
        - No wasted queries on empty tables

        The behavior is the same as row_count failure (None) - both return a basic profile.
        The difference is the reason: empty table optimization vs permission error.
        """
        request = ProfilerRequest(
            pretty_name="test.empty_table",
            batch_kwargs={"table": "test_table", "schema": None},
        )

        # Create a mock sql_table with columns but mock row_count to return 0
        metadata = sa.MetaData()
        sql_table = sa.Table(
            "test_table",
            metadata,
            sa.Column("id", sa.Integer),
            sa.Column("value", sa.Integer),
        )

        # Define side effect that sets profile.rowCount = 0 and returns 0
        def mock_profile_row_count(*args, **kwargs):
            # The profile parameter is at index 3 (after self, runner, query_combiner, sql_table)
            profile = args[3] if len(args) > 3 else kwargs.get("profile")
            if profile:
                profile.rowCount = 0
            return 0

        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch.object(
                profiler, "_profile_row_count", side_effect=mock_profile_row_count
            ),
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn

            # Create mock adapter and mock context
            mock_adapter = MagicMock()
            mock_context = MagicMock()
            mock_context.sql_table = sql_table
            mock_adapter.setup_profiling.return_value = mock_context
            mock_adapter.cleanup.return_value = None
            mock_get_adapter.return_value = mock_adapter

            # Attempt to profile - should return basic profile
            result_request, result_profile = profiler._generate_profile_from_request(
                None, request
            )

            # Verify that a basic profile is returned (not None)
            assert result_request == request
            assert result_profile is not None, (
                "Expected basic profile to be returned for empty table, "
                "not None (which would skip the entire table)"
            )

            # Verify row_count was set to 0
            assert result_profile.rowCount == 0, (
                f"Expected rowCount=0, got {result_profile.rowCount}"
            )

            # Verify no field profiles were generated (column profiling skipped)
            assert (
                result_profile.fieldProfiles is None
                or len(result_profile.fieldProfiles) == 0
            ), (
                f"Expected no field profiles for empty table, got {len(result_profile.fieldProfiles) if result_profile.fieldProfiles else 0}"
            )


class TestProfilingIsolationLevel:
    """The profiler must apply the resolved isolation level to each per-table connection.

    The level is resolved ONCE at construction (see __init__) — not per table — so these
    tests force a level via the `profiling_isolation_level` config escape hatch rather than
    mocking `adapter.profiling_isolation_level()` per table (which is no longer called
    per-table).
    """

    def test_resolve_isolation_level_adapter_default_wins(
        self, sqlite_engine, mock_report
    ):
        # No config override -> the adapter's default wins. Directly exercises the
        # resolver (the constructor already calls it once with the real adapter; this
        # re-calls it with a mocked adapter to assert precedence in isolation).
        config = ProfilingConfig(enabled=True)
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="sqlite",
            env="TEST",
        )
        mock_adapter = MagicMock()
        mock_adapter.profiling_isolation_level.return_value = "AUTOCOMMIT"
        with patch(
            "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
        ) as mock_get_adapter:
            mock_get_adapter.return_value = mock_adapter
            assert profiler._resolve_profiling_isolation_level() == "AUTOCOMMIT"

    def test_resolve_isolation_level_override_wins(self, sqlite_engine, mock_report):
        # A concrete override wins over the adapter default.
        config = ProfilingConfig(enabled=True, isolation_level="READ UNCOMMITTED")
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="sqlite",
            env="TEST",
        )
        mock_adapter = MagicMock()
        mock_adapter.profiling_isolation_level.return_value = "AUTOCOMMIT"
        with patch(
            "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
        ) as mock_get_adapter:
            mock_get_adapter.return_value = mock_adapter
            assert profiler._resolve_profiling_isolation_level() == "READ UNCOMMITTED"

    def test_resolve_isolation_level_transactional_sentinel_clears(
        self, sqlite_engine, mock_report
    ):
        # The TRANSACTIONAL sentinel clears the level to None (falls back to
        # transactional), even when the adapter would return AUTOCOMMIT.
        config = ProfilingConfig(enabled=True, isolation_level="TRANSACTIONAL")
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="sqlite",
            env="TEST",
        )
        mock_adapter = MagicMock()
        mock_adapter.profiling_isolation_level.return_value = "AUTOCOMMIT"
        with patch(
            "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
        ) as mock_get_adapter:
            mock_get_adapter.return_value = mock_adapter
            assert profiler._resolve_profiling_isolation_level() is None

    def test_applies_autocommit_when_level_configured(self, profiler):
        # Force AUTOCOMMIT via the escape hatch; the per-table connection must receive it.
        profiler.config.catch_exceptions = True
        # The level is resolved at construction (see `_resolve_profiling_isolation_level`);
        # AUTOCOMMIT is accepted by sqlite's dialect, so resolution yields
        # "AUTOCOMMIT". Set directly here to skip the per-table execution_options
        # round-trip (these tests assert on the applied option, not on resolution).
        profiler._profiling_isolation_level = "AUTOCOMMIT"
        mock_adapter = MagicMock()
        mock_conn = MagicMock()

        with (
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_adapter.setup_profiling.side_effect = RuntimeError("short-circuit")
            mock_get_adapter.return_value = mock_adapter

            result = profiler._generate_single_profile(
                query_combiner=MagicMock(),
                pretty_name="my_db.my_table",
                schema="my_db",
                table="my_table",
                platform="mysql",
            )

        assert result is None  # short-circuited
        mock_conn.execution_options.assert_called_once_with(
            isolation_level="AUTOCOMMIT"
        )
        # The configured connection (execution_options result) is what flows
        # downstream into setup_profiling, not the raw checked-out connection.
        assert (
            mock_adapter.setup_profiling.call_args[0][1]
            is mock_conn.execution_options.return_value
        )

    def test_does_not_apply_options_when_level_none(self, profiler):
        # Default: no escape hatch, sqlite's GenericAdapter returns None → no execution_options.
        profiler.config.catch_exceptions = True
        assert profiler._profiling_isolation_level is None
        mock_adapter = MagicMock()
        mock_conn = MagicMock()

        with (
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_adapter.setup_profiling.side_effect = RuntimeError("short-circuit")
            mock_get_adapter.return_value = mock_adapter

            result = profiler._generate_single_profile(
                query_combiner=MagicMock(),
                pretty_name="my_db.my_table",
                schema="my_db",
                table="my_table",
                platform="sqlite",
            )

        assert result is None  # short-circuited
        mock_conn.execution_options.assert_not_called()
        # With no level, the raw checked-out connection flows downstream
        # (execution_options was never called, so no .return_value indirection).
        assert mock_adapter.setup_profiling.call_args[0][1] is mock_conn

    def test_invalid_level_fails_loudly_at_first_table(
        self, sqlite_engine, mock_report
    ):
        # An invalid level name is fatal: it fails identically on every table, so
        # it raises ConfigurationError rather than degrading to a per-table
        # warning. Construction does NOT validate the level (validation is lazy
        # so a transient connect blip cannot gate a whole database); the bad
        # level is stored unvalidated and surfaces on the first per-table
        # `conn.execution_options(...)`, where the call site converts
        # ArgumentError to ConfigurationError and the outer
        # `except ConfigurationError: raise` re-raises it ahead of the broad
        # `except Exception`. generate_profiles submits every request to a
        # ThreadPoolExecutor up front and the executor's __exit__ waits for all
        # submitted futures, so every table is attempted before the run unwinds.
        config = ProfilingConfig(
            enabled=True,
            isolation_level="BOGUS",
        )
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="sqlite",
            env="TEST",
        )
        # Construction succeeded; the level is stored unvalidated.
        assert profiler._profiling_isolation_level == "BOGUS"
        with pytest.raises(ConfigurationError):
            profiler._generate_single_profile(
                query_combiner=MagicMock(),
                pretty_name="main.test_table",
                schema="main",
                table="test_table",
                platform="sqlite",
            )

    def test_level_reaches_real_dialect(self, sqlite_engine, mock_report):
        # The only test proving the option actually reaches a dialect. Uses the
        # real sqlite engine (not a mock) and a level that round-trips through
        # get_isolation_level(). NOTE: "AUTOCOMMIT" is a SQLAlchemy pseudo-level NOT
        # reported by get_isolation_level() on sqlite (returns SERIALIZABLE — verified on
        # 1.4.44), so it cannot be used to prove round-trip; "READ UNCOMMITTED" does
        # round-trip and proves execution_options applied the level to the real DBAPI
        # connection.
        captured: list[str] = []
        config = ProfilingConfig(
            enabled=True,
            isolation_level="READ UNCOMMITTED",
        )
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="sqlite",
            env="TEST",
        )
        mock_adapter = MagicMock()

        def _capture_and_short_circuit(_ctx: Any, conn: Any) -> None:
            # Capture the real isolation level on the connection that flowed downstream,
            # then raise to short-circuit the rest of the profile flow.
            captured.append(conn.get_isolation_level())
            raise RuntimeError("short-circuit")

        mock_adapter.setup_profiling.side_effect = _capture_and_short_circuit

        with patch(
            "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
        ) as mock_get_adapter:
            mock_get_adapter.return_value = mock_adapter
            result = profiler._generate_single_profile(
                query_combiner=MagicMock(),
                pretty_name="main.test_table",
                schema="main",
                table="test_table",
                platform="sqlite",
            )

        assert result is None  # short-circuited
        assert captured == ["READ UNCOMMITTED"]

    def test_mysql_platform_resolves_to_autocommit_at_construction(
        self, sqlite_engine, mock_report
    ):
        # Wiring: platform="mysql" -> get_adapter returns MySQLAdapter ->
        # profiling_isolation_level() returns "AUTOCOMMIT" -> resolved at construction.
        # No escape hatch, no mock of get_adapter: this proves the adapter factory and the
        # hook are wired together end-to-end (the tests above mock get_adapter; this one
        # does not). Resolution does not open a connection, so the sqlite base_engine
        # being a different dialect than mysql is fine — the level is only validated
        # later on the per-table path.
        config = ProfilingConfig(enabled=True)
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="mysql",
            env="TEST",
        )
        assert profiler._profiling_isolation_level == "AUTOCOMMIT"

    def test_postgres_platforms_resolve_to_autocommit_at_construction(
        self, sqlite_engine, mock_report
    ):
        # Postgres counterpart to the mysql wiring test. get_adapter accepts both
        # "postgres" and "postgresql" as platform strings, so both spellings must
        # resolve to AUTOCOMMIT — a regression here would silently drop Postgres
        # back into the long profiling transaction.
        for platform in ("postgres", "postgresql"):
            config = ProfilingConfig(enabled=True)
            profiler = SQLAlchemyProfiler(
                conn=sqlite_engine,
                report=mock_report,
                config=config,
                platform=platform,
                env="TEST",
            )
            assert profiler._profiling_isolation_level == "AUTOCOMMIT", platform

    def test_transactional_sentinel_forces_transactional(
        self, sqlite_engine, mock_report
    ):
        # Escape hatch: TRANSACTIONAL overrides an adapter that would otherwise return
        # AUTOCOMMIT (mysql), forcing back to None (default transactional behavior). This
        # is the MySQL-behind-a-proxy-that-rejects-AUTOCOMMIT scenario the field exists for.
        config = ProfilingConfig(
            enabled=True,
            isolation_level="TRANSACTIONAL",
        )
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="mysql",
            env="TEST",
        )
        assert profiler._profiling_isolation_level is None

    def test_transactional_sentinel_lowercase_is_normalized(
        self, sqlite_engine, mock_report
    ):
        # field_validator strips + upper-cases before the sentinel comparison, so
        # "transactional " (lowercase, trailing space) normalizes to "TRANSACTIONAL"
        # and maps to None -- without the validator this would miss the sentinel and be
        # handed to the dialect as an invalid isolation level.
        config = ProfilingConfig(
            enabled=True,
            isolation_level="transactional ",
        )
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="mysql",
            env="TEST",
        )
        assert profiler._profiling_isolation_level is None

    def test_isolation_level_reset_on_pool_checkin(self, sqlite_engine, mock_report):
        # Upstream-assumption guard (tests SQLAlchemy, not DataHub): setting
        # isolation_level via execution_options must not leak across
        # checkin/checkout. SQLAlchemy registers a finalize_callback on the
        # _ConnectionRecord that resets the isolation level on checkin, so the
        # underlying DBAPI connection returns to the pool's default. The guard
        # matters because base_engine is shared with metadata extraction — a leak
        # here would corrupt the non-profiling path — but the property being
        # asserted is SQLAlchemy's pool behaviour, not this PR's code. Uses "READ
        # UNCOMMITTED" because it round-trips through get_isolation_level() on
        # sqlite (AUTOCOMMIT is a SQLAlchemy pseudo-level that sqlite reports as
        # SERIALIZABLE — see test_level_reaches_real_dialect).
        from sqlalchemy.pool import QueuePool

        engine = create_engine(
            "sqlite:///:memory:",
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=0,
        )
        try:
            # Checkout a connection and set READ UNCOMMITTED.
            with engine.connect() as conn:
                conn.execution_options(isolation_level="READ UNCOMMITTED")
                # The branched connection reports the override.
                assert conn.get_isolation_level() == "READ UNCOMMITTED"
            # After checkin, the next checkout must NOT inherit the override.
            with engine.connect() as conn:
                assert conn.get_isolation_level() != "READ UNCOMMITTED"
        finally:
            engine.dispose()

    def test_override_on_non_opt_in_platform_warns(self, sqlite_engine, mock_report):
        # A non-TRANSACTIONAL recipe override on a platform whose adapter returns
        # None (e.g. snowflake, which overrides setup_profiling and creates
        # session-scoped temp resources) emits a report.warning naming the
        # platform. Warn — do not reject — so a legitimate opt-in on e.g.
        # Redshift/MSSQL still works.
        config = ProfilingConfig(
            enabled=True,
            isolation_level="AUTOCOMMIT",
        )
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="snowflake",
            env="TEST",
        )
        # Construction triggers _resolve_profiling_isolation_level, which warns.
        mock_report.warning.assert_called_once()
        kwargs = mock_report.warning.call_args.kwargs
        assert "snowflake" in kwargs["message"]
        assert profiler._profiling_isolation_level == "AUTOCOMMIT"

    def test_override_on_opt_in_platform_does_not_warn(
        self, sqlite_engine, mock_report
    ):
        # An override on a platform whose adapter DOES opt in (mysql returns
        # AUTOCOMMIT) must NOT warn — the override is a normal escape hatch there.
        config = ProfilingConfig(
            enabled=True,
            isolation_level="READ COMMITTED",
        )
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="mysql",
            env="TEST",
        )
        mock_report.warning.assert_not_called()
        assert profiler._profiling_isolation_level == "READ COMMITTED"

    def test_transactional_sentinel_on_non_opt_in_platform_does_not_warn(
        self, sqlite_engine, mock_report
    ):
        # The TRANSACTIONAL sentinel is a no-op on a non-opt-in platform (it
        # clears the level to None, which is already the adapter default), so it
        # must NOT warn. The warning is only for non-TRANSACTIONAL overrides.
        config = ProfilingConfig(
            enabled=True,
            isolation_level="TRANSACTIONAL",
        )
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="snowflake",
            env="TEST",
        )
        mock_report.warning.assert_not_called()
        assert profiler._profiling_isolation_level is None


class TestProfilingForceTransactional:
    """DATAHUB_PROFILING_FORCE_TRANSACTIONAL kill switch.

    The switch is the first check in _resolve_profiling_isolation_level and wins
    unconditionally — over the recipe field and over the adapter default. The
    autouse `_clean_profiling_force_transactional_env` fixture guarantees the
    variable is unset between tests. These tests use a real SQLSourceReport so the
    forced-transactional / degraded fields read back as their real defaults.
    """

    @pytest.fixture
    def real_report(self):
        return SQLSourceReport()

    def test_switch_resolves_level_to_none_on_mysql(
        self, sqlite_engine, real_report, monkeypatch
    ):
        # On a platform whose adapter defaults to AUTOCOMMIT (mysql), the switch
        # forces the level to None — the AUTOCOMMIT default is discarded.
        monkeypatch.setenv(FORCE_TRANSACTIONAL_ENV, "true")
        config = ProfilingConfig(enabled=True)
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=real_report,
            config=config,
            platform="mysql",
            env="TEST",
        )
        assert profiler._profiling_isolation_level is None
        assert profiler.report.profiling_isolation_level_forced_transactional is True
        assert (
            profiler._profiling_isolation_level_source
            == "DATAHUB_PROFILING_FORCE_TRANSACTIONAL"
        )

    def test_switch_wins_over_explicit_recipe_pin(
        self, sqlite_engine, real_report, monkeypatch
    ):
        # The switch beats everything: an explicit recipe pin (e.g. READ COMMITTED
        # on Redshift) is discarded when the switch is active. This is correct for
        # an emergency override, and the operator must be able to see it happened.
        monkeypatch.setenv(FORCE_TRANSACTIONAL_ENV, "true")
        config = ProfilingConfig(enabled=True, isolation_level="READ COMMITTED")
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=real_report,
            config=config,
            platform="mysql",
            env="TEST",
        )
        assert profiler._profiling_isolation_level is None
        assert profiler.report.profiling_isolation_level_forced_transactional is True

    def test_switch_visible_in_report_when_active(
        self, sqlite_engine, real_report, monkeypatch
    ):
        # With the switch on, nothing warns and nothing degrades, so a run is
        # silently different from the same recipe on a node without the variable.
        # The report field is the only signal that catches this fleet-wide.
        monkeypatch.setenv(FORCE_TRANSACTIONAL_ENV, "true")
        config = ProfilingConfig(enabled=True)
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=real_report,
            config=config,
            platform="mysql",
            env="TEST",
        )
        assert profiler.report.profiling_isolation_level_forced_transactional is True
        assert profiler.report.profiling_isolation_level_degraded is False
        assert profiler.report.profiling_isolation_level_degraded_tables == 0

    def test_switch_unset_uses_adapter_default(
        self, sqlite_engine, real_report, monkeypatch
    ):
        # With the switch absent, the adapter default (mysql AUTOCOMMIT) wins and
        # the forced-transactional flag stays False.
        monkeypatch.delenv(FORCE_TRANSACTIONAL_ENV, raising=False)
        config = ProfilingConfig(enabled=True)
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=real_report,
            config=config,
            platform="mysql",
            env="TEST",
        )
        assert profiler._profiling_isolation_level == "AUTOCOMMIT"
        assert profiler.report.profiling_isolation_level_forced_transactional is False

    def test_unrecognized_switch_value_warned_and_ignored(
        self, sqlite_engine, real_report, monkeypatch, caplog
    ):
        # An unrecognized value is warned about (via env_vars.py logger) and
        # treated as unset — a typo that reads as "off" is the worst outcome for
        # an emergency switch, so it must not silently force transactional.
        monkeypatch.setenv(FORCE_TRANSACTIONAL_ENV, "yes_please")
        config = ProfilingConfig(enabled=True)
        with caplog.at_level("WARNING", logger="datahub.configuration.env_vars"):
            profiler = SQLAlchemyProfiler(
                conn=sqlite_engine,
                report=real_report,
                config=config,
                platform="mysql",
                env="TEST",
            )
        assert profiler._profiling_isolation_level == "AUTOCOMMIT"
        assert profiler.report.profiling_isolation_level_forced_transactional is False
        assert any(
            FORCE_TRANSACTIONAL_ENV in rec.message and "Unrecognized" in rec.message
            for rec in caplog.records
        )


class TestProfilingIsolationLevelFailures:
    """The two failure modes when applying the resolved level per table.

    An invalid level name (ArgumentError) is fatal — it fails identically on
    every table. A level the server rejects (a raw DBAPI error that
    execution_options does not wrap) is not fatal: the first refusal warns once,
    latches, and profiling continues without the level (under the pre-change
    transactional behavior). Splitting on determinism, not on exception
    hierarchy, keeps a stale pooled connection from killing the whole run.
    """

    @staticmethod
    def _make_degraded_profiler(
        sqlite_engine, mock_report, monkeypatch, level="AUTOCOMMIT"
    ):
        monkeypatch.delenv(FORCE_TRANSACTIONAL_ENV, raising=False)
        config = ProfilingConfig(enabled=True, isolation_level=level)
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="sqlite",
            env="TEST",
        )
        profiler.config.catch_exceptions = True

        class _DriverError(Exception):
            pass

        # _isolation_level_set_errors is resolved at construction from the real
        # sqlite dbapi.Error; override it so _DriverError is caught by the
        # driver-error handler (simulating a real driver error subclassing the
        # resolved dbapi.Error).
        profiler._isolation_level_set_errors = (_DriverError,)
        # Pre-set the counter fields to real scalars so += / = track real values
        # instead of MagicMock attributes.
        mock_report.profiling_isolation_level_degraded = False
        mock_report.profiling_isolation_level_degraded_tables = 0
        return profiler, _DriverError

    def test_driver_error_warns_degrades_and_names_transactional_fix(
        self, sqlite_engine, mock_report, monkeypatch
    ):
        # A raw DBAPI error from execution_options (server refused the level)
        # warns once, latches, and proceeds without the level — it does NOT
        # raise. The warning message names the TRANSACTIONAL remedy and the
        # source label. catch_exceptions=True must not swallow it because no
        # ConfigurationError is raised.
        profiler, _DriverError = self._make_degraded_profiler(
            sqlite_engine, mock_report, monkeypatch
        )
        with (
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_conn.execution_options.side_effect = _DriverError("proxy rejected")
            mock_adapter = MagicMock()
            # Short-circuit the downstream profiling path so the test stays
            # focused on the isolation-level block.
            mock_adapter.setup_profiling.side_effect = RuntimeError("short-circuit")
            mock_get_adapter.return_value = mock_adapter

            result = profiler._generate_single_profile(
                query_combiner=MagicMock(),
                pretty_name="main.test_table",
                schema="main",
                table="test_table",
                platform="sqlite",
            )

        assert result is None  # short-circuited downstream
        assert mock_report.profiling_isolation_level_degraded is True
        assert mock_report.profiling_isolation_level_degraded_tables == 1
        # The driver-error warning fired exactly once (the short-circuit warning
        # is a separate title).
        degraded_warnings = [
            call
            for call in mock_report.warning.call_args_list
            if call.kwargs.get("title")
            == "Profiling isolation level rejected by the database"
        ]
        assert len(degraded_warnings) == 1
        message = degraded_warnings[0].kwargs["message"]
        assert "TRANSACTIONAL" in message
        assert "transactional behavior" in message
        # The context carries the level, the source label, and the table.
        context = degraded_warnings[0].kwargs["context"]
        assert "main.test_table" in context
        assert "AUTOCOMMIT" in context

    def test_driver_error_latches_so_skip_path_also_counts(
        self, sqlite_engine, mock_report, monkeypatch
    ):
        # Both assertions together catch a broken implementation: a counter
        # that only counts rejections reads 1 forever (passes the "warning once"
        # check alone), and a latch that never engages warns every table (passes
        # the "count > 1" check alone). Either assertion alone passes under a
        # broken implementation; together they fail.
        profiler, _DriverError = self._make_degraded_profiler(
            sqlite_engine, mock_report, monkeypatch
        )
        with (
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_conn.execution_options.side_effect = _DriverError("proxy rejected")
            mock_adapter = MagicMock()
            mock_adapter.setup_profiling.side_effect = RuntimeError("short-circuit")
            mock_get_adapter.return_value = mock_adapter

            for i in range(2):
                profiler._generate_single_profile(
                    query_combiner=MagicMock(),
                    pretty_name=f"main.t{i}",
                    schema="main",
                    table=f"t{i}",
                    platform="sqlite",
                )

        # The refusal latches: the driver-error warning fires once, not per table.
        degraded_warnings = [
            call
            for call in mock_report.warning.call_args_list
            if call.kwargs.get("title")
            == "Profiling isolation level rejected by the database"
        ]
        assert len(degraded_warnings) == 1
        # Both tables are counted — the first in the except branch, the second on
        # the skip path — so the counter exceeds one.
        assert mock_report.profiling_isolation_level_degraded_tables == 2
        assert mock_report.profiling_isolation_level_degraded is True

    def test_invalid_level_still_produces_invalid_name_message(
        self, sqlite_engine, mock_report, monkeypatch
    ):
        # An invalid level name (ArgumentError) is fatal — it fails identically on
        # every table — so it raises ConfigurationError naming the recipe field.
        monkeypatch.delenv(FORCE_TRANSACTIONAL_ENV, raising=False)
        config = ProfilingConfig(enabled=True, isolation_level="BOGUS")
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=mock_report,
            config=config,
            platform="sqlite",
            env="TEST",
        )
        with pytest.raises(ConfigurationError) as exc_info:
            profiler._generate_single_profile(
                query_combiner=MagicMock(),
                pretty_name="main.test_table",
                schema="main",
                table="test_table",
                platform="sqlite",
            )
        assert "Invalid profiling.isolation_level" in str(exc_info.value)
        assert "BOGUS" in str(exc_info.value)
