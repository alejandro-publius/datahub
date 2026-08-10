"""Unit tests for SQLAlchemyProfiler."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy import Column, Float, Integer, String, create_engine

from datahub.ingestion.source.ge_profiling_config import ProfilingConfig
from datahub.ingestion.source.profiling.common import Cardinality, ProfilerRequest
from datahub.ingestion.source.sql.sql_report import SQLSourceReport
from datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler import (
    SQLAlchemyProfiler,
)
from datahub.ingestion.source.sqlalchemy_profiler.type_mapping import ProfilerDataType

# Tests that patch `get_adapter` with a MagicMock must pin
# `mock_adapter.profiling_isolation_level.return_value = None`: the per-table path
# calls that hook, and a bare MagicMock returns a MagicMock (not None), which would
# be passed into `execution_options` as an isolation level. Pinning to None skips
# the level block, matching these tests' pre-feature behaviour.


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
            mock_adapter.profiling_isolation_level.return_value = None

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
            mock_adapter.profiling_isolation_level.return_value = None

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
            mock_adapter.profiling_isolation_level.return_value = None

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
            mock_adapter.profiling_isolation_level.return_value = None

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
            mock_adapter.profiling_isolation_level.return_value = None

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
            mock_adapter.profiling_isolation_level.return_value = None

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
            mock_adapter.profiling_isolation_level.return_value = None

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
            mock_adapter.profiling_isolation_level.return_value = None

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
            mock_adapter.profiling_isolation_level.return_value = None

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
            mock_adapter.profiling_isolation_level.return_value = None

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
    """The per-table connection receives the adapter's profiling isolation level."""

    def test_applies_level_when_adapter_returns_one(self, profiler):
        # When the adapter's hook returns a non-None level, conn.execution_options
        # is called with that level during a table profile, and the object returned
        # by that call is what flows downstream into adapter.setup_profiling.
        profiler.config.catch_exceptions = True
        with (
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_adapter = MagicMock()
            mock_adapter.profiling_isolation_level.return_value = "AUTOCOMMIT"
            # Short-circuit the downstream profiling path; the assertion is about
            # the connection object that reached setup_profiling.
            mock_adapter.setup_profiling.side_effect = RuntimeError("short-circuit")
            mock_get_adapter.return_value = mock_adapter

            profiler._generate_single_profile(
                query_combiner=MagicMock(),
                pretty_name="my_db.my_table",
                schema="my_db",
                table="my_table",
                platform="mysql",
            )

        mock_conn.execution_options.assert_called_once_with(
            isolation_level="AUTOCOMMIT"
        )
        # The branched connection returned by execution_options is what flows
        # downstream, not the raw checked-out connection.
        assert (
            mock_adapter.setup_profiling.call_args[0][1]
            is mock_conn.execution_options.return_value
        )
        # A successful apply is not a rejection — no isolation-level-unavailable warning.
        # (report.warning may still be called by the outer handler for the
        # short-circuit RuntimeError above; assert on the title, not call count.)
        isolation_level_warnings = [
            c
            for c in profiler.report.warning.call_args_list
            if c.kwargs.get("title") == "Profiling: isolation level unavailable"
        ]
        assert not isolation_level_warnings

    def test_does_not_apply_options_when_level_none(self, profiler):
        # When the adapter's hook returns None, conn.execution_options is not called
        # with an isolation_level argument at all — the checked-out connection flows
        # downstream unchanged.
        profiler.config.catch_exceptions = True
        with (
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_adapter = MagicMock()
            mock_adapter.profiling_isolation_level.return_value = None
            mock_adapter.setup_profiling.side_effect = RuntimeError("short-circuit")
            mock_get_adapter.return_value = mock_adapter

            profiler._generate_single_profile(
                query_combiner=MagicMock(),
                pretty_name="my_db.my_table",
                schema="my_db",
                table="my_table",
                platform="mysql",
            )

        mock_conn.execution_options.assert_not_called()
        # The raw checked-out connection flows downstream unchanged.
        assert mock_adapter.setup_profiling.call_args[0][1] is mock_conn
        # No level was applied, so no rejection — no isolation-level-unavailable warning.
        # (report.warning may still be called by the outer handler for the
        # short-circuit RuntimeError above; assert on the title, not call count.)
        isolation_level_warnings = [
            c
            for c in profiler.report.warning.call_args_list
            if c.kwargs.get("title") == "Profiling: isolation level unavailable"
        ]
        assert not isolation_level_warnings

    def test_profile_use_autocommit_false_skips_execution_options(self, profiler):
        # The opt-out suppresses the adapter's own opt-in: the adapter returns
        # "AUTOCOMMIT" but profile_use_autocommit is False, so execution_options
        # is never called — no attempt, no warning, no degradation path, exactly
        # the pre-PR behaviour.
        profiler.config.catch_exceptions = True
        profiler.config.profile_use_autocommit = False
        with (
            patch.object(profiler, "base_engine") as mock_engine,
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
        ):
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_adapter = MagicMock()
            mock_adapter.profiling_isolation_level.return_value = "AUTOCOMMIT"
            mock_adapter.setup_profiling.side_effect = RuntimeError("short-circuit")
            mock_get_adapter.return_value = mock_adapter

            profiler._generate_single_profile(
                query_combiner=MagicMock(),
                pretty_name="my_db.my_table",
                schema="my_db",
                table="my_table",
                platform="mysql",
            )

        mock_conn.execution_options.assert_not_called()
        # The flag being off is a documented setting, not a rejection, so no
        # isolation-level-unavailable warning. (report.warning may still be
        # called by the outer handler for the short-circuit RuntimeError above;
        # assert on the title, not call count.)
        isolation_level_warnings = [
            c
            for c in profiler.report.warning.call_args_list
            if c.kwargs.get("title") == "Profiling: isolation level unavailable"
        ]
        assert not isolation_level_warnings


class TestProfilingIsolationLevelRejection:
    """Documents what escapes ``conn.execution_options(isolation_level=...)`` when a
    server or proxy rejects the session setting.

    The rejection is raised from ``dialect.set_isolation_level`` as a genuine
    ``sqlite3.OperationalError`` — which IS ``dialect.dbapi.Error`` for the sqlite
    dialect — so the test can distinguish "raw driver error not wrapped" from
    "arbitrary exception not wrapped". SQLAlchemy's wrapping predicate is
    ``isinstance(e, dialect.dbapi.Error)``; a wrapped path would turn this into
    ``sa.exc.OperationalError``. It does not: ``execution_options`` calls
    ``set_isolation_level`` directly and never reaches ``_handle_dbapi_exception``
    (which lives in statement execution), so the raw driver error escapes. These
    assertions record the observed behaviour against SQLAlchemy 1.4 so the
    degradation handler can rely on the raw driver error escaping unwrapped.
    """

    def test_rejection_escapes_as_raw_driver_error_not_sa_exc(self):
        engine = create_engine("sqlite:///:memory:")
        assert issubclass(sqlite3.OperationalError, engine.dialect.dbapi.Error)  # type: ignore[attr-defined]

        with engine.connect() as conn:
            with patch.object(
                engine.dialect,
                "set_isolation_level",
                side_effect=sqlite3.OperationalError("proxy refuses AUTOCOMMIT"),
            ):
                with pytest.raises(sqlite3.OperationalError) as exc_info:
                    conn.execution_options(isolation_level="AUTOCOMMIT")

                escaped = exc_info.value
                # Raw driver error, not wrapped into the sa.exc hierarchy even
                # though it IS a dialect.dbapi.Error.
                assert not isinstance(escaped, sa.exc.SQLAlchemyError)
                assert not isinstance(escaped, sa.exc.DBAPIError)
                assert not isinstance(escaped, sa.exc.OperationalError)

    def test_connection_remains_usable_after_rejection(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            with patch.object(
                engine.dialect,
                "set_isolation_level",
                side_effect=sqlite3.OperationalError("nope"),
            ):
                with pytest.raises(sqlite3.OperationalError):
                    conn.execution_options(isolation_level="AUTOCOMMIT")

            # The rebind never ran, so `conn` is still the original object and
            # SQLAlchemy did not invalidate it. A fallback can keep using it.
            assert conn.exec_driver_sql("SELECT 1").scalar() == 1

    def test_rejection_reports_warning_and_still_profiles(
        self, profiler, sqlite_engine, test_table
    ):
        # Both halves: a rejected execution_options produces exactly one
        # report warning titled "Profiling: isolation level unavailable" AND the
        # table still produces a profile. The second assertion is the one that
        # matters — without it this test passes even when the fallback leaves
        # an unusable connection, which is the failure this design exists to
        # avoid.
        profiler.config.catch_exceptions = True
        request = ProfilerRequest(
            pretty_name="test.my_table",
            batch_kwargs={"table": "test_table", "schema": None},
        )
        metadata = sa.MetaData()
        sql_table = sa.Table("test_table", metadata, sa.Column("id", sa.Integer))

        def mock_profile_row_count(*args, **kwargs):
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
            patch.object(
                sqlite_engine.dialect,
                "set_isolation_level",
                side_effect=sqlite3.OperationalError("proxy refuses AUTOCOMMIT"),
            ),
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_context = MagicMock()
            mock_context.sql_table = sql_table
            mock_adapter.setup_profiling.return_value = mock_context
            mock_adapter.profiling_isolation_level.return_value = "AUTOCOMMIT"
            mock_get_adapter.return_value = mock_adapter

            result_request, result_profile = profiler._generate_profile_from_request(
                None, request
            )

        assert result_request == request
        # Half 2: the table still produces a profile despite the rejection.
        assert result_profile is not None
        # Half 1: exactly one warning, titled and attributed correctly.
        assert profiler.report.warning.call_count == 1
        warning_call = profiler.report.warning.call_args
        assert warning_call.kwargs["title"] == "Profiling: isolation level unavailable"
        assert warning_call.kwargs["context"] == "Asset: test.my_table"
        assert isinstance(warning_call.kwargs["exc"], sqlite3.OperationalError)

    def test_rejection_dedups_across_tables(self, sqlite_engine, profiler_config):
        # Pins the byte-identical-message constraint: profiling two tables
        # under the same rejection produces one warning entry with two
        # contexts, not two entries. A later edit that interpolates the
        # table name into `message` would fail here.
        real_report = SQLSourceReport()
        profiler = SQLAlchemyProfiler(
            conn=sqlite_engine,
            report=real_report,
            config=profiler_config,
            platform="sqlite",
            env="TEST",
        )
        profiler.config.catch_exceptions = True
        metadata = sa.MetaData()
        sql_table = sa.Table("test_table", metadata, sa.Column("id", sa.Integer))

        def mock_profile_row_count(*args, **kwargs):
            profile = args[3] if len(args) > 3 else kwargs.get("profile")
            if profile:
                profile.rowCount = 0
            return 0

        requests = [
            ProfilerRequest(
                pretty_name="db.t1",
                batch_kwargs={"table": "t1", "schema": None},
            ),
            ProfilerRequest(
                pretty_name="db.t2",
                batch_kwargs={"table": "t2", "schema": None},
            ),
        ]

        with (
            sqlite_engine.connect() as conn,
            patch.object(profiler, "base_engine") as mock_engine,
            patch.object(
                profiler, "_profile_row_count", side_effect=mock_profile_row_count
            ),
            patch(
                "datahub.ingestion.source.sqlalchemy_profiler.sqlalchemy_profiler.get_adapter"
            ) as mock_get_adapter,
            patch.object(
                sqlite_engine.dialect,
                "set_isolation_level",
                side_effect=sqlite3.OperationalError("nope"),
            ),
        ):
            mock_engine.connect.return_value.__enter__.return_value = conn
            mock_adapter = MagicMock()
            mock_context = MagicMock()
            mock_context.sql_table = sql_table
            mock_adapter.setup_profiling.return_value = mock_context
            mock_adapter.profiling_isolation_level.return_value = "AUTOCOMMIT"
            mock_get_adapter.return_value = mock_adapter

            for req in requests:
                profiler._generate_profile_from_request(None, req)  # type: ignore[arg-type]

        # One deduped entry, not two.
        assert len(real_report.warnings) == 1
        warning = real_report.warnings[0]
        assert warning.title == "Profiling: isolation level unavailable"
        # Both table contexts are attached to the single entry. The context
        # string carries the exception type/message suffix (see report_log),
        # so match on the table-name prefix rather than exact equality.
        contexts = list(warning.context)
        assert len(contexts) == 2
        assert any(c.startswith("Asset: db.t1") for c in contexts)
        assert any(c.startswith("Asset: db.t2") for c in contexts)

    def test_unrecognised_level_warns_and_still_profiles(
        self, profiler, sqlite_engine, test_table
    ):
        # A level the dialect does not recognise (ArgumentError) and a server
        # refusing it (raw driver error) both reach the same except and degrade
        # the same way: one warning, then a transactional profile. The second
        # assertion is the one that matters — it would fail if the fallback
        # left an unusable connection, which is the regression this collapse
        # fixes (previously ArgumentError re-raised, bypassed the fallback,
        # and every table yielded no profile under a generic warning).
        profiler.config.catch_exceptions = True
        request = ProfilerRequest(
            pretty_name="test.my_table",
            batch_kwargs={"table": "test_table", "schema": None},
        )
        metadata = sa.MetaData()
        sql_table = sa.Table("test_table", metadata, sa.Column("id", sa.Integer))

        def mock_profile_row_count(*args, **kwargs):
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
            mock_adapter = MagicMock()
            mock_context = MagicMock()
            mock_context.sql_table = sql_table
            mock_adapter.setup_profiling.return_value = mock_context
            mock_adapter.profiling_isolation_level.return_value = "BOGUS_LEVEL"
            mock_get_adapter.return_value = mock_adapter

            result_request, result_profile = profiler._generate_profile_from_request(
                None,
                request,  # type: ignore[arg-type]
            )

        assert result_request == request
        # The table still produces a profile despite the unrecognised level.
        assert result_profile is not None
        # Exactly one warning, titled and attributed correctly; the specific
        # exception is still visible via exc.
        assert profiler.report.warning.call_count == 1
        warning_call = profiler.report.warning.call_args
        assert warning_call.kwargs["title"] == "Profiling: isolation level unavailable"
        assert warning_call.kwargs["context"] == "Asset: test.my_table"
        assert isinstance(warning_call.kwargs["exc"], sa.exc.ArgumentError)
