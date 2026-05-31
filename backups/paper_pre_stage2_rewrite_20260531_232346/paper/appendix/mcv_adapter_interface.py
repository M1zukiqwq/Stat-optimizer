"""
Statistics-Format Conversion Interface
======================================

Abstract reference interfaces for translating between engine-native
statistics layouts and the canonical full-distribution representation used by
OASIS. The concrete appendix example is PostgreSQL-style `pg_statistic` metadata.

Important modeling detail: PostgreSQL single-column histograms store ordered
histogram boundary values for the residual non-MCV population. They do not store
explicit per-bucket frequencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class MCVEntry:
    """Most-common-value entry with explicit probability mass."""

    value: float
    frequency: float


@dataclass
class RangeBucket:
    """Canonical range bucket used inside the OASIS-facing full distribution."""

    lower: float
    upper: float
    frequency: float


@dataclass
class SingletonBucket:
    """Canonical point-mass bucket used for explicit high-frequency values."""

    value: float
    frequency: float


@dataclass
class FullHistogram:
    """
    Canonical distribution exposed to OASIS.

    `singletons` capture explicit high-frequency values.
    `ranges` capture the residual mass as range buckets.
    `target_histogram_size` records how many PostgreSQL histogram boundary values
    should be emitted when decomposing back to `pg_statistic`.
    """

    singletons: List[SingletonBucket]
    ranges: List[RangeBucket]
    null_fraction: float = 0.0
    target_histogram_size: int = 0
    n_distinct: int = 0

    def total_non_null_frequency(self) -> float:
        """Return the total non-null probability mass in the canonical view."""

        return sum(item.frequency for item in self.singletons) + sum(
            item.frequency for item in self.ranges
        )


@dataclass
class PostgreSQLStatistics:
    """
    PostgreSQL-style statistics for the single-column path considered here.

    `histogram_bounds` stores the ordered residual histogram boundary values.
    If there are `m + 1` bounds, they define `m` residual intervals of equal
    depth over the non-MCV population.
    """

    mcv_list: List[MCVEntry]
    histogram_bounds: List[float]
    n_distinct: int
    null_fraction: float = 0.0


# ============================================================================
# Phase 1: Reconstruction (PostgreSQL -> canonical full histogram)
# ============================================================================


class MCVToFullHistogramConverter(ABC):
    """Convert PostgreSQL statistics into the canonical OASIS representation."""

    @abstractmethod
    def reconstruct(
        self,
        pg_stats: PostgreSQLStatistics,
        singleton_threshold: float = 0.0,
    ) -> FullHistogram:
        """
        Build a canonical full distribution from PostgreSQL statistics.

        Expected behavior:
        1. Materialize each MCV entry as a singleton bucket.
        2. Compute residual mass:
           `p_res = 1 - null_fraction - sum(mcv frequencies)`.
        3. Interpret adjacent `histogram_bounds` pairs as residual intervals.
        4. Assign each interval an equal share of `p_res`.
        5. Emit a `FullHistogram` whose non-null mass is preserved exactly.
        """

    @abstractmethod
    def validate_reconstruction(
        self,
        pg_stats: PostgreSQLStatistics,
        full_hist: FullHistogram,
        tolerance: float = 1e-12,
    ) -> Tuple[bool, str]:
        """
        Validate the reconstruction result.

        Typical checks:
        - histogram bounds are monotone,
        - every MCV entry appears as a singleton,
        - non-null mass is preserved within `tolerance`.
        """


# ============================================================================
# Phase 3: Decomposition (canonical full histogram -> PostgreSQL)
# ============================================================================


class FullHistogramToMCVConverter(ABC):
    """Convert a corrected full distribution back to PostgreSQL statistics."""

    @abstractmethod
    def decompose(
        self,
        full_hist: FullHistogram,
        mcv_threshold: float = 0.01,
        mcv_limit: int = 100,
    ) -> PostgreSQLStatistics:
        """
        Decompose the canonical full distribution back to PostgreSQL format.

        Expected behavior:
        1. Select singleton buckets with `frequency >= mcv_threshold`.
        2. Keep the top `mcv_limit` candidates by frequency.
        3. Form a residual distribution from the remaining singletons and ranges.
        4. Emit `histogram_bounds` by sampling residual quantiles according to
           `full_hist.target_histogram_size`.
        5. Preserve the non-null mass exactly.
        """

    @abstractmethod
    def extract_mcv_candidates(
        self,
        full_hist: FullHistogram,
        mcv_threshold: float,
        mcv_limit: int,
    ) -> List[MCVEntry]:
        """Return the selected MCV entries from the canonical distribution."""

    @abstractmethod
    def construct_histogram_bounds(
        self,
        full_hist: FullHistogram,
        selected_mcv: List[MCVEntry],
    ) -> List[float]:
        """
        Build PostgreSQL residual histogram bounds from the residual mixture.

        A concrete implementation typically converts the residual mixture into a
        CDF/quantile representation and samples `target_histogram_size` ordered
        boundary values.
        """

    @abstractmethod
    def validate_decomposition(
        self,
        full_hist: FullHistogram,
        pg_stats: PostgreSQLStatistics,
        tolerance: float = 1e-12,
    ) -> Tuple[bool, str]:
        """
        Validate the decomposition result.

        Typical checks:
        - histogram bounds are monotone,
        - selected MCV values are not duplicated in the residual view,
        - non-null mass is preserved within `tolerance`.
        """


# ============================================================================
# End-to-end conversion interface
# ============================================================================


class PostgreSQLStatisticsAdapter(ABC):
    """
    Abstract end-to-end conversion layer used to wrap OASIS around PostgreSQL statistics.

    Phase 1: `PostgreSQLStatistics -> FullHistogram`
    Phase 2: `FullHistogram -> corrected FullHistogram` (OASIS core)
    Phase 3: `FullHistogram -> PostgreSQLStatistics`
    """

    def __init__(
        self,
        mcv_threshold: float = 0.01,
        mcv_limit: int = 100,
        singleton_threshold: float = 0.0,
    ) -> None:
        self.mcv_threshold = mcv_threshold
        self.mcv_limit = mcv_limit
        self.singleton_threshold = singleton_threshold
        self.reconstructor = self._create_reconstructor()
        self.decomposer = self._create_decomposer()

    @abstractmethod
    def _create_reconstructor(self) -> MCVToFullHistogramConverter:
        """Create the Phase-1 reconstructor."""

    @abstractmethod
    def _create_decomposer(self) -> FullHistogramToMCVConverter:
        """Create the Phase-3 decomposer."""

    def to_full_histogram(self, pg_stats: PostgreSQLStatistics) -> FullHistogram:
        """Run Phase 1 and validate the reconstructed canonical distribution."""

        full_hist = self.reconstructor.reconstruct(pg_stats, self.singleton_threshold)
        is_valid, message = self.reconstructor.validate_reconstruction(
            pg_stats, full_hist
        )
        if not is_valid:
            raise ValueError(f"Reconstruction validation failed: {message}")
        return full_hist

    def from_full_histogram(self, full_hist: FullHistogram) -> PostgreSQLStatistics:
        """Run Phase 3 and validate the emitted PostgreSQL statistics."""

        pg_stats = self.decomposer.decompose(
            full_hist,
            mcv_threshold=self.mcv_threshold,
            mcv_limit=self.mcv_limit,
        )
        is_valid, message = self.decomposer.validate_decomposition(
            full_hist, pg_stats
        )
        if not is_valid:
            raise ValueError(f"Decomposition validation failed: {message}")
        return pg_stats

    def round_trip_test(
        self,
        pg_stats: PostgreSQLStatistics,
        tolerance: float = 1e-12,
    ) -> Tuple[bool, str]:
        """Check that `PostgreSQL -> canonical -> PostgreSQL` preserves mass."""

        original_non_null = 1.0 - pg_stats.null_fraction
        reconstructed = self.from_full_histogram(self.to_full_histogram(pg_stats))
        reconstructed_non_null = 1.0 - reconstructed.null_fraction
        if abs(original_non_null - reconstructed_non_null) > tolerance:
            return (
                False,
                "non-null mass changed during round-trip: "
                f"{original_non_null:.12f} -> {reconstructed_non_null:.12f}",
            )
        return True, "round-trip mass check passed"


# ============================================================================
# Sketch extension: SQL Server
# ============================================================================


@dataclass
class SQLServerStatistics:
    """Minimal sketch of SQL Server statistics for appendix discussion."""

    density: float
    steps: List[RangeBucket]
    n_distinct: int


class SQLServerStatisticsAdapter(ABC):
    """Abstract extension point for the SQL Server discussion in the appendix."""

    @abstractmethod
    def to_full_histogram(self, sql_stats: SQLServerStatistics) -> FullHistogram:
        """Convert SQL Server statistics to the canonical full distribution."""

    @abstractmethod
    def from_full_histogram(self, full_hist: FullHistogram) -> SQLServerStatistics:
        """Convert the canonical full distribution back to SQL Server format."""


# ============================================================================
# Example
# ============================================================================


def example_usage() -> None:
    """Print a small example object for the appendix reader."""

    pg_stats = PostgreSQLStatistics(
        mcv_list=[
            MCVEntry(value=0.50, frequency=0.15),
            MCVEntry(value=0.80, frequency=0.10),
        ],
        histogram_bounds=[0.00, 0.30, 0.60, 1.00],
        n_distinct=1000,
        null_fraction=0.0,
    )

    print("Abstract appendix conversion interface only; no concrete adapter is instantiated.")
    print(f"Example PostgreSQL stats: {pg_stats}")


if __name__ == "__main__":
    example_usage()
