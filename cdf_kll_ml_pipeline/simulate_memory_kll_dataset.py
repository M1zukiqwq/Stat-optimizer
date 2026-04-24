import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from histogram_math import clamp01, evaluate_piecewise_cdf
from kll_codec import default_quantile_levels, encode_simulated_kll

PREDICATES = ["<", "<=", ">", ">=", "=", "BETWEEN"]


class MemoryTable:
    def __init__(self, data: List[float], null_count: int, min_val: float = 0.0, max_val: float = 1.0):
        self.data = data
        self.null_count = null_count
        self.min_val = min_val
        self.max_val = max_val

    def total_rows(self) -> int:
        return len(self.data) + self.null_count

    def get_null_fraction(self) -> float:
        total = self.total_rows()
        if total == 0:
            return 0.0
        return self.null_count / total

    def get_quantiles(self, levels: List[float]) -> List[float]:
        """Compute quantiles using linear interpolation (matches numpy.percentile 'linear').

        This is more accurate than floor indexing and better aligns with the
        continuous-distribution semantics of KLL sketches.
        """
        if not self.data:
            return [self.min_val] * len(levels)
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        out = []
        for p in levels:
            # Linear interpolation: p * (n-1) maps [0,1] -> [0, n-1] index range
            idx_float = p * (n - 1)
            idx_low = int(idx_float)
            idx_high = min(idx_low + 1, n - 1)
            frac = idx_float - idx_low
            interpolated = sorted_data[idx_low] * (1.0 - frac) + sorted_data[idx_high] * frac
            out.append(interpolated)
        return out

    def get_bucket_boundaries(self, bucket_count: int) -> List[float]:
        levels = [i / bucket_count for i in range(1, bucket_count)]
        quantiles = self.get_quantiles(levels)
        return [self.min_val] + quantiles + [self.max_val]

    def query_conditional_sel(self, predicate: str, val: float, val_upper: Optional[float] = None) -> float:
        """Return selectivity among non-null rows only (conditional on row being non-null)."""
        non_null_count = len(self.data)
        if non_null_count == 0:
            return 0.0

        if predicate in {"<", "<="}:
            matches = sum(1 for x in self.data if x <= val)
        elif predicate in {">", ">="}:
            matches = sum(1 for x in self.data if x >= val)
        elif predicate == "BETWEEN":
            if val_upper is None:
                matches = 0
            else:
                matches = sum(1 for x in self.data if val <= x <= val_upper)
        else:  # "="
            epsilon = (self.max_val - self.min_val) * 0.005
            matches = sum(1 for x in self.data if abs(x - val) <= epsilon)

        return matches / non_null_count

    def apply_drift(self, rng: random.Random, q_mods: int, persistent_center: Optional[float] = None):
        """每轮执行全部四种操作（insert / delete / update / null_change），
        更真实地模拟生产环境中同时发生多类 DML 的场景。
        q_mods 控制每种操作重复的轮次数。
        为避免随机中心相互抵消导致全局平坦化，支持传入 persistent_center 以制造持续的分布偏斜。
        """
        for _ in range(q_mods):
            # ── insert：集中向某个区间注入数据，制造分布偏斜 ──
            center = persistent_center if persistent_center is not None else rng.uniform(0.1, 0.9)
            batch = rng.randint(10, 100)
            for _ in range(batch):
                self.data.append(clamp01(rng.normalvariate(center, 0.05)))

            # ── delete：随机删除一批行 ──
            if self.data:
                batch = min(len(self.data), rng.randint(10, 100))
                for _ in range(batch):
                    idx = rng.randint(0, len(self.data) - 1)
                    del self.data[idx]

            # ── update：随机扰动一批行的值 ──
            if self.data:
                batch = min(len(self.data), rng.randint(10, 100))
                for _ in range(batch):
                    idx = rng.randint(0, len(self.data) - 1)
                    self.data[idx] = clamp01(self.data[idx] + rng.uniform(-0.1, 0.1))

            # ── null_change：调整 null 行数量 ──
            self.null_count = max(0, self.null_count + rng.randint(-50, 50))


def _draw_observation(
    rng: random.Random,
    table: MemoryTable,
    prior_x: List[float],
    prior_p: List[float],
    prior_null_frac: float,
    ts: datetime,
) -> dict:
    """Draw one simulated query observation.

    Semantics:
    - ``estimated_sel`` = what Presto's CBO would compute using the STALE prior
      statistics (prior CDF + prior null_fraction that was snapshotted at ANALYZE
      time).  Intentionally does NOT use the current post-drift null_fraction.
    - ``actual_sel`` = the true selectivity of the query against the CURRENT
      (post-drift) table, computed as:
          P(condition | non-null) * P(non-null current)
      This matches the convention in generate_synthetic_json_dataset.py so that
      cdf_teacher._effective_actual() can invert the null-fraction factor
      uniformly across both generators.
    """
    predicate = rng.choice(PREDICATES)
    val = rng.uniform(0.0, 1.0)
    val_upper = None

    if predicate == "BETWEEN":
        val2 = rng.uniform(0.0, 1.0)
        val, val_upper = sorted((val, val2))
        # Conditional selectivity from stale prior CDF
        est_cond = clamp01(
            evaluate_piecewise_cdf(prior_x, prior_p, val_upper)
            - evaluate_piecewise_cdf(prior_x, prior_p, val)
        )
    elif predicate in {"<", "<="}:
        est_cond = clamp01(evaluate_piecewise_cdf(prior_x, prior_p, val))
    elif predicate in {">", ">="}:
        est_cond = clamp01(1.0 - evaluate_piecewise_cdf(prior_x, prior_p, val))
    else:  # "="
        width = 0.01
        left = max(0.0, val - width)
        right = min(1.0, val + width)
        est_cond = clamp01(
            evaluate_piecewise_cdf(prior_x, prior_p, right)
            - evaluate_piecewise_cdf(prior_x, prior_p, left)
        )

    # estimated_sel: stale prior CDF * stale prior non-null fraction.
    # This is what Presto's query optimizer would estimate using ANALYZE statistics.
    est_overall = clamp01(est_cond * max(1.0 - prior_null_frac, 1e-6))

    # actual_sel: ground truth, computed as conditional_sel * current_non_null_frac.
    # Using query_conditional_sel (denominator = non-null rows) then scaling by
    # the current non-null fraction mirrors generate_synthetic_json_dataset.py
    # and allows cdf_teacher._effective_actual to invert the null-fraction factor.
    current_non_null_frac = max(1.0 - table.get_null_fraction(), 1e-6)
    act_cond = table.query_conditional_sel(predicate, val, val_upper)
    act_overall = clamp01(act_cond * current_non_null_frac)

    observation = {
        "predicate_type": predicate,
        "value": round(val, 6),
        "estimated_sel": round(est_overall, 6),
        "actual_sel": round(act_overall, 6),
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
    }
    if val_upper is not None:
        observation["value_upper"] = round(val_upper, 6)
    return observation


def generate_initial_data(rng: random.Random, size: int) -> Tuple[List[float], int]:
    data = []
    # Mix normal distributions to create an interesting initial histogram
    centers = [rng.uniform(0.1, 0.9) for _ in range(rng.randint(2, 4))]
    for _ in range(size):
        c = rng.choice(centers)
        v = rng.normalvariate(c, 0.1)
        data.append(clamp01(v))
    null_count = int(size * rng.uniform(0.01, 0.1))
    return data, null_count


def build_case(case_index: int, rng: random.Random, bucket_count: int, sketch_k: int, initial_rows: int, q_modifications: int) -> dict:
    data, null_count = generate_initial_data(rng, initial_rows)
    table = MemoryTable(data, null_count)

    # Prior metadata
    prior_null_frac = table.get_null_fraction()
    quantile_levels = default_quantile_levels(bucket_count - 1)
    prior_boundaries = table.get_bucket_boundaries(bucket_count)
    prior_quantiles = prior_boundaries[1:-1]
    
    prior_x = prior_boundaries
    prior_p = [i / bucket_count for i in range(bucket_count + 1)]

    prior_sketch_base64 = encode_simulated_kll(
        quantile_levels=quantile_levels,
        quantile_values=prior_quantiles,
        min_value=0.0,
        max_value=1.0,
        value_type="double",
        sketch_k=sketch_k,
    )

    observations = []
    observation_count = rng.randint(8, 24)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=case_index)
    
    # 确定本 case 的持续漂移方向（Persistent Hotspot），制造定向的显著偏差
    persistent_center = rng.uniform(0.1, 0.9)

    # Simulate drift interleaving with observations
    for obs_index in range(observation_count):
        # Apply q modifications before each observation to drift the table over time
        # 传入 persistent_center 确保所有的 insert 都向同一区域倾斜
        table.apply_drift(rng, q_modifications, persistent_center=persistent_center)
        
        ts = base_time + timedelta(hours=obs_index)
        obs = _draw_observation(rng, table, prior_x, prior_p, prior_null_frac, ts)
        observations.append(obs)

    # Finally, get Ground Truth Corrected Quantiles from the heavily drifted table
    true_boundaries = table.get_bucket_boundaries(bucket_count)
    true_quantiles = true_boundaries[1:-1]

    return {
        "prior_kll": {
            "type": "double",
            "k": sketch_k,
            "min": 0.0,
            "max": 1.0,
            "null_fraction": round(prior_null_frac, 6),
            "quantile_levels": [round(level, 6) for level in quantile_levels],
            "quantile_values": [round(value, 6) for value in prior_quantiles],
            "bucket_boundaries": [round(value, 6) for value in prior_boundaries],
            "sketch_bytes_base64": prior_sketch_base64,
        },
        "observations": observations,
        "corrected_kll": {
            "type": "double",
            "k": sketch_k,
            "quantile_levels": [round(level, 6) for level in quantile_levels],
            "quantile_values": [round(value, 6) for value in true_quantiles],
            "bucket_boundaries": [round(value, 6) for value in true_boundaries],
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic KLL dataset via Memory Table simulation")
    parser.add_argument("--output-dir", default="training_data_sim", help="Output directory")
    parser.add_argument("--k", type=int, default=32, help="Generate k training JSON files")
    parser.add_argument("--num-buckets", type=int, default=10, help="Bucket count for bounding grid")
    parser.add_argument("--sketch-k", type=int, default=1024, help="K parameter for KLL metadata")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--initial-rows", type=int, default=10000, help="Initial memory table size")
    parser.add_argument("--q", type=int, dest="q_mods", default=5, help="Number of modification batches between queries")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for case_index in range(args.k):
        payload = build_case(
            case_index=case_index, 
            rng=rng, 
            bucket_count=args.num_buckets, 
            sketch_k=args.sketch_k,
            initial_rows=args.initial_rows,
            q_modifications=args.q_mods
        )
        output_file = output_dir / f"sim_case_{case_index:04d}.json"
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Generated {args.k} simulated files into {output_dir.resolve()}")


if __name__ == "__main__":
    main()
