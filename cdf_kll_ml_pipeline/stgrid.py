import random
from typing import List

def correct_stgrid(
    prior_min: float,
    prior_max: float,
    prior_quantiles: List[float], # The bucket boundaries from prior
    observations: List[dict],
    num_buckets: int = 10,
    lr: float = 0.5,
) -> List[float]:
    """
    Self-Tuning Grid (STGrid) Simulator:
    Refines bucket frequencies online using query feedback.
    Maintains equal-width buckets and updates density proportionally.
    """
    if prior_max <= prior_min:
        return prior_quantiles # fallback
        
    B = num_buckets
    # Fixed equal-width grid boundaries
    grid_boundaries = [prior_min + i * (prior_max - prior_min) / B for i in range(B + 1)]
    
    # Initialize densities using prior quantiles (approximate CDF)
    cdf_x = [prior_min] + prior_quantiles + [prior_max]
    b_prior = len(cdf_x) - 1
    cdf_p = [i / b_prior for i in range(b_prior + 1)]
    
    from histogram_math import evaluate_piecewise_cdf
    
    # Initialize bucket probabilities (total sum = 1.0)
    p = [0.0] * B
    for i in range(B):
        p_left = evaluate_piecewise_cdf(cdf_x, cdf_p, grid_boundaries[i])
        p_right = evaluate_piecewise_cdf(cdf_x, cdf_p, grid_boundaries[i+1])
        p[i] = max(0.0, p_right - p_left)
        
    # Normalize probabilities
    s = sum(p)
    if s > 0:
        p = [x / s for x in p]
        
    # Process observations chronologically
    for obs in observations:
        predicate = obs["predicate_type"]
        val = obs["value"]
        act_sel = obs["actual_sel"]
        
        # Calculate overlap fraction for each bucket
        fracs = [0.0] * B
        for i in range(B):
            b_min = grid_boundaries[i]
            b_max = grid_boundaries[i+1]
            b_width = b_max - b_min
            if b_width <= 0: continue
            
            overlap_min = b_min
            overlap_max = b_max
            
            if predicate in {"<", "<="}:
                overlap_max = min(b_max, val)
            elif predicate in {">", ">="}:
                overlap_min = max(b_min, val)
            elif predicate == "BETWEEN":
                val_upper = obs.get("value_upper", val)
                overlap_min = max(b_min, val)
                overlap_max = min(b_max, val_upper)
            elif predicate == "=":
                epsilon = (prior_max - prior_min) * 0.005
                overlap_min = max(b_min, val - epsilon)
                overlap_max = min(b_max, val + epsilon)
            
            if overlap_min < overlap_max:
                fracs[i] = (overlap_max - overlap_min) / b_width
        
        # Estimate overall selectivity
        est_sel = sum(p[i] * fracs[i] for i in range(B))
        error = act_sel - est_sel
        
        # Update overlapping buckets proportionally to their overlap
        # Using a learning rate (alpha) mechanism similar to STHist
        total_overlap = sum(fracs)
        if total_overlap > 0:
            for i in range(B):
                # Gradient-like step: add error proportionally
                p[i] += lr * error * (fracs[i] / total_overlap)
                p[i] = max(0.0, p[i]) # clamp non-negative
                
            # Recompute and normalize
            s = sum(p)
            if s > 0:
                p = [x / s for x in p]
            else:
                # If all zeroed out, uniform distribution
                p = [1.0/B]*B

    # Convert the updated grid probabilities back to equi-depth quantiles
    # (since the original evaluation script expects equip-depth bucket boundaries)
    # We construct the piece-wise CDF of the Grid and resample it at equi-depth levels.
    grid_cdf_x = grid_boundaries
    grid_cdf_p = [0.0] * (B + 1)
    for i in range(B):
        grid_cdf_p[i+1] = grid_cdf_p[i] + p[i]
    grid_cdf_p[-1] = 1.0 # Guarantee 1.0
    
    # Resample to B-1 internal quantiles
    def inverse_cdf(target_p: float) -> float:
        for i in range(1, B + 1):
            if grid_cdf_p[i] >= target_p:
                if grid_cdf_p[i] == grid_cdf_p[i-1]:
                    return grid_cdf_x[i]
                # Linear interpolate
                fraction = (target_p - grid_cdf_p[i-1]) / (grid_cdf_p[i] - grid_cdf_p[i-1])
                return grid_cdf_x[i-1] + fraction * (grid_cdf_x[i] - grid_cdf_x[i-1])
        return grid_cdf_x[-1]

    target_levels = [i / B for i in range(1, B)]
    corrected_quantiles = [inverse_cdf(lvl) for lvl in target_levels]
    
    return corrected_quantiles
