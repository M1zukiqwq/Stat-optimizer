#!/usr/bin/env python3
"""
FastAPI service for OASIS KLL histogram correction.
Receives requests from Presto KllHistogramCorrector and returns corrected quantiles.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cdf_teacher import estimate_observation_coverage
from histogram_math import clamp, project_monotonic
from histogram_types import FeedbackObservation, KllFeedbackSample, KllPrior
from ridge_histogram_model import RidgeMultiOutputRegressor
from tensorizer import tensorize_sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OASIS KLL Correction Service")

# Global model instance
_model: Optional[RidgeMultiOutputRegressor] = None
_max_observations: int = 16


class PriorKllRequest(BaseModel):
    """Prior KLL histogram from Presto (actual value domain)"""
    min: float
    max: float
    null_fraction: float = 0.0
    quantile_levels: List[float] = Field(..., min_items=1)
    quantile_values: List[float] = Field(..., min_items=1)


class ObservationRequest(BaseModel):
    """Feedback observation from ML feedback listener"""
    predicate_type: str = Field(..., alias="predicate")
    value: float = Field(..., alias="predicate_value")
    value_upper: Optional[float] = Field(None, alias="predicate_value_upper")
    estimated_selectivity: float = Field(0.0, alias="estimated_sel")
    actual_selectivity: float = Field(..., alias="actual_sel")
    timestamp: str = Field(..., alias="query_timestamp")

    class Config:
        populate_by_name = True


class PredictRequest(BaseModel):
    """Request format from Presto KllHistogramCorrector"""
    prior_kll: PriorKllRequest
    observations: List[ObservationRequest] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """Response format expected by KllHistogramCorrector"""
    corrected_quantile_values: List[float]


def _normalize_values(values: List[float], min_val: float, max_val: float) -> List[float]:
    """Normalize values from [min, max] to [0, 1]"""
    value_range = max(max_val - min_val, 1e-12)
    return [clamp((v - min_val) / value_range, 0.0, 1.0) for v in values]


def _denormalize_values(values: List[float], min_val: float, max_val: float) -> List[float]:
    """Denormalize values from [0, 1] to [min, max]"""
    value_range = max(max_val - min_val, 1e-12)
    return [clamp(min_val + v * value_range, min_val, max_val) for v in values]


def _convert_to_feedback_sample(request: PredictRequest) -> KllFeedbackSample:
    """Convert FastAPI request to KllFeedbackSample with normalization"""
    prior_req = request.prior_kll

    # Normalize quantile values to [0, 1]
    normalized_values = _normalize_values(
        prior_req.quantile_values,
        prior_req.min,
        prior_req.max
    )

    # Create normalized prior (min=0, max=1 as required by KllPrior)
    prior = KllPrior(
        min_value=0.0,
        max_value=1.0,
        null_fraction=prior_req.null_fraction,
        quantile_levels=list(prior_req.quantile_levels),
        quantile_values=normalized_values,
        value_type="double",
        sketch_k=1024,
    )

    # Convert observations - normalize predicate values
    observations = []
    for obs_req in request.observations:
        try:
            from datetime import datetime
            timestamp = datetime.fromisoformat(obs_req.timestamp.replace("Z", "+00:00"))

            normalized_value = clamp(
                (obs_req.value - prior_req.min) / max(prior_req.max - prior_req.min, 1e-12),
                0.0, 1.0
            )
            normalized_upper = None
            if obs_req.value_upper is not None:
                normalized_upper = clamp(
                    (obs_req.value_upper - prior_req.min) / max(prior_req.max - prior_req.min, 1e-12),
                    0.0, 1.0
                )

            observations.append(FeedbackObservation(
                predicate_type=obs_req.predicate_type,
                value=normalized_value,
                value_upper=normalized_upper,
                estimated_selectivity=obs_req.estimated_selectivity,
                actual_selectivity=obs_req.actual_selectivity,
                timestamp=timestamp,
            ))
        except Exception as e:
            logger.warning(f"Skipping invalid observation: {e}")
            continue

    return KllFeedbackSample(
        prior=prior,
        observations=observations,
        corrected_quantile_values=None,
        source_path=None,
    )


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global _model, _max_observations

    model_path = Path(__file__).parent / "artifacts" / "kll_ridge_model.json"
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        logger.warning("Service will return prior quantiles as fallback")
        return

    try:
        _model = RidgeMultiOutputRegressor.load(str(model_path))
        metadata = RidgeMultiOutputRegressor.load_metadata(str(model_path))
        _max_observations = int(metadata.get("max_observations", 16))
        logger.info(f"Loaded model from {model_path}, max_observations={_max_observations}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Service will return prior quantiles as fallback")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict corrected KLL quantiles.

    Input: prior_kll (actual value domain) + observations
    Output: corrected_quantile_values (actual value domain)
    """
    try:
        # Convert request to normalized FeedbackSample
        sample = _convert_to_feedback_sample(request)

        # Store original min/max for denormalization
        original_min = request.prior_kll.min
        original_max = request.prior_kll.max

        # Fallback if insufficient observations
        if len(sample.observations) < 3:
            logger.info("Insufficient observations (<3), returning prior quantiles")
            return PredictResponse(
                corrected_quantile_values=list(request.prior_kll.quantile_values)
            )

        # Fallback if model not loaded
        if _model is None:
            logger.warning("Model not loaded, returning prior quantiles")
            return PredictResponse(
                corrected_quantile_values=list(request.prior_kll.quantile_values)
            )

        # Run prediction on normalized data
        tensor_record = tensorize_sample(sample, max_observations=_max_observations, teacher_fn=None)
        predicted_normalized = _model.predict([tensor_record.feature_tensor])[0]
        predicted_normalized = project_monotonic(predicted_normalized)

        # Blend with prior if low coverage
        coverage = estimate_observation_coverage(sample)
        if coverage < 0.2:
            blend = coverage / 0.2
            prior_normalized = sample.prior.quantile_values
            predicted_normalized = [
                (1.0 - blend) * prior + blend * pred
                for prior, pred in zip(prior_normalized, predicted_normalized)
            ]

        # Denormalize back to original value domain
        corrected_values = _denormalize_values(predicted_normalized, original_min, original_max)

        logger.info(f"Predicted {len(corrected_values)} quantiles, coverage={coverage:.3f}")
        return PredictResponse(corrected_quantile_values=corrected_values)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "max_observations": _max_observations,
    }


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8081, help='Service port')
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
