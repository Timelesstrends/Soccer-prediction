"""
model_trainer.py — Dixon-Coles attack/defence coefficient fitting.

How it works
------------
The Dixon-Coles model (Dixon & Coles, 1997) treats goals scored by each
team as independent Poisson random variables whose means are:

    λ_home = attack_home * defence_away * home_advantage
    λ_away = attack_away * defence_home

We fit one attack (α) and one defence (β) coefficient per team by
maximising the log-likelihood of the observed scorelines in our results DB.

A low-scoring correction factor ρ is also fitted — it adjusts the joint
probabilities for 0-0, 1-0, 0-1, and 1-1 results which occur slightly
more/less often than independent Poisson predicts.

Fallback
--------
If fewer than MIN_MATCHES resolved matches exist, training is skipped and
the existing hand-tuned blend weights remain active. This is intentional —
overfitting 30 matches is worse than good priors.

Output
------
Fitted coefficients are saved to MODEL_COEFFICIENTS_PATH as JSON:

    {
        "fitted": true,
        "trained_on": 412,
        "trained_at": "2026-03-15T02:05:31",
        "home_advantage": 1.24,
        "rho": -0.13,
        "teams": {
            "Arsenal":  {"attack": 1.41, "defence": 0.72},
            "Chelsea":  {"attack": 1.08, "defence": 1.03},
            ...
        }
    }

When fitted=false the file still exists but signals to the client that
hand-tuned weights should remain active.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_COEFFICIENTS_PATH = Path("model_coefficients.json")

# Minimum resolved matches before we trust the fitted model over priors
MIN_MATCHES = 100

# Bounds for scipy.optimize — keeps coefficients physically sensible
_ATTACK_BOUNDS  = (0.2, 4.0)
_DEFENCE_BOUNDS = (0.2, 4.0)
_HOME_ADV_BOUNDS = (1.0, 1.8)
_RHO_BOUNDS      = (-0.5, 0.0)


# ── Dixon-Coles likelihood helpers ───────────────────────────────────────────

def _tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    """
    Low-scoring correction factor τ.
    Adjusts joint probability for scorelines 0-0, 1-0, 0-1, 1-1.
    """
    if   home_goals == 0 and away_goals == 0:
        return 1.0 - lam * mu * rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + lam * rho
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function P(X=k | λ)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except (OverflowError, ValueError):
        return 1e-10


def _match_log_likelihood(
    home_goals: int,
    away_goals: int,
    attack_home: float,
    defence_home: float,
    attack_away: float,
    defence_away: float,
    home_advantage: float,
    rho: float,
) -> float:
    """Log-likelihood contribution of a single match."""
    lam = attack_home * defence_away * home_advantage   # home expected goals
    mu  = attack_away * defence_home                    # away expected goals

    p_home = _poisson_pmf(home_goals, lam)
    p_away = _poisson_pmf(away_goals, mu)
    tau    = _tau(home_goals, away_goals, lam, mu, rho)

    prob = p_home * p_away * tau
    return math.log(max(prob, 1e-10))


# ── Training data loader ──────────────────────────────────────────────────────

def _load_training_data() -> List[Dict[str, Any]]:
    """
    Pull resolved matches from the predictions DB.

    Returns a list of dicts:
        [{"home_team", "away_team", "home_goals", "away_goals"}, ...]
    """
    try:
        import database as db
        rows = db.get_training_rows()
        return rows
    except Exception as exc:
        logger.warning("Failed to load training data from DB: %s", exc)
        return []


# ── MLE optimisation ─────────────────────────────────────────────────────────

def _fit_coefficients(
    matches: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Fit Dixon-Coles attack/defence coefficients via MLE.

    Uses scipy.optimize.minimize with L-BFGS-B (bounded, gradient-based).
    Falls back to None if scipy is unavailable or optimisation fails.
    """
    try:
        from scipy.optimize import minimize
        import numpy as np
    except ImportError:
        logger.error("scipy/numpy not installed — cannot fit model. Run: pip install scipy numpy")
        return None

    # Build sorted team list for stable parameter ordering
    teams = sorted({m["home_team"] for m in matches} | {m["away_team"] for m in matches})
    n     = len(teams)
    idx   = {t: i for i, t in enumerate(teams)}

    logger.info("Fitting Dixon-Coles model: %d teams, %d matches", n, len(matches))

    # Parameter layout: [attack_0..n-1, defence_0..n-1, home_advantage, rho]
    # Initialise attack=1.0, defence=1.0, home_advantage=1.2, rho=-0.1
    x0 = np.ones(n * 2 + 2)
    x0[-2] = 1.2   # home_advantage
    x0[-1] = -0.1  # rho

    bounds = (
        [_ATTACK_BOUNDS]  * n +
        [_DEFENCE_BOUNDS] * n +
        [_HOME_ADV_BOUNDS, _RHO_BOUNDS]
    )

    # Identifiability constraint: fix average attack = 1.0 via soft penalty
    # (avoids the scale degeneracy where attack↑ and defence↑ give same fit)
    PENALTY = 1000.0

    def neg_log_likelihood(x: np.ndarray) -> float:
        attacks   = x[:n]
        defences  = x[n:2*n]
        home_adv  = x[-2]
        rho       = x[-1]

        total_ll = 0.0
        for m in matches:
            hi = idx[m["home_team"]]
            ai = idx[m["away_team"]]
            total_ll += _match_log_likelihood(
                m["home_goals"], m["away_goals"],
                attacks[hi], defences[hi],
                attacks[ai], defences[ai],
                home_adv, rho,
            )

        # Soft identifiability penalty: mean(attacks) → 1.0
        penalty = PENALTY * (np.mean(attacks) - 1.0) ** 2
        return -total_ll + penalty

    try:
        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9},
        )
    except Exception as exc:
        logger.error("Optimisation failed: %s", exc)
        return None

    if not result.success:
        logger.warning("Optimisation did not fully converge: %s", result.message)
        # Still use the result — partial convergence is usually fine

    x      = result.x
    attacks   = x[:n]
    defences  = x[n:2*n]
    home_adv  = float(x[-2])
    rho       = float(x[-1])

    team_coeffs = {
        team: {
            "attack":  round(float(attacks[i]),  4),
            "defence": round(float(defences[i]), 4),
        }
        for i, team in enumerate(teams)
    }

    logger.info(
        "Fit complete — home_advantage=%.3f, rho=%.4f, log-likelihood=%.1f",
        home_adv, rho, -result.fun,
    )

    return {
        "fitted":         True,
        "trained_on":     len(matches),
        "trained_at":     datetime.utcnow().isoformat(timespec="seconds"),
        "home_advantage": round(home_adv, 4),
        "rho":            round(rho, 4),
        "teams":          team_coeffs,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def train(force: bool = False) -> Dict[str, Any]:
    """
    Load training data, fit coefficients, and save to MODEL_COEFFICIENTS_PATH.

    Returns the coefficients dict (fitted or fallback).

    Args:
        force: If True, train even if fewer than MIN_MATCHES matches exist.
               Useful for testing — do NOT use in production.
    """
    matches = _load_training_data()
    n       = len(matches)

    logger.info("Model trainer: %d resolved matches available", n)

    if n < MIN_MATCHES and not force:
        logger.info(
            "Skipping fit — need %d matches, have %d. "
            "Hand-tuned weights remain active.",
            MIN_MATCHES, n,
        )
        result = {
            "fitted":     False,
            "trained_on": n,
            "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
            "min_matches": MIN_MATCHES,
            "message": (
                f"Need {MIN_MATCHES} resolved matches to fit model. "
                f"Currently have {n}. Check back in "
                f"{max(0, MIN_MATCHES - n)} more matches."
            ),
        }
        _save(result)
        return result

    coeffs = _fit_coefficients(matches)

    if coeffs is None:
        logger.error("Fitting failed — hand-tuned weights remain active.")
        result = {
            "fitted":     False,
            "trained_on": n,
            "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
            "message":    "Fitting failed — see server logs.",
        }
        _save(result)
        return result

    _save(coeffs)
    logger.info(
        "Model coefficients saved to %s (%d teams)",
        MODEL_COEFFICIENTS_PATH, len(coeffs.get("teams", {})),
    )
    return coeffs


def load() -> Optional[Dict[str, Any]]:
    """
    Load saved coefficients from disk.
    Returns None if the file doesn't exist or isn't fitted yet.
    """
    if not MODEL_COEFFICIENTS_PATH.exists():
        return None
    try:
        data = json.loads(MODEL_COEFFICIENTS_PATH.read_text(encoding="utf-8"))
        if not data.get("fitted"):
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load model coefficients: %s", exc)
        return None


def get_status() -> Dict[str, Any]:
    """Return model status dict for the /api/model endpoint."""
    if not MODEL_COEFFICIENTS_PATH.exists():
        return {
            "mode":       "hand_tuned",
            "fitted":     False,
            "trained_on": 0,
            "message":    "No training run yet. Nightly trainer runs at 02:30 UTC.",
        }
    try:
        data = json.loads(MODEL_COEFFICIENTS_PATH.read_text(encoding="utf-8"))
        if data.get("fitted"):
            return {
                "mode":         "dixon_coles_mle",
                "fitted":       True,
                "trained_on":   data.get("trained_on", 0),
                "trained_at":   data.get("trained_at"),
                "home_advantage": data.get("home_advantage"),
                "rho":          data.get("rho"),
                "team_count":   len(data.get("teams", {})),
            }
        return {
            "mode":       "hand_tuned",
            "fitted":     False,
            "trained_on": data.get("trained_on", 0),
            "min_matches": data.get("min_matches", MIN_MATCHES),
            "message":    data.get("message", ""),
        }
    except (json.JSONDecodeError, OSError):
        return {"mode": "hand_tuned", "fitted": False}


def _save(data: Dict[str, Any]) -> None:
    try:
        MODEL_COEFFICIENTS_PATH.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.error("Could not save model coefficients: %s", exc)
