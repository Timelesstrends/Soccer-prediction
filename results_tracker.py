"""
results_tracker.py — Fetches finished match results from football-data.org
and reconciles them against logged predictions in the database.

Flow:
    1. Read all unresolved predictions (match_date < today)
    2. Per league, fetch FINISHED matches from the API
    3. Match by match_id and log the actual score
    4. Rebuild performance stats
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional

from database import (
    get_unresolved_predictions,
    log_result,
    rebuild_performance,
)

logger = logging.getLogger(__name__)

# Map full league name → football-data.org competition code
_LEAGUE_CODES: Dict[str, str] = {
    "Premier League": "PL",
    "La Liga":        "PD",
    "Bundesliga":     "BL1",
    "Serie A":        "SA",
    "Ligue 1":        "FL1",
}


class ResultsTracker:
    """
    Reconciles logged predictions against actual match results.

    Usage:
        tracker = ResultsTracker(football_client)
        summary = tracker.run()
        # {"checked": 40, "resolved": 12, "skipped": 28}
    """

    def __init__(self, football_client) -> None:
        self._client = football_client

    def run(self, rebuild_stats: bool = True) -> Dict[str, int]:
        """
        Fetch results for all unresolved predictions whose match date has passed.
        Returns a summary dict: checked / resolved / skipped.
        """
        today = str(date.today())
        unresolved = get_unresolved_predictions(before_date=today)

        if not unresolved:
            logger.info("ResultsTracker: nothing to resolve.")
            return {"checked": 0, "resolved": 0, "skipped": 0}

        logger.info("ResultsTracker: %d unresolved predictions", len(unresolved))

        # Group by league to minimise API calls
        by_league: Dict[str, List[Dict]] = {}
        for pred in unresolved:
            by_league.setdefault(pred["league"], []).append(pred)

        resolved = 0
        skipped  = 0

        for league_name, preds in by_league.items():
            code = _LEAGUE_CODES.get(league_name)
            if not code:
                logger.warning("Unknown league '%s' — skipping", league_name)
                skipped += len(preds)
                continue

            dates     = [p["match_date"] for p in preds]
            date_from = date.fromisoformat(min(dates))
            date_to   = date.fromisoformat(max(dates))

            try:
                finished = self._client.get_matches(
                    code,
                    date_from = date_from,
                    date_to   = date_to,
                    status    = "FINISHED",
                )
            except Exception as exc:
                logger.warning("Results fetch failed for %s: %s", league_name, exc)
                skipped += len(preds)
                continue

            result_map = _build_result_map(finished)

            for pred in preds:
                mid = pred["match_id"]
                if mid in result_map:
                    hg, ag = result_map[mid]
                    try:
                        if log_result(mid, hg, ag):
                            resolved += 1
                            logger.info(
                                "  Resolved %s vs %s: %d-%d",
                                pred["home_team"], pred["away_team"], hg, ag,
                            )
                        else:
                            skipped += 1
                    except Exception as exc:
                        logger.warning("Error logging result %s: %s", mid, exc)
                        skipped += 1
                else:
                    skipped += 1
                    logger.debug(
                        "  No result yet for %s vs %s (%s)",
                        pred["home_team"], pred["away_team"], pred["match_date"],
                    )

        logger.info(
            "ResultsTracker done: %d resolved, %d skipped / %d total",
            resolved, skipped, len(unresolved),
        )

        if resolved > 0 and rebuild_stats:
            try:
                rebuild_performance()
            except Exception as exc:
                logger.error("Performance rebuild failed: %s", exc)

        return {"checked": len(unresolved), "resolved": resolved, "skipped": skipped}


def _build_result_map(matches: List[Dict]) -> Dict[str, tuple]:
    """Build {match_id: (home_goals, away_goals)} from API match dicts."""
    out: Dict[str, tuple] = {}
    for m in matches:
        try:
            mid  = str(m["id"])
            ft   = m["score"]["fullTime"]
            home = ft["home"]
            away = ft["away"]
            if home is not None and away is not None:
                out[mid] = (int(home), int(away))
        except (KeyError, TypeError, ValueError):
            continue
    return out
