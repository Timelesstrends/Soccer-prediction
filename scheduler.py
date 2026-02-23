"""
Scheduler — orchestrates automatic data fetching and dashboard generation.

Uses Python's built-in `sched` module (no external dependencies).
For production, replace with APScheduler or Celery + beat.

Architecture:
    Scheduler
        ├── every 15 min  → fetch odds  → attach value bets → refresh dashboard
        ├── 06:00 & 18:00 → fetch fixtures → rebuild predictions → refresh dashboard
        └── Monday 02:00  → fetch StatsBomb + Understat training data
"""

from __future__ import annotations

import logging
import sched
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional

from cache import DiskCache
from client_football_data import FootballDataClient
from client_odds_api import OddsAPIClient
from client_training_data import StatsBombClient, UnderstatClient
from config import FOOTBALL_DATA, SCHEDULER, SchedulerConfig
from prediction_dashboard import ModelOutput, PredictionDashboard

logger = logging.getLogger(__name__)


class PredictionScheduler:
    """
    Runs fetch-and-refresh cycles on a background thread.

    Usage:
        scheduler = PredictionScheduler(dashboard, output_dir="predictions")
        scheduler.start()          # non-blocking
        ...
        scheduler.stop()
    """

    def __init__(
        self,
        dashboard: PredictionDashboard,
        config: SchedulerConfig = SCHEDULER,
        output_dir: str = "predictions",
    ) -> None:
        self._dashboard  = dashboard
        self._config     = config
        self._output_dir = Path(output_dir)
        self._cache      = DiskCache()

        # Shared state: latest predictions (protected by a lock)
        self._predictions: List[ModelOutput] = []
        self._lock = threading.Lock()

        # Clients
        self._fixtures_client = FootballDataClient(cache=self._cache)
        self._odds_client     = OddsAPIClient(cache=self._cache)
        self._statsbomb       = StatsBombClient(cache=self._cache)
        self._understat       = UnderstatClient(cache=self._cache)

        # Internal scheduler
        self._scheduler = sched.scheduler(time.monotonic, time.sleep)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler on a background daemon thread."""
        if self._running:
            logger.warning("Scheduler already running.")
            return

        self._running = True
        self._schedule_all()

        self._thread = threading.Thread(
            target=self._run_loop,
            name="PredictionScheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("Scheduler started.")

    def stop(self) -> None:
        """Signal the scheduler to stop after the current cycle."""
        self._running = False
        # Cancel all pending events
        for event in list(self._scheduler.queue):
            try:
                self._scheduler.cancel(event)
            except ValueError:
                pass
        logger.info("Scheduler stopped.")

    def run_now(self) -> List[ModelOutput]:
        """
        Trigger an immediate full fetch cycle (blocking).
        Useful for the first run on startup or manual refreshes.
        """
        logger.info("Running immediate full fetch cycle.")
        predictions = self._fetch_fixtures()
        predictions = self._attach_odds(predictions)
        self._update_predictions(predictions)
        self._refresh_dashboard()
        return predictions

    # ── Scheduled tasks ───────────────────────────────────────────────────────

    def _schedule_all(self) -> None:
        """Queue all recurring tasks."""
        # Odds refresh — every N minutes
        self._schedule_repeating(
            interval_seconds=self._config.odds_fetch_interval_minutes * 60,
            task=self._task_refresh_odds,
            name="odds_refresh",
        )

        # Fixtures fetch — at configured UTC hours
        for hour, minute in self._config.fixtures_fetch_times:
            self._schedule_daily(hour, minute, self._task_fetch_fixtures)

        # Training data — weekly on the configured day
        self._schedule_weekly(
            day=self._config.training_data_fetch_day,
            task=self._task_fetch_training_data,
        )

    def _task_fetch_fixtures(self) -> None:
        """Fetch fresh fixtures and rebuild predictions."""
        logger.info("[Scheduler] Fetching fixtures.")
        try:
            predictions = self._fetch_fixtures()
            predictions = self._attach_odds(predictions)
            self._update_predictions(predictions)
            self._refresh_dashboard()
        except Exception as exc:
            logger.error("[Scheduler] Fixture fetch failed: %s", exc)

    def _task_refresh_odds(self) -> None:
        """Re-fetch odds for current predictions and refresh dashboard."""
        logger.info("[Scheduler] Refreshing odds.")
        try:
            with self._lock:
                predictions = list(self._predictions)

            if not predictions:
                logger.info("[Scheduler] No predictions to update odds for.")
                return

            # Invalidate odds cache so we get fresh prices
            self._cache.clear_source("odds")
            predictions = self._attach_odds(predictions)
            self._update_predictions(predictions)
            self._refresh_dashboard()
        except Exception as exc:
            logger.error("[Scheduler] Odds refresh failed: %s", exc)

    def _task_fetch_training_data(self) -> None:
        """Pull latest StatsBomb and Understat data for model retraining."""
        logger.info("[Scheduler] Fetching training data.")
        try:
            # StatsBomb: Premier League, season 3 (example open dataset)
            comps = self._statsbomb.get_competitions()
            epl = comps[
                (comps["competition_name"] == "Premier League")
                & (comps["country_name"] == "England")
            ]
            if not epl.empty:
                comp_id   = int(epl.iloc[0]["competition_id"])
                season_id = int(epl.iloc[0]["season_id"])
                df = self._statsbomb.build_xg_training_data(comp_id, season_id)
                out = self._output_dir / "training_statsbomb.csv"
                df.to_csv(out, index=False)
                logger.info("[Scheduler] StatsBomb training data → %s", out)

            # Understat: current Premier League season
            year = datetime.now().year
            df_xg = self._understat.get_team_xg("Premier League", year)
            if not df_xg.empty:
                out = self._output_dir / "training_understat_xg.csv"
                df_xg.to_csv(out, index=False)
                logger.info("[Scheduler] Understat xG data → %s", out)

        except Exception as exc:
            logger.error("[Scheduler] Training data fetch failed: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_fixtures(self) -> List[ModelOutput]:
        """Pull fixtures from all configured leagues."""
        predictions: List[ModelOutput] = []
        for league_code in FOOTBALL_DATA.league_codes:
            try:
                preds = self._fixtures_client.build_model_outputs(league_code)
                predictions.extend(preds)
                logger.info(
                    "Fetched %d fixtures for league %d", len(preds), league_code
                )
            except RuntimeError as exc:
                logger.warning(
                    "Skipping league %s — fetch error: %s", league_code, exc
                )
        return predictions

    def _attach_odds(self, predictions: List[ModelOutput]) -> List[ModelOutput]:
        """Try to attach value bets; log and continue on failure."""
        try:
            return self._odds_client.attach_value_bets(predictions)
        except RuntimeError as exc:
            logger.warning("Odds attachment failed: %s — continuing without.", exc)
            return predictions

    def _update_predictions(self, predictions: List[ModelOutput]) -> None:
        with self._lock:
            self._predictions = predictions
        logger.info("Predictions updated: %d matches", len(predictions))

    def _refresh_dashboard(self) -> None:
        """Re-render all outputs from the latest predictions."""
        with self._lock:
            predictions = list(self._predictions)

        if not predictions:
            return

        try:
            self._dashboard.generate_html_dashboard(predictions)
            self._dashboard.export_json(predictions)
            self._dashboard.export_csv(predictions)
            logger.info("[Dashboard] Refreshed — %d predictions", len(predictions))
        except Exception as exc:
            logger.error("[Dashboard] Refresh failed: %s", exc)

    # ── Scheduling helpers ────────────────────────────────────────────────────

    def _schedule_repeating(
        self, interval_seconds: float, task: Callable, name: str
    ) -> None:
        """Schedule a task to run every `interval_seconds` seconds."""

        def _wrapper() -> None:
            if not self._running:
                return
            logger.debug("[Scheduler] Running task: %s", name)
            task()
            # Re-schedule itself
            self._scheduler.enter(interval_seconds, 1, _wrapper)

        self._scheduler.enter(interval_seconds, 1, _wrapper)

    def _schedule_daily(self, hour: int, minute: int, task: Callable) -> None:
        """Schedule a task to run every day at (hour, minute) UTC."""
        def _wrapper() -> None:
            if not self._running:
                return
            task()
            # Schedule next occurrence 24 hours from now
            self._scheduler.enter(86_400, 1, _wrapper)

        delay = self._seconds_until(hour, minute)
        self._scheduler.enter(delay, 1, _wrapper)
        logger.info(
            "[Scheduler] Daily task scheduled at %02d:%02d UTC (in %.0fs)",
            hour, minute, delay,
        )

    def _schedule_weekly(self, day: str, task: Callable) -> None:
        """Schedule a task to run once per week on the given day name."""
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        }
        target_weekday = day_map.get(day.lower(), 0)

        def _wrapper() -> None:
            if not self._running:
                return
            task()
            self._scheduler.enter(7 * 86_400, 1, _wrapper)

        now = datetime.utcnow()
        days_ahead = (target_weekday - now.weekday()) % 7 or 7
        delay = days_ahead * 86_400 - now.hour * 3600 - now.minute * 60
        self._scheduler.enter(max(delay, 1), 1, _wrapper)

    @staticmethod
    def _seconds_until(hour: int, minute: int) -> float:
        """Seconds from now until the next occurrence of (hour, minute) UTC."""
        now    = datetime.utcnow()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    def _run_loop(self) -> None:
        """Main loop — runs the sched scheduler until stopped."""
        logger.info("[Scheduler] Background thread running.")
        while self._running:
            self._scheduler.run(blocking=False)
            time.sleep(1)
        logger.info("[Scheduler] Background thread exiting.")
