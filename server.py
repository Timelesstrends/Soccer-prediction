"""
Soccer Prediction Dashboard — production-ready web server.

Development (Termux):
    python server.py            # mock data, no API calls
    python server.py --live     # real API data

Production (Render / any host):
    gunicorn "server:create_app()" --workers 2 --bind 0.0.0.0:$PORT

Environment variables (set in Render dashboard, never hardcode):
    FOOTBALL_DATA_KEY   — football-data.org API key
    ODDS_API_KEY        — the-odds-api.com API key
    DASHBOARD_PASSWORD  — password to access the dashboard (optional)
    LIVE_MODE           — set to "true" to enable live API fetching
    PORT                — port number (Render sets this automatically)
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import threading
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request, Response

from prediction_dashboard import ModelOutput, MatchResult, PredictionDashboard, TeamForm, ValueBet, SummaryStats
import database as db
from results_tracker import ResultsTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")


# ── Password protection ───────────────────────────────────────────────────────

def _check_auth(username: str, password: str) -> bool:
    """Validate credentials against the DASHBOARD_PASSWORD env variable."""
    expected = os.getenv("DASHBOARD_PASSWORD", "")
    if not expected:
        return True   # no password set → open access
    return username == "admin" and password == expected


def _request_auth_response() -> Response:
    """Tell the browser to show a login prompt."""
    return Response(
        "Authentication required.",
        401,
        {"WWW-Authenticate": 'Basic realm="Soccer Dashboard"'},
    )


def requires_auth(f: Callable) -> Callable:
    """Decorator — applies HTTP Basic Auth if DASHBOARD_PASSWORD is set."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not os.getenv("DASHBOARD_PASSWORD"):
            return f(*args, **kwargs)   # skip auth if no password configured
        auth = request.authorization
        if not auth or not _check_auth(auth.username, auth.password):
            return _request_auth_response()
        return f(*args, **kwargs)
    return decorated


# ── Shared state ──────────────────────────────────────────────────────────────

_predictions: List[ModelOutput] = []
_last_updated: str = "Never"
_lock = threading.Lock()

# Phase status — broadcast to SSE clients so UI can show live progress
# Keys: "phase" (1|2|done), "step" (human label), "leagues_done" (list)
_phase_status: dict = {"phase": 0, "step": "Initialising...", "leagues_done": [], "stale": False}
_status_lock = threading.Lock()

def _set_status(phase: int, step: str, leagues_done: list = None, stale: bool = False) -> None:
    global _phase_status
    with _status_lock:
        _phase_status = {
            "phase":        phase,
            "step":         step,
            "leagues_done": leagues_done if leagues_done is not None else _phase_status["leagues_done"],
            "stale":        stale,
        }


def update_predictions(predictions: List[ModelOutput]) -> None:
    """Thread-safe update of the shared prediction state."""
    global _last_updated
    with _lock:
        _predictions.clear()
        _predictions.extend(predictions)
        _last_updated = datetime.now().strftime("%H:%M:%S")
    logger.info("Predictions updated: %d matches", len(predictions))


# ── Flask app factory (required for gunicorn) ─────────────────────────────────

def create_app() -> Flask:
    """
    Application factory pattern.
    Gunicorn calls this to get the Flask app instance.
    Live data loading is triggered by the LIVE_MODE env variable.
    """
    app = Flask(__name__)

    # Initialise SQLite database (creates file + tables on first run)
    try:
        db.init_db()
    except Exception as _db_exc:
        logger.warning("DB init failed: %s", _db_exc)

    # Force ASCII-safe JSON and HTML encoding so special characters
    # in team names (accents etc.) never cause UnicodeEncodeError
    app.config["JSON_AS_ASCII"] = True
    app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"
    app.jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": True}

    # Force all responses through ASCII-safe encoding
    # This prevents UnicodeEncodeError from accented team names
    @app.after_request
    def ensure_ascii_response(response):
        if response.content_type and "text/html" in response.content_type:
            try:
                text = response.get_data(as_text=True)
                safe = text.encode("ascii", errors="xmlcharrefreplace").decode("ascii")
                response.set_data(safe)
            except Exception:
                pass
        return response

    # Register routes
    _register_routes(app)

    # If running under gunicorn with LIVE_MODE=true, load live data
    if os.getenv("LIVE_MODE", "").lower() == "true":
        _start_live_mode(app)
    else:
        logger.info("LIVE_MODE not set — loading mock data.")
        update_predictions(_build_mock_predictions())

    return app


def _register_routes(app: Flask) -> None:
    """Attach all URL routes to the app."""

    @app.route("/")
    @requires_auth
    def index() -> str:
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/predictions")
    @requires_auth
    def api_predictions() -> Response:
        with _lock:
            predictions = list(_predictions)
            last_updated = _last_updated

        stats = SummaryStats.from_predictions(predictions)
        return jsonify({
            "last_updated": last_updated,
            "stats": {
                "total":           stats.total,
                "high_confidence": stats.high_confidence,
                "avg_goals":       round(stats.avg_goals, 2),
                "value_bets":      stats.value_bets,
            },
            "predictions": [_serialise_prediction(p) for p in predictions],
        })

    @app.route("/api/stream")
    @requires_auth
    def api_stream() -> Response:
        """
        Server-Sent Events endpoint.
        Pushes each prediction individually as it arrives so the browser
        can render matches one-by-one without waiting for all data.
        Also pushes a stats update after every match so the header stays current.
        """
        import json as _json

        def generate():
            sent_ids = set()
            import json as _json

            # Send current phase status immediately on connect
            with _status_lock:
                status = dict(_phase_status)
            yield "event: phase\ndata: " + _json.dumps(status, ensure_ascii=True) + "\n\n"

            # Send whatever is already loaded immediately
            with _lock:
                current = list(_predictions)
                last    = _last_updated

            for pred in current:
                data = _serialise_prediction(pred)
                yield "event: match\ndata: " + _json.dumps(data, ensure_ascii=True) + "\n\n"
                sent_ids.add(pred.match_id)

            # Send initial stats
            stats = SummaryStats.from_predictions(current)
            stats_payload = {
                "last_updated": last,
                "total":           stats.total,
                "high_confidence": stats.high_confidence,
                "avg_goals":       round(stats.avg_goals, 2),
                "value_bets":      stats.value_bets,
            }
            yield "event: stats\ndata: " + _json.dumps(stats_payload, ensure_ascii=True) + "\n\n"

            # Keep watching for new predictions (form loading in Phase 2)
            import time
            last_phase_sent = {}
            for _ in range(240):   # poll for up to ~4 minutes (240 x 1s)
                time.sleep(1)
                with _lock:
                    current = list(_predictions)
                    last    = _last_updated

                # Emit phase status if it changed
                with _status_lock:
                    current_status = dict(_phase_status)
                if current_status != last_phase_sent:
                    yield "event: phase\ndata: " + _json.dumps(current_status, ensure_ascii=True) + "\n\n"
                    last_phase_sent = current_status

                new_preds = [p for p in current if p.match_id not in sent_ids]
                updated   = [p for p in current if p.match_id in sent_ids
                             and (p.home_form or p.away_form)]

                for pred in new_preds:
                    data = _serialise_prediction(pred)
                    yield "event: match\ndata: " + _json.dumps(data, ensure_ascii=True) + "\n\n"
                    sent_ids.add(pred.match_id)

                if new_preds or updated:
                    stats = SummaryStats.from_predictions(current)
                    stats_payload = {
                        "last_updated": last,
                        "total":           stats.total,
                        "high_confidence": stats.high_confidence,
                        "avg_goals":       round(stats.avg_goals, 2),
                        "value_bets":      stats.value_bets,
                    }
                    yield "event: stats\ndata: " + _json.dumps(stats_payload, ensure_ascii=True) + "\n\n"

                    # Re-send updated predictions so browser refreshes form badges
                    for pred in updated:
                        data = _serialise_prediction(pred)
                        yield "event: update\ndata: " + _json.dumps(data, ensure_ascii=True) + "\n\n"

                # Stop polling once phase 3 (all done) is reached
                if current_status.get("phase") == 3:
                    break

            yield "event: done\ndata: {}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/refresh")
    @requires_auth
    def api_refresh() -> Response:
        with _lock:
            count = len(_predictions)
        return jsonify({"status": "ok", "predictions_loaded": count})

    @app.route("/api/status")
    @requires_auth
    def api_status() -> Response:
        """Return current loading phase status for the UI progress overlay."""
        with _status_lock:
            status = dict(_phase_status)
        with _lock:
            count = len(_predictions)
        status["predictions_loaded"] = count
        return jsonify(status)

    @app.route("/api/refresh/force")
    @requires_auth
    def api_refresh_force() -> Response:
        """
        Force a fresh data fetch by invalidating fixtures and odds cache,
        then re-triggering Phase 1. Form/H2H cache is preserved (slow to refetch).
        """
        from cache import DiskCache
        from config import FOOTBALL_DATA
        cache = DiskCache()
        # Only clear fixtures and odds — not form/H2H (too slow)
        cache.clear_source("fixtures")
        cache.clear_source("odds")
        _set_status(0, "Manual refresh triggered...", [])
        if os.getenv("LIVE_MODE", "").lower() == "true":
            _start_live_mode(app)
        return jsonify({"status": "ok", "message": "Cache cleared. Refresh started."})

    @app.route("/api/performance")
    @requires_auth
    def api_performance() -> Response:
        """Return model accuracy and P&L stats from the database."""
        try:
            perf    = db.get_performance()
            summary = db.get_db_summary()
            return jsonify({"status": "ok", "performance": perf, "db": summary})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/results/recent")
    @requires_auth
    def api_results_recent() -> Response:
        """Return the last 100 resolved predictions for the history browser."""
        try:
            results = db.get_recent_results(limit=100)
            return jsonify({"status": "ok", "count": len(results), "results": results})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/results/refresh")
    @requires_auth
    def api_results_refresh() -> Response:
        """Manually trigger a results fetch (useful for testing)."""
        from client_football_data import FootballDataClient
        from results_tracker import ResultsTracker
        try:
            tracker = ResultsTracker(FootballDataClient())
            summary = tracker.run()
            return jsonify({"status": "ok", **summary})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/history")
    @requires_auth
    def history_page() -> str:
        return render_template_string(HISTORY_TEMPLATE)

    @app.route("/favicon.ico")
    def favicon() -> Response:
        return Response(status=204)

    @app.route("/health")
    def health() -> Response:
        with _lock:
            count = len(_predictions)
        return jsonify({"status": "healthy", "predictions": count})

def _start_live_mode(app: Flask) -> None:
    """
    Fetch live data in two phases so the dashboard shows data quickly.

    Phase 1 (fast, ~30s per league):
        Fixtures + standings (total/home/away) + odds.
        Uses season strength + venue splits only — no form API calls.
        Browser shows real xG values immediately.

    Phase 2 (slow, ~6s per team):
        Adds weighted form + H2H for each match via build_model_outputs().
        Replaces Phase 1 predictions with fully-blended model outputs
        progressively after each league completes.
    """
    from client_football_data import FootballDataClient
    from client_odds_api import OddsAPIClient
    from scheduler import PredictionScheduler
    from config import FOOTBALL_DATA
    from cache import DiskCache

    dashboard = PredictionDashboard(output_dir="predictions")

    def _initialise() -> None:
        football_client = FootballDataClient()
        odds_client     = OddsAPIClient()
        cache           = DiskCache()

        _set_status(1, "Checking cache for instant load...", [])

        # ── Stale-while-revalidate: serve cached predictions immediately ──
        # If we have any stale fixtures cached, build predictions from them
        # right now so the browser gets data in <1s, then refresh in the background.
        stale_predictions: List[ModelOutput] = []
        from datetime import date, timedelta
        today     = str(date.today())
        tomorrow  = str(date.today() + timedelta(days=1))

        for league_code in FOOTBALL_DATA.league_codes:
            cache_key = f"{league_code}_{today}"
            stale = cache.get_stale("fixtures", cache_key)
            if stale:
                try:
                    standings = cache.get_stale("fixtures", f"standings_{league_code}")
                    if standings and "total" in standings:
                        strength_map = football_client._build_strength_map(standings["total"])
                        home_map     = football_client._build_venue_split_map(standings["home"], "home")
                        away_map     = football_client._build_venue_split_map(standings["away"], "away")
                        pos_map      = football_client._build_position_map(standings["total"])
                        for match in stale:
                            try:
                                stale_predictions.append(
                                    football_client._match_to_output(
                                        match, league_code, strength_map, home_map, away_map, pos_map
                                    )
                                )
                            except Exception:
                                continue
                except Exception:
                    pass

        if stale_predictions:
            try:
                stale_predictions = odds_client.attach_value_bets(stale_predictions)
            except Exception:
                pass
            update_predictions(stale_predictions)
            _set_status(1, "Showing cached data \u2014 refreshing in background...", [], stale=True)
            logger.info("Stale cache served: %d predictions shown instantly", len(stale_predictions))

        # ── Phase 1: fresh fixtures + standings ───────────────────────────
        _set_status(1, "Fetching live fixtures...", [])
        logger.info("Phase 1: fetching fixtures and standings...")
        all_predictions: List[ModelOutput] = []
        leagues_done: List[str] = []

        league_labels = {"PL": "Premier League", "PD": "La Liga", "BL1": "Bundesliga",
                         "SA": "Serie A", "FL1": "Ligue 1"}

        for league_code in FOOTBALL_DATA.league_codes:
            try:
                _set_status(1, f"Fetching {league_labels.get(league_code, league_code)}...", leagues_done[:])
                matches   = football_client.get_matches(league_code)
                standings = football_client.get_standings(league_code)

                strength_map = football_client._build_strength_map(standings["total"])
                home_map     = football_client._build_venue_split_map(standings["home"], "home")
                away_map     = football_client._build_venue_split_map(standings["away"], "away")
                pos_map      = football_client._build_position_map(standings["total"])

                for match in matches:
                    try:
                        all_predictions.append(
                            football_client._match_to_output(
                                match, league_code, strength_map, home_map, away_map, pos_map
                            )
                        )
                    except (KeyError, TypeError):
                        continue

                leagues_done.append(league_labels.get(league_code, league_code))
                _set_status(1, f"Loaded {league_labels.get(league_code, league_code)}", leagues_done[:])
                logger.info("  Phase 1 %s: %d fixtures", league_code, len(matches))
            except Exception as exc:
                logger.warning("  Phase 1 %s skipped — %s", league_code, exc)

        _set_status(1, "Attaching odds...", leagues_done[:])
        try:
            all_predictions = odds_client.attach_value_bets(all_predictions)
        except Exception as exc:
            logger.warning("Odds skipped in Phase 1 — %s", exc)

        update_predictions(all_predictions)
        _set_status(1, "Phase 1 complete", leagues_done[:], stale=False)
        logger.info("Phase 1 complete: %d matches visible in browser", len(all_predictions))

        try:
            db.log_predictions_bulk(all_predictions)
        except Exception as exc:
            logger.warning("DB log failed (Phase 1): %s", exc)

        # ── Phase 2: full model per league (form + H2H) ───────────────────
        logger.info("Phase 2: fetching form + H2H (runs in background)...")
        _set_status(2, "Fetching team form + H2H history...", [])
        phase2_all: List[ModelOutput] = []
        p2_leagues_done: List[str] = []

        for league_code in FOOTBALL_DATA.league_codes:
            try:
                _set_status(2, f"Form: {league_labels.get(league_code, league_code)}...", p2_leagues_done[:])
                league_preds = football_client.build_model_outputs(league_code)
                phase2_all.extend(league_preds)
                p2_leagues_done.append(league_labels.get(league_code, league_code))
                _set_status(2, f"Form loaded: {league_labels.get(league_code, league_code)}", p2_leagues_done[:])
                logger.info("  Phase 2 done for %s (%d matches)", league_code, len(league_preds))

                phase2_ids = {p.match_id for p in phase2_all}
                merged     = phase2_all + [
                    p for p in all_predictions if p.match_id not in phase2_ids
                ]

                try:
                    merged = odds_client.attach_value_bets(merged)
                except Exception:
                    pass

                update_predictions(merged)

                try:
                    db.log_predictions_bulk(league_preds)
                except Exception as exc:
                    logger.warning("DB log failed (Phase 2 %s): %s", league_code, exc)

            except Exception as exc:
                logger.warning("  Phase 2 failed for %s — %s", league_code, exc)

        _set_status(3, "All data loaded", p2_leagues_done[:])
        logger.info("Phase 2 complete: full model active for all leagues")

        # ── Start scheduler ───────────────────────────────────────────────
        scheduler = PredictionScheduler(dashboard, output_dir="predictions")
        original_update = scheduler._update_predictions

        def hooked_update(preds: List[ModelOutput]) -> None:
            original_update(preds)
            update_predictions(preds)

        scheduler._update_predictions = hooked_update
        scheduler.start()
        logger.info("Scheduler started.")

    thread = threading.Thread(target=_initialise, daemon=True, name="LiveInit")
    thread.start()


# _merge_form removed — Phase 2 now fully replaces Phase 1 predictions
# rather than patching them, so this helper is no longer needed.


# ── Serialisation helper ──────────────────────────────────────────────────────

def _serialise_prediction(pred: ModelOutput) -> dict:
    def _serialise_form(form) -> Optional[dict]:
        if form is None:
            return None
        return {
            "team_name":     form.team_name,
            "results":       form.results,
            "goals_scored":  form.goals_scored,
            "goals_conceded": form.goals_conceded,
            "form_string":   form.form_string,
            "momentum":      form.momentum,
            "avg_scored":    form.avg_scored,
            "avg_conceded":  form.avg_conceded,
            "points":        form.points,
        }

    def _serialise_split(split) -> Optional[dict]:
        if split is None:
            return None
        return {
            "venue":        split.venue,
            "played":       split.played,
            "avg_scored":   split.avg_scored,
            "avg_conceded": split.avg_conceded,
            "strength":     split.strength_rating,
        }

    def _serialise_h2h(h2h) -> Optional[dict]:
        if h2h is None:
            return None
        return {
            "matches_played":  h2h.matches_played,
            "home_wins":       h2h.home_wins,
            "draws":           h2h.draws,
            "away_wins":       h2h.away_wins,
            "avg_goals_total": h2h.avg_goals_total,
            "home_avg_scored": h2h.home_avg_scored,
            "away_avg_scored": h2h.away_avg_scored,
            "home_win_rate":   h2h.home_win_rate,
            "away_win_rate":   h2h.away_win_rate,
            "home_dominance":  h2h.home_dominance,
            "is_high_scoring": h2h.is_high_scoring,
        }

    return {
        "match_id":                pred.match_id,
        "date":                    pred.date,
        "league":                  pred.league,
        "home_team":               pred.home_team,
        "away_team":               pred.away_team,
        "predicted_goals":         round(pred.predicted_goals, 2),
        "over_prob":               round(pred.over_prob, 4),
        "under_prob":              round(pred.under_prob, 4),
        "confidence":              pred.confidence.value,
        "goal_distribution_items": pred.goal_distribution_items,
        "expected_home_goals":     round(pred.expected_home_goals, 2),
        "expected_away_goals":     round(pred.expected_away_goals, 2),
        "home_form":               _serialise_form(pred.home_form),
        "away_form":               _serialise_form(pred.away_form),
        "home_split":              _serialise_split(pred.home_split),
        "away_split":              _serialise_split(pred.away_split),
        "h2h":                     _serialise_h2h(pred.h2h),
        "match_result": {
            "home_prob":         pred.match_result.home_prob,
            "draw_prob":         pred.match_result.draw_prob,
            "away_prob":         pred.match_result.away_prob,
            "outcome":           pred.match_result.outcome,
            "outcome_confidence": pred.match_result.outcome_confidence,
            "most_likely_score": list(pred.match_result.most_likely_score),
            "most_likely_prob":  pred.match_result.most_likely_prob,
            "home_position":     pred.match_result.home_position,
            "away_position":     pred.match_result.away_position,
        } if pred.match_result else None,
        "value_bet": {
            "recommendation": pred.value_bet.recommendation,
            "edge":           pred.value_bet.edge,
            "kelly_stake":    pred.value_bet.kelly_stake,
        } if pred.value_bet else None,
    }


# ── Mock data ─────────────────────────────────────────────────────────────────

def _build_mock_predictions() -> List[ModelOutput]:
    return [
        ModelOutput(
            match_id="EPL_001", date="2025-08-16", league="Premier League",
            home_team="Arsenal", away_team="Chelsea",
            expected_home_goals=1.8, expected_away_goals=1.3,
            home_strength=1.45, away_strength=1.20,
            goal_distribution={
                "exactly_0": 0.05, "exactly_1": 0.14, "exactly_2": 0.22,
                "exactly_3": 0.25, "exactly_4": 0.18, "exactly_5": 0.10,
                "exactly_6": 0.06,
            },
            value_bet=ValueBet(
                recommendation="Over 2.5 Goals @ 1.85 (Pinnacle)",
                edge=0.14, kelly_stake=8.5,
            ),
            match_result=MatchResult(
                home_prob=0.52, draw_prob=0.24, away_prob=0.24,
                most_likely_score=(2, 1), most_likely_prob=0.11,
                home_position=2, away_position=6,
            ),
            home_form=TeamForm(
                team_name="Arsenal",
                results=["W", "W", "D", "W", "W"],
                goals_scored=[2, 3, 1, 2, 2],
                goals_conceded=[0, 1, 1, 0, 1],
            ),
            away_form=TeamForm(
                team_name="Chelsea",
                results=["L", "W", "D", "L", "W"],
                goals_scored=[0, 2, 1, 1, 3],
                goals_conceded=[2, 1, 1, 2, 2],
            ),
        ),
        ModelOutput(
            match_id="LAL_001", date="2025-08-16", league="La Liga",
            home_team="Barcelona", away_team="Real Madrid",
            expected_home_goals=1.5, expected_away_goals=1.4,
            home_strength=1.38, away_strength=1.35,
            goal_distribution={
                "exactly_0": 0.08, "exactly_1": 0.18, "exactly_2": 0.26,
                "exactly_3": 0.23, "exactly_4": 0.14, "exactly_5": 0.07,
                "exactly_6": 0.04,
            },
            value_bet=None,
            match_result=MatchResult(
                home_prob=0.38, draw_prob=0.30, away_prob=0.32,
                most_likely_score=(1, 1), most_likely_prob=0.13,
                home_position=3, away_position=1,
            ),
            home_form=TeamForm(
                team_name="Barcelona",
                results=["W", "D", "W", "L", "W"],
                goals_scored=[3, 1, 2, 0, 2],
                goals_conceded=[1, 1, 0, 1, 1],
            ),
            away_form=TeamForm(
                team_name="Real Madrid",
                results=["W", "W", "W", "D", "L"],
                goals_scored=[2, 3, 1, 1, 0],
                goals_conceded=[0, 1, 0, 1, 2],
            ),
        ),
        ModelOutput(
            match_id="BUN_001", date="2025-08-17", league="Bundesliga",
            home_team="Bayern Munich", away_team="Borussia Dortmund",
            expected_home_goals=2.2, expected_away_goals=1.6,
            home_strength=1.80, away_strength=1.42,
            goal_distribution={
                "exactly_0": 0.03, "exactly_1": 0.09, "exactly_2": 0.18,
                "exactly_3": 0.24, "exactly_4": 0.22, "exactly_5": 0.14,
                "exactly_6": 0.10,
            },
            value_bet=ValueBet(
                recommendation="Over 3.5 Goals @ 2.10 (Bet365)",
                edge=0.08, kelly_stake=4.2,
            ),
            match_result=MatchResult(
                home_prob=0.61, draw_prob=0.19, away_prob=0.20,
                most_likely_score=(3, 2), most_likely_prob=0.09,
                home_position=1, away_position=4,
            ),
            home_form=TeamForm(
                team_name="Bayern Munich",
                results=["W", "W", "W", "W", "D"],
                goals_scored=[4, 3, 2, 3, 1],
                goals_conceded=[1, 0, 1, 2, 1],
            ),
            away_form=TeamForm(
                team_name="Borussia Dortmund",
                results=["L", "W", "L", "W", "W"],
                goals_scored=[1, 3, 0, 2, 2],
                goals_conceded=[2, 1, 2, 1, 0],
            ),
        ),
        ModelOutput(
            match_id="SA_001", date="2025-08-17", league="Serie A",
            home_team="Inter Milan", away_team="AC Milan",
            expected_home_goals=1.6, expected_away_goals=1.2,
            home_strength=1.52, away_strength=1.18,
            goal_distribution={
                "exactly_0": 0.07, "exactly_1": 0.16, "exactly_2": 0.24,
                "exactly_3": 0.24, "exactly_4": 0.16, "exactly_5": 0.08,
                "exactly_6": 0.05,
            },
            value_bet=None,
            match_result=MatchResult(
                home_prob=0.49, draw_prob=0.27, away_prob=0.24,
                most_likely_score=(1, 0), most_likely_prob=0.12,
                home_position=2, away_position=5,
            ),
            home_form=TeamForm(
                team_name="Inter Milan",
                results=["W", "D", "W", "W", "D"],
                goals_scored=[2, 0, 1, 2, 1],
                goals_conceded=[0, 0, 0, 1, 1],
            ),
            away_form=TeamForm(
                team_name="AC Milan",
                results=["D", "L", "W", "D", "L"],
                goals_scored=[1, 0, 2, 1, 0],
                goals_conceded=[1, 2, 1, 1, 1],
            ),
        ),
    ]

# Shared state updated by the scheduler
_predictions: List[ModelOutput] = []
_last_updated: str = "Never"
_lock = threading.Lock()

# Phase status — broadcast to SSE clients so UI can show live progress
# Keys: "phase" (1|2|done), "step" (human label), "leagues_done" (list)
_phase_status: dict = {"phase": 0, "step": "Initialising...", "leagues_done": [], "stale": False}
_status_lock = threading.Lock()

def _set_status(phase: int, step: str, leagues_done: list = None, stale: bool = False) -> None:
    global _phase_status
    with _status_lock:
        _phase_status = {
            "phase":        phase,
            "step":         step,
            "leagues_done": leagues_done if leagues_done is not None else _phase_status["leagues_done"],
            "stale":        stale,
        }


# ── Mock data (used without --live flag) ──────────────────────────────────────
MOCK_PREDICTIONS = [
    ModelOutput(
        match_id="EPL_001", date="2025-08-16", league="Premier League",
        home_team="Arsenal", away_team="Chelsea",
        expected_home_goals=1.8, expected_away_goals=1.3,
        home_strength=1.45, away_strength=1.20,
        goal_distribution={
            "exactly_0": 0.05, "exactly_1": 0.14, "exactly_2": 0.22,
            "exactly_3": 0.25, "exactly_4": 0.18, "exactly_5": 0.10,
            "exactly_6": 0.06,
        },
        value_bet=ValueBet(
            recommendation="Over 2.5 Goals @ 1.85 (Pinnacle)",
            edge=0.14, kelly_stake=8.5,
        ),
    ),
    ModelOutput(
        match_id="LAL_001", date="2025-08-16", league="La Liga",
        home_team="Barcelona", away_team="Real Madrid",
        expected_home_goals=1.5, expected_away_goals=1.4,
        home_strength=1.38, away_strength=1.35,
        goal_distribution={
            "exactly_0": 0.08, "exactly_1": 0.18, "exactly_2": 0.26,
            "exactly_3": 0.23, "exactly_4": 0.14, "exactly_5": 0.07,
            "exactly_6": 0.04,
        },
        value_bet=None,
    ),
    ModelOutput(
        match_id="BUN_001", date="2025-08-17", league="Bundesliga",
        home_team="Bayern Munich", away_team="Borussia Dortmund",
        expected_home_goals=2.2, expected_away_goals=1.6,
        home_strength=1.80, away_strength=1.42,
        goal_distribution={
            "exactly_0": 0.03, "exactly_1": 0.09, "exactly_2": 0.18,
            "exactly_3": 0.24, "exactly_4": 0.22, "exactly_5": 0.14,
            "exactly_6": 0.10,
        },
        value_bet=ValueBet(
            recommendation="Over 3.5 Goals @ 2.10 (Bet365)",
            edge=0.08, kelly_stake=4.2,
        ),
    ),
    ModelOutput(
        match_id="SA_001", date="2025-08-17", league="Serie A",
        home_team="Inter Milan", away_team="AC Milan",
        expected_home_goals=1.6, expected_away_goals=1.2,
        home_strength=1.52, away_strength=1.18,
        goal_distribution={
            "exactly_0": 0.07, "exactly_1": 0.16, "exactly_2": 0.24,
            "exactly_3": 0.24, "exactly_4": 0.16, "exactly_5": 0.08,
            "exactly_6": 0.05,
        },
        value_bet=None,
    ),
]


# ── HTML template ────────────────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soccer Predictions</title>
    <style>
        :root {
            --bg:      #090e1a;
            --card:    #101825;
            --border:  rgba(255,255,255,0.07);
            --bh:      rgba(255,255,255,0.13);
            --t1:      #f0f4f8;
            --t2:      #7a8fa6;
            --t3:      #3d5068;
            --green:   #10b981;
            --yellow:  #f59e0b;
            --red:     #ef4444;
            --blue:    #3b82f6;
        }
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg); color: var(--t1); min-height:100vh;
        }

        /* ── Header ── */
        .hdr {
            background: linear-gradient(160deg,#0d1a35,#131b30,#1a1040);
            border-bottom: 1px solid rgba(139,92,246,0.2);
            padding: 0.9rem 1.25rem 0;
            position: sticky; top:0; z-index:200;
            box-shadow: 0 4px 24px rgba(0,0,0,0.5);
        }
        .hdr-row {
            display:flex; align-items:center; justify-content:space-between;
            max-width:1200px; margin:0 auto;
        }
        .hdr h1 { font-size:1.2rem; font-weight:700; }
        .hdr-nav { display:flex; gap:.15rem; margin-left:1.2rem; }
        .hdr-nav a {
            font-size:.72rem; font-weight:600; padding:.28rem .7rem;
            border-radius:6px; text-decoration:none; color:var(--t2);
            transition:background .15s, color .15s;
        }
        .hdr-nav a:hover  { background:rgba(255,255,255,.07); color:var(--t1); }
        .hdr-nav a.active { background:rgba(139,92,246,.18); color:#a78bfa; }
        .live-pill {
            display:inline-flex; align-items:center; gap:5px;
            background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
            color:var(--green); padding:0.18rem 0.6rem; border-radius:20px;
            font-size:0.7rem; font-weight:600;
        }
        .pulse {
            width:6px; height:6px; background:var(--green);
            border-radius:50%; animation:blink 1.6s ease-in-out infinite; flex-shrink:0;
        }
        @keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.25;transform:scale(.65)} }

        /* ── Stats strip ── */
        .stats-strip {
            display:flex; max-width:1200px; margin:0.7rem auto 0;
            border-top:1px solid var(--border);
        }
        .stat-pill {
            flex:1; text-align:center; padding:0.4rem 0.25rem;
            border-right:1px solid var(--border);
        }
        .stat-pill:last-child { border-right:none; }
        .stat-label { font-size:0.58rem; text-transform:uppercase; letter-spacing:.07em; color:var(--t3); }
        .stat-num   { font-size:1.05rem; font-weight:700; }

        /* ── Loading overlay ── */
        .load-overlay {
            position:fixed; inset:0; z-index:999;
            background:var(--bg);
            display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            gap:1.2rem; padding:2rem;
            transition:opacity .4s ease;
        }
        .load-overlay.fade-out { opacity:0; pointer-events:none; }
        .load-overlay.hidden   { display:none; }
        .load-spinner {
            width:40px; height:40px;
            border:3px solid rgba(139,92,246,.2);
            border-top-color:#a78bfa;
            border-radius:50%;
            animation:spin .8s linear infinite;
        }
        @keyframes spin { to { transform:rotate(360deg); } }
        .load-title  { font-size:1rem; font-weight:700; color:var(--t1); }
        .load-step   { font-size:.78rem; color:var(--t2); min-height:1.2em; text-align:center; }
        .load-bar-bg { width:260px; height:4px; background:rgba(255,255,255,.08); border-radius:2px; overflow:hidden; }
        .load-bar-fill { height:100%; width:0%; background:linear-gradient(90deg,#6366f1,#a78bfa); border-radius:2px; transition:width .4s ease; }
        .load-leagues { display:flex; flex-wrap:wrap; gap:.4rem; justify-content:center; max-width:320px; }
        .load-league  { font-size:.65rem; padding:.18rem .45rem; border-radius:4px; background:rgba(255,255,255,.06); color:var(--t3); transition:all .3s; }
        .load-league.done { background:rgba(16,185,129,.15); color:var(--green); }
        .load-stale   { font-size:.68rem; color:var(--yellow); display:flex; align-items:center; gap:.3rem; }

        /* ── Inline progress bar (shown under header while phase 2 runs) ── */
        .progress-wrap { max-width:1200px; margin:0.6rem auto 0; padding:0 1.25rem; display:none; }
        .progress-wrap.show { display:block; }
        .progress-bg   { height:3px; background:rgba(255,255,255,0.07); border-radius:2px; overflow:hidden; }
        .progress-fill { height:100%; width:0%; background:linear-gradient(90deg,var(--blue),var(--green)); border-radius:2px; transition:width 0.3s ease; }
        .progress-label { font-size:0.62rem; color:var(--t3); margin-top:3px; text-align:right; }

        /* ── Skeleton cards ── */
        .skeleton { animation:shimmer 1.6s ease-in-out infinite; }
        @keyframes shimmer {
            0%,100% { opacity:.45; }
            50%      { opacity:.15; }
        }
        .skel-card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1rem; height:140px; }

        /* ── Refresh button ── */
        .refresh-btn {
            background:rgba(139,92,246,.12); border:1px solid rgba(139,92,246,.3);
            color:#a78bfa; padding:.22rem .65rem; border-radius:6px;
            font-size:.68rem; font-weight:600; cursor:pointer;
            transition:all .15s; white-space:nowrap;
        }
        .refresh-btn:hover { background:rgba(139,92,246,.22); }
        .refresh-btn:disabled { opacity:.4; cursor:default; }
        .stale-banner {
            max-width:1200px; margin:.5rem auto 0; padding:.35rem 1rem;
            background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.2);
            border-radius:6px; font-size:.7rem; color:var(--yellow);
            display:none; align-items:center; gap:.5rem;
        }
        .stale-banner.show { display:flex; }

        /* ── Filter bar ── */
        .filter-bar {
            max-width:1200px; margin:0.85rem auto 0;
            padding:0 1rem; display:flex; flex-wrap:wrap; gap:0.4rem; align-items:center;
        }
        .fsep { width:1px; height:20px; background:var(--border); flex-shrink:0; }
        .filter-group { display:flex; flex-wrap:wrap; gap:0.4rem; align-items:center; }
        .fbtn {
            background:var(--card); border:1px solid var(--border);
            color:var(--t2); padding:0.28rem 0.65rem; border-radius:20px;
            font-size:0.72rem; cursor:pointer; font-family:inherit;
            transition:border-color .12s,color .12s,background .12s;
            white-space:nowrap;
        }
        .fbtn:hover { border-color:var(--bh); color:var(--t1); }
        .fbtn.on    { background:var(--blue); border-color:var(--blue); color:#fff; font-weight:600; }
        .fbtn.von   { background:var(--green); border-color:var(--green); color:#fff; font-weight:600; }
        .shown-count { font-size:0.68rem; color:var(--t3); margin-left:auto; }

        /* ── Grid ── */
        .container { max-width:1200px; margin:0 auto; padding:0.9rem 1rem 3rem; }
        .grid {
            display:grid;
            grid-template-columns:repeat(auto-fill,minmax(310px,1fr));
            gap:0.9rem;
        }

        /* ── Card ── */
        .card {
            background:var(--card); border-radius:14px; border:1px solid var(--border);
            overflow:hidden; transition:transform .15s,border-color .15s,box-shadow .15s;
        }
        .card:hover { transform:translateY(-2px); border-color:var(--bh); box-shadow:0 8px 32px rgba(0,0,0,.45); }
        .card.has-value { border-color:rgba(16,185,129,0.3); }
        .card.has-value:hover { border-color:rgba(16,185,129,0.55); }
        .conf-high   { border-top:3px solid var(--green); }
        .conf-medium { border-top:3px solid var(--yellow); }
        .conf-low    { border-top:3px solid var(--red); }

        .card-head { padding:.75rem .9rem .65rem; border-bottom:1px solid var(--border); }
        .meta-row  { display:flex; align-items:center; gap:.4rem; margin-bottom:.5rem; }
        .league-tag {
            font-size:.65rem; color:var(--t2); background:rgba(255,255,255,.05);
            border:1px solid var(--border); padding:.1rem .42rem; border-radius:4px; font-weight:600;
        }
        .conf-tag { font-size:.6rem; font-weight:700; padding:.1rem .42rem; border-radius:4px; text-transform:uppercase; }
        .tag-high   { background:rgba(16,185,129,.15); color:var(--green); }
        .tag-medium { background:rgba(245,158,11,.15);  color:var(--yellow); }
        .tag-low    { background:rgba(239,68,68,.12);   color:var(--red); }
        .date-tag   { margin-left:auto; font-size:.65rem; color:var(--t2); font-weight:500; }

        .teams-row { display:flex; align-items:center; gap:.5rem; }
        .team      { flex:1; font-size:.9rem; font-weight:700; line-height:1.2; }
        .team-h    { text-align:left; }
        .team-a    { text-align:right; }
        .vs-col    { flex-shrink:0; display:flex; flex-direction:column; align-items:center; gap:1px; }
        .vs-text   { font-size:.62rem; color:var(--t3); font-weight:700; text-transform:uppercase; }
        .xg-line   { font-size:.68rem; color:var(--t2); white-space:nowrap; font-variant-numeric:tabular-nums; }

        .card-body { padding:.85rem .9rem; }
        .goals-big {
            text-align:center; font-size:2.6rem; font-weight:800; line-height:1;
            background:linear-gradient(135deg,#059669,var(--green),#34d399);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
            margin-bottom:.2rem;
        }
        .goals-sub { text-align:center; font-size:.68rem; color:var(--t3); margin-bottom:.8rem; text-transform:uppercase; }

        .pbar-row   { display:flex; align-items:center; gap:.45rem; margin-bottom:.4rem; }
        .pbar-label { width:75px; font-size:.72rem; color:var(--t2); flex-shrink:0; }
        .pbar-bg    { flex:1; height:16px; background:rgba(255,255,255,.05); border-radius:8px; overflow:hidden; }
        .pbar-fill  { height:100%; border-radius:8px; }
        .fill-over  { background:linear-gradient(90deg,#047857,var(--green)); }
        .fill-under { background:linear-gradient(90deg,#1e40af,var(--blue)); }
        .pbar-val   { width:34px; text-align:right; font-size:.78rem; font-weight:700; flex-shrink:0; font-variant-numeric:tabular-nums; }

        .form-box   { margin:.75rem 0; padding:.6rem .7rem; background:rgba(255,255,255,.03); border:1px solid var(--border); border-radius:8px; }
        .sec-label  { font-size:.6rem; text-transform:uppercase; letter-spacing:.08em; color:var(--t3); margin-bottom:.4rem; }
        .form-row   { display:flex; align-items:center; gap:.4rem; margin-bottom:.3rem; }
        .form-row:last-child { margin-bottom:0; }
        .form-name  { font-size:.7rem; font-weight:600; min-width:90px; max-width:90px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .badges     { display:flex; gap:3px; flex-shrink:0; }
        .badge      { width:19px; height:19px; border-radius:4px; font-size:.6rem; font-weight:800; display:inline-flex; align-items:center; justify-content:center; }
        .bW { background:rgba(16,185,129,.18); color:var(--green);  border:1px solid rgba(16,185,129,.45); }
        .bD { background:rgba(245,158,11,.15); color:var(--yellow); border:1px solid rgba(245,158,11,.4); }
        .bL { background:rgba(239,68,68,.13);  color:var(--red);    border:1px solid rgba(239,68,68,.38); }
        .form-stats { margin-left:auto; font-size:.62rem; color:var(--t3); text-align:right; flex-shrink:0; }
        .mbar-bg    { height:3px; background:rgba(255,255,255,.07); border-radius:2px; overflow:hidden; margin-top:2px; }
        .mbar-fill  { height:100%; border-radius:2px; background:linear-gradient(90deg,var(--yellow),var(--green)); }

        .dist-box   { margin-top:.75rem; }
        .dist-bars  { display:flex; gap:4px; height:46px; align-items:flex-end; }
        .dist-col   { flex:1; display:flex; flex-direction:column; align-items:center; gap:1px; }
        .dist-pct   { font-size:.54rem; color:var(--t2); line-height:1; }
        .dist-bar   { width:100%; border-radius:3px 3px 0 0; min-height:2px; }
        .bar-norm   { background:linear-gradient(to top,#1e3a8a,var(--blue)); }
        .bar-hi     { background:linear-gradient(to top,#047857,var(--green)); }
        .dist-lbls  { display:flex; gap:4px; margin-top:3px; }
        .dist-lbl   { flex:1; text-align:center; font-size:.56rem; color:var(--t3); }

        /* ── Venue split rows ── */
        .venue-row  { display:flex; align-items:baseline; justify-content:space-between; gap:.5rem; margin-bottom:.3rem; flex-wrap:wrap; }
        .venue-row:last-child { margin-bottom:0; }
        .venue-team { font-size:.7rem; font-weight:600; flex-shrink:0; }
        .venue-stat { font-size:.65rem; color:var(--t3); text-align:right; }

        /* ── H2H bar ── */
        .h2h-bar-wrap { margin:.3rem 0; }
        .h2h-bar      { display:flex; height:18px; border-radius:6px; overflow:hidden; }
        .h2h-seg      { display:flex; align-items:center; justify-content:center; font-size:.58rem; font-weight:700; color:rgba(255,255,255,0.85); transition:width .3s; }
        .h2h-home     { background:var(--green); }
        .h2h-draw     { background:var(--t3); }
        .h2h-away     { background:var(--red); }
        .h2h-labels   { display:flex; justify-content:space-between; margin-top:.25rem; font-size:.6rem; font-weight:600; }
        .h2h-goals    { font-size:.65rem; color:var(--t3); margin-top:.35rem; }

        /* ── 1X2 result prediction ── */
        .result-box     { border:1px solid rgba(139,92,246,.25); background:rgba(139,92,246,.05); }
        .result-bar-wrap { margin:.3rem 0; }
        .result-bar     { display:flex; height:22px; border-radius:6px; overflow:hidden; gap:2px; }
        .result-seg     { display:flex; align-items:center; justify-content:center; font-size:.62rem; font-weight:700; color:rgba(255,255,255,.9); transition:width .4s ease; }
        .result-home    { background:rgba(16,185,129,.55); }
        .result-draw    { background:rgba(100,116,139,.55); }
        .result-away    { background:rgba(239,68,68,.55); }
        .seg-winner     { filter:brightness(1.35); box-shadow:0 0 6px rgba(255,255,255,.15); }
        .result-footer  { display:flex; justify-content:space-between; align-items:center; margin-top:.35rem; flex-wrap:wrap; gap:.3rem; }
        .result-outcome { font-size:.72rem; font-weight:700; color:#a78bfa; }
        .result-score   { font-size:.65rem; color:var(--t3); }
        .result-score strong { color:var(--t1); }
        .oc-tag         { font-size:.55rem; font-weight:600; padding:1px 5px; border-radius:4px; margin-left:.4rem; vertical-align:middle; }
        .oc-high        { background:rgba(16,185,129,.2); color:var(--green); }
        .oc-medium      { background:rgba(234,179,8,.15); color:#fbbf24; }
        .oc-low         { background:rgba(100,116,139,.2); color:var(--t3); }
        .pos-row        { display:flex; align-items:center; justify-content:space-between; margin-top:.4rem; font-size:.63rem; }
        .pos-badge      { background:rgba(139,92,246,.2); color:#a78bfa; padding:1px 7px; border-radius:4px; font-weight:700; }
        .pos-label      { color:var(--t3); }

        .value-box  { display:flex; align-items:flex-start; gap:.5rem; margin-top:.75rem; padding:.5rem .65rem; background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.28); border-radius:8px; }
        .value-icon { font-size:.9rem; flex-shrink:0; margin-top:1px; }
        .value-info { min-width:0; }
        .value-rec  { font-size:.72rem; font-weight:600; color:var(--green); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .value-stat { font-size:.62rem; color:rgba(16,185,129,.65); margin-top:2px; }

        .empty { grid-column:1/-1; text-align:center; padding:3.5rem 1rem; color:var(--t2); }
        .empty-icon { font-size:2.2rem; margin-bottom:.6rem; }
        .empty h3   { margin-bottom:.4rem; }
        .empty p    { font-size:.82rem; color:var(--t3); }

        .footer { text-align:center; font-size:.72rem; color:var(--t3); margin-top:2rem; padding-top:1.25rem; border-top:1px solid var(--border); }

        @media(max-width:580px){
            .hdr { padding:.75rem .9rem 0; }
            .hdr h1 { font-size:1.05rem; }
            .filter-bar { padding:0 .75rem; }
            .container  { padding:.75rem .75rem 2.5rem; }
            .grid       { grid-template-columns:1fr; gap:.7rem; }
            .team       { font-size:.82rem; }
            .goals-big  { font-size:2.2rem; }
        }
    </style>
</head>
<body>

<!-- ── Loading overlay (hidden once first data arrives) ── -->
<div class="load-overlay" id="load-overlay">
    <div class="load-spinner"></div>
    <div class="load-title">Soccer Predictions</div>
    <div class="load-step" id="load-step">Connecting...</div>
    <div class="load-bar-bg"><div class="load-bar-fill" id="load-bar"></div></div>
    <div class="load-leagues" id="load-leagues">
        <span class="load-league" id="ll-PL">PL</span>
        <span class="load-league" id="ll-PD">La Liga</span>
        <span class="load-league" id="ll-BL1">Buli</span>
        <span class="load-league" id="ll-SA">SA</span>
        <span class="load-league" id="ll-FL1">L1</span>
    </div>
</div>

<div class="hdr">
    <div class="hdr-row">
        <div style="display:flex;align-items:center;gap:.5rem;">
            <h1>&#x26BD; Soccer Predictions</h1>
            <nav class="hdr-nav">
                <a href="/" class="active">Fixtures</a>
                <a href="/history">History</a>
            </nav>
        </div>
        <div style="display:flex;align-items:center;gap:.6rem;">
            <button class="refresh-btn" id="refresh-btn" onclick="forceRefresh()" title="Force fresh data fetch">&#x21BB; Refresh</button>
            <div class="live-pill">
                <div class="pulse"></div>
                <span id="status-text">Connecting...</span>
            </div>
        </div>
    </div>
    <div class="stale-banner" id="stale-banner">
        &#x26A0; Showing cached data from a previous session &mdash; refreshing in background...
    </div>
    <div class="stats-strip" id="stats-strip"></div>
    <div class="progress-wrap" id="progress-wrap">
        <div class="progress-bg"><div class="progress-fill" id="progress-fill"></div></div>
        <div class="progress-label" id="progress-label"></div>
    </div>
</div>

<div class="filter-bar" id="filter-bar"></div>

<div class="container">
    <div class="grid" id="grid"></div>
    <div class="footer" id="footer"></div>
</div>

<script>
(function() {

    /* ── State ─────────────────────────────────────────────────────────── */
    var ALL    = {};
    var LEAGUE = 'all';
    var DATE   = 'all';
    var CONF   = 'all';
    var VONLY  = false;
    var TOTAL_EXPECTED = 0;
    var _overlayDismissed = false;

    /* ── Loading overlay helpers ─────────────────────────────────────────── */
    var LEAGUE_LABELS = {
        'Premier League': 'll-PL',
        'La Liga':        'll-PD',
        'Bundesliga':     'll-BL1',
        'Serie A':        'll-SA',
        'Ligue 1':        'll-FL1',
    };
    var PHASE_BARS = { 0: 5, 1: 40, 2: 80, 3: 100 };

    function handlePhase(s) {
        var step   = document.getElementById('load-step');
        var bar    = document.getElementById('load-bar');
        var banner = document.getElementById('stale-banner');
        var pbar   = document.getElementById('progress-wrap');
        var pfill  = document.getElementById('progress-fill');
        var plabel = document.getElementById('progress-label');

        if (step)  step.textContent = s.step || '';
        if (bar)   bar.style.width  = (PHASE_BARS[s.phase] || 5) + '%';

        /* Mark league badges done */
        var done = s.leagues_done || [];
        done.forEach(function(lgName) {
            var id = LEAGUE_LABELS[lgName];
            if (id) {
                var el = document.getElementById(id);
                if (el) el.classList.add('done');
            }
        });

        /* Stale banner */
        if (banner) {
            if (s.stale) banner.classList.add('show');
            else         banner.classList.remove('show');
        }

        /* Phase 2 inline progress bar */
        if (s.phase === 2 && pbar && pfill) {
            pbar.classList.add('show');
            var p2pct = done.length ? Math.round((done.length / 5) * 100) : 5;
            pfill.style.width = p2pct + '%';
            if (plabel) plabel.textContent = 'Loading form + H2H... ' + done.length + '/5 leagues';
        }

        /* Dismiss overlay once first real data is visible */
        if (!_overlayDismissed && Object.keys(ALL).length > 0) {
            dismissOverlay();
        }

        /* Phase 3 = all done */
        if (s.phase === 3) {
            dismissOverlay();
            if (pbar) {
                pfill.style.width = '100%';
                setTimeout(function(){ pbar.classList.remove('show'); }, 1500);
            }
            var rbtn = document.getElementById('refresh-btn');
            if (rbtn) rbtn.disabled = false;
        }
    }

    function dismissOverlay() {
        if (_overlayDismissed) return;
        _overlayDismissed = true;
        var ov = document.getElementById('load-overlay');
        if (!ov) return;
        ov.classList.add('fade-out');
        setTimeout(function(){ ov.classList.add('hidden'); }, 450);
    }

    function showSkeletons(count) {
        var grid = document.getElementById('grid');
        if (!grid || grid.children.length > 0) return;
        var html = '';
        for (var i = 0; i < (count || 6); i++) {
            html += '<div class="skel-card skeleton"></div>';
        }
        grid.innerHTML = html;
    }

    function forceRefresh() {
        var btn = document.getElementById('refresh-btn');
        if (btn) btn.disabled = true;
        _overlayDismissed = false;
        var ov = document.getElementById('load-overlay');
        if (ov) { ov.classList.remove('fade-out','hidden'); }
        var step = document.getElementById('load-step');
        if (step) step.textContent = 'Refreshing data...';
        ALL = {};
        var grid = document.getElementById('grid');
        if (grid) grid.innerHTML = '';
        showSkeletons(6);
        fetch('/api/refresh/force')
            .then(function(r){ return r.json(); })
            .then(function(d){
                if (d.status === 'ok') startStream();
            })
            .catch(function(){
                if (btn) btn.disabled = false;
                dismissOverlay();
            });
    }

    var LEAGUE_ABBR = {
        'Premier League': 'PL',
        'La Liga':        'LaLiga',
        'Bundesliga':     'Buli',
        'Serie A':        'SA',
        'Ligue 1':        'L1'
    };

    /* ── Helpers ────────────────────────────────────────────────────────── */
    function cleanName(n) {
        if (!n) return '';
        return n.replace(/ +F[.]?C[.]?$/i,'').replace(/^F[.]?C[.]? +/i,'')
                .replace(/ +A[.]?F[.]?C[.]?$/i,'').replace(/ +S[.]?C[.]?$/i,'')
                .replace(/ +C[.]?F[.]?$/i,'').trim();
    }

    function fmtDate(iso) {
        if (!iso) return '';
        var MO = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        var DA = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
        var d  = new Date(iso + 'T12:00:00Z');
        var now   = new Date(); now.setHours(0,0,0,0);
        var tom   = new Date(now); tom.setDate(now.getDate()+1);
        var match = new Date(iso + 'T00:00:00Z'); match.setHours(0,0,0,0);
        if (match.getTime() === now.getTime()) return 'Today';
        if (match.getTime() === tom.getTime()) return 'Tomorrow';
        return DA[d.getUTCDay()] + ' ' + d.getUTCDate() + ' ' + MO[d.getUTCMonth()];
    }

    function dateBucket(iso) {
        if (!iso) return 'later';
        var now   = new Date(); now.setHours(0,0,0,0);
        var tom   = new Date(now); tom.setDate(now.getDate()+1);
        var match = new Date(iso + 'T00:00:00Z'); match.setHours(0,0,0,0);
        if (match.getTime() === now.getTime()) return 'today';
        if (match.getTime() === tom.getTime()) return 'tomorrow';
        return 'later';
    }

    function leagueAbbr(l) { return LEAGUE_ABBR[l] || (l || '').slice(0,3).toUpperCase(); }

    function esc(s) {
        return String(s || '')
            .replace(/&/g,'&amp;').replace(/</g,'&lt;')
            .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    function sf(v, d) {
        var n = parseFloat(v);
        if (isNaN(n)) { var z='0.'; for(var i=0;i<d;i++) z+='0'; return z; }
        return n.toFixed(d);
    }

    /* ── Filtering ──────────────────────────────────────────────────────── */
    function getFiltered() {
        var list = Object.values ? Object.values(ALL) : Object.keys(ALL).map(function(k){return ALL[k];});
        var out = [], i, p;
        for (i = 0; i < list.length; i++) {
            p = list[i];
            if (LEAGUE !== 'all' && p.league !== LEAGUE) continue;
            if (DATE   !== 'all' && dateBucket(p.date) !== DATE) continue;
            if (CONF   !== 'all' && p.confidence !== CONF) continue;
            if (VONLY  && !p.value_bet) continue;
            out.push(p);
        }
        out.sort(function(a,b) {
            if (!!b.value_bet !== !!a.value_bet) return b.value_bet ? 1 : -1;
            var o = {high:0,medium:1,low:2};
            return (o[a.confidence]||2) - (o[b.confidence]||2);
        });
        return out;
    }

    /* ── Stats strip ────────────────────────────────────────────────────── */

    /* Always compute stats live from ALL so they're always accurate */
    function computeStats() {
        var list = Object.values ? Object.values(ALL) : Object.keys(ALL).map(function(k){return ALL[k];});
        var total = list.length;
        var highConf = 0, valueBets = 0, totalGoals = 0;
        for (var i = 0; i < list.length; i++) {
            if (list[i].confidence === 'high') highConf++;
            if (list[i].value_bet) valueBets++;
            totalGoals += (parseFloat(list[i].predicted_goals) || 0);
        }
        return {
            total:           total,
            high_confidence: highConf,
            avg_goals:       total > 0 ? totalGoals / total : 0,
            value_bets:      valueBets
        };
    }

    function renderStats(s) {
        /* Accept either a server stats object or compute from ALL */
        var st = s || computeStats();
        var avg = parseFloat(st.avg_goals);
        var h = '';
        h += spill('Matches',   st.total || 0,                          '');
        h += spill('High Conf', st.high_confidence || 0,                'color:var(--green)');
        h += spill('Avg Goals', isNaN(avg) ? '0.0' : avg.toFixed(1),   'color:var(--blue)');
        h += spill('Value Bets',st.value_bets || 0,                     'color:var(--yellow)');
        document.getElementById('stats-strip').innerHTML = h;
        if (s && s.last_updated && s.last_updated !== 'Never') {
            document.getElementById('status-text').textContent = s.last_updated;
        }
    }

    function spill(label, val, style) {
        return '<div class="stat-pill"><div class="stat-label">' + label +
               '</div><div class="stat-num" style="' + style + '">' + val + '</div></div>';
    }

    /* ── Filter bar — uses data-* attrs + event delegation ─────────────── */

    var _filterRebuildTimer = null;

    /* Throttle: only rebuild the bar at most every 200ms during streaming */
    function scheduleFilterBar() {
        if (_filterRebuildTimer) return;
        _filterRebuildTimer = setTimeout(function() {
            _filterRebuildTimer = null;
            buildFilterBar();
        }, 200);
    }

    function buildFilterBar() {
        var list = Object.values ? Object.values(ALL) : Object.keys(ALL).map(function(k){return ALL[k];});
        var leagues = [], dates = [], seenL = {}, seenD = {}, i, p, b;

        /* Collect unique leagues and date buckets from ALL predictions */
        for (i = 0; i < list.length; i++) {
            p = list[i];
            if (p.league && !seenL[p.league]) { seenL[p.league]=1; leagues.push(p.league); }
            b = dateBucket(p.date);
            if (!seenD[b]) { seenD[b]=1; dates.push(b); }
        }

        /* Sort date buckets: today, tomorrow, later */
        var dateOrder = {today:0, tomorrow:1, later:2};
        dates.sort(function(a,b){ return (dateOrder[a]||99)-(dateOrder[b]||99); });

        /* Sort leagues alphabetically */
        leagues.sort();

        var filtered = getFiltered();
        var h = '';

        /* ── Date row ── */
        h += '<div class="filter-group">';
        h += mkbtn('all-dates', DATE==='all', 'date', 'all', 'All Dates');
        for (i = 0; i < dates.length; i++) {
            var dlbl = dates[i]==='today' ? 'Today' : dates[i]==='tomorrow' ? 'Tomorrow' : 'Later';
            h += mkbtn('d'+i, DATE===dates[i], 'date', dates[i], dlbl);
        }
        h += '</div>';

        h += '<div class="fsep"></div>';

        /* ── League row ── */
        h += '<div class="filter-group">';
        h += mkbtn('lg-all', LEAGUE==='all', 'league', 'all', 'All Leagues');
        for (i = 0; i < leagues.length; i++) {
            h += mkbtn('lg'+i, LEAGUE===leagues[i], 'league', leagues[i], leagueAbbr(leagues[i]));
        }
        h += '</div>';

        h += '<div class="fsep"></div>';

        /* ── Confidence row ── */
        h += '<div class="filter-group">';
        h += mkbtn('conf-all', CONF==='all',    'conf', 'all',    'All Conf');
        h += mkbtn('conf-hi',  CONF==='high',   'conf', 'high',   '&#x1F7E2; High');
        h += mkbtn('conf-med', CONF==='medium', 'conf', 'medium', '&#x1F7E1; Med');
        h += mkbtn('conf-low', CONF==='low',    'conf', 'low',    '&#x1F534; Low');
        h += '</div>';

        h += '<div class="fsep"></div>';

        /* ── Value toggle + count ── */
        h += '<button class="fbtn' + (VONLY?' von':'') + '" data-action="value">&#x1F4B0; Value Only</button>';
        h += '<div class="shown-count">' + filtered.length + ' of ' + list.length + ' shown</div>';

        var bar = document.getElementById('filter-bar');
        bar.innerHTML = h;

        /* Single delegated listener — no inline JS at all */
        bar.onclick = function(e) {
            var btn = e.target;
            if (!btn || btn.tagName !== 'BUTTON') return;
            var action = btn.getAttribute('data-action');
            var val    = btn.getAttribute('data-val');
            if (action === 'date')  { DATE   = val;    buildFilterBar(); renderGrid(getFiltered()); }
            if (action === 'league'){ LEAGUE = val;    buildFilterBar(); renderGrid(getFiltered()); }
            if (action === 'conf')  { CONF   = val;    buildFilterBar(); renderGrid(getFiltered()); }
            if (action === 'value') { VONLY  = !VONLY; buildFilterBar(); renderGrid(getFiltered()); }
        };
    }

    function mkbtn(id, active, action, val, label) {
        return '<button class="fbtn' + (active?' on':'') + '"' +
               ' data-action="' + esc(action) + '"' +
               ' data-val="'    + esc(val)    + '">' +
               label + '</button>';
    }

    /* ── Grid rendering ─────────────────────────────────────────────────── */
    function renderGrid(preds) {
        var grid = document.getElementById('grid');
        grid.innerHTML = '';
        if (!preds || preds.length === 0) {
            grid.innerHTML = '<div class="empty"><div class="empty-icon">&#x1F50D;</div>' +
                '<h3>No matches found</h3><p>Try a different filter</p></div>';
            return;
        }
        var frag = document.createDocumentFragment();
        var i, w;
        for (i = 0; i < preds.length; i++) {
            try {
                w = document.createElement('div');
                w.innerHTML = buildCard(preds[i]);
                if (w.firstChild) frag.appendChild(w.firstChild);
            } catch(e) { console.error('card err', e.message); }
        }
        grid.appendChild(frag);
    }

    /* Append a single card (for streaming) */
    function appendCard(p) {
        var id = 'card-' + p.match_id;
        var existing = document.getElementById(id);
        if (existing) {
            /* update in-place if already rendered */
            try {
                var w = document.createElement('div');
                w.innerHTML = buildCard(p);
                if (w.firstChild) existing.parentNode.replaceChild(w.firstChild, existing);
            } catch(e) {}
            return;
        }
        /* Check if it passes current filters */
        if (LEAGUE !== 'all' && p.league !== LEAGUE) return;
        if (DATE   !== 'all' && dateBucket(p.date) !== DATE) return;
        if (CONF   !== 'all' && p.confidence !== CONF) return;
        if (VONLY  && !p.value_bet) return;

        var grid = document.getElementById('grid');
        /* Remove empty state if present */
        var empty = grid.querySelector('.empty');
        if (empty) grid.innerHTML = '';

        try {
            var w2 = document.createElement('div');
            w2.innerHTML = buildCard(p);
            if (w2.firstChild) grid.appendChild(w2.firstChild);
        } catch(e) { console.error('append card err', e.message); }
    }

    /* ── Card builders ──────────────────────────────────────────────────── */
    function formRow(form) {
        if (!form) return '';
        var i, r, cls, badges = '';
        for (i = 0; i < form.results.length; i++) {
            r = form.results[i];
            cls = r==='W'?'bW':r==='D'?'bD':'bL';
            badges += '<span class="badge ' + cls + '">' + r + '</span>';
        }
        var mom = sf(form.momentum*100, 0);
        return '<div class="form-row">' +
            '<div class="form-name">' + esc(cleanName(form.team_name)) + '</div>' +
            '<div class="badges">' + badges + '</div>' +
            '<div class="form-stats">GF ' + form.avg_scored + ' GA ' + form.avg_conceded +
                '<div class="mbar-bg"><div class="mbar-fill" style="width:' + mom + '%"></div></div>' +
            '</div></div>';
    }

    function buildDist(items, pred) {
        if (!items || !items.length) return '';
        var peak = Math.round(pred || 0);
        var bars='', lbls='', i, goals, prob, pct, h, cls;
        for (i = 0; i < items.length; i++) {
            goals = items[i][0]; prob = items[i][1];
            pct = (prob*100).toFixed(0);
            h   = Math.max(prob*100, 2);
            cls = (goals===peak) ? 'bar-hi' : 'bar-norm';
            bars += '<div class="dist-col"><div class="dist-pct">' + pct + '%</div>' +
                    '<div class="dist-bar ' + cls + '" style="height:' + h + 'px"></div></div>';
            lbls += '<div class="dist-lbl">' + goals + '</div>';
        }
        return '<div class="dist-box"><div class="sec-label">Goal Distribution</div>' +
               '<div class="dist-bars">' + bars + '</div>' +
               '<div class="dist-lbls">' + lbls + '</div></div>';
    }

    function build1X2(p) {
        if (!p.match_result) return '';
        var r       = p.match_result;
        var hPct    = Math.round((r.home_prob  || 0) * 100);
        var dPct    = Math.round((r.draw_prob  || 0) * 100);
        var aPct    = Math.round((r.away_prob  || 0) * 100);
        var score   = r.most_likely_score ? r.most_likely_score[0] + '-' + r.most_likely_score[1] : '?';
        var sProb   = Math.round((r.most_likely_prob || 0) * 100);
        var outcome = r.outcome || '';
        var oc      = r.outcome_confidence || 'low';

        // Highlight the winning segment
        var hClass  = (outcome === 'Home Win')  ? ' seg-winner' : '';
        var dClass  = (outcome === 'Draw')      ? ' seg-winner' : '';
        var aClass  = (outcome === 'Away Win')  ? ' seg-winner' : '';

        // Table positions
        var posHtml = '';
        if (r.home_position || r.away_position) {
            posHtml = '<div class="pos-row">' +
                '<span class="pos-badge">#' + (r.home_position || '?') + '</span>' +
                '<span class="pos-label">League Positions</span>' +
                '<span class="pos-badge">#' + (r.away_position || '?') + '</span>' +
            '</div>';
        }

        return '<div class="form-box result-box">' +
            '<div class="sec-label">Match Result Prediction ' +
                '<span class="oc-tag oc-' + oc + '">' + oc + ' confidence</span>' +
            '</div>' +
            '<div class="result-bar-wrap">' +
                '<div class="result-bar">' +
                    '<div class="result-seg result-home' + hClass + '" style="width:' + hPct + '%">' +
                        (hPct >= 10 ? hPct + '%' : '') +
                    '</div>' +
                    '<div class="result-seg result-draw' + dClass + '" style="width:' + dPct + '%">' +
                        (dPct >= 10 ? dPct + '%' : '') +
                    '</div>' +
                    '<div class="result-seg result-away' + aClass + '" style="width:' + aPct + '%">' +
                        (aPct >= 10 ? aPct + '%' : '') +
                    '</div>' +
                '</div>' +
                '<div class="h2h-labels">' +
                    '<span style="color:var(--green)">Home Win</span>' +
                    '<span style="color:var(--t3)">Draw</span>' +
                    '<span style="color:var(--red)">Away Win</span>' +
                '</div>' +
            '</div>' +
            '<div class="result-footer">' +
                '<span class="result-outcome">&#x25B6; ' + esc(outcome) + '</span>' +
                '<span class="result-score">Most likely score: <strong>' + score +
                    '</strong> (' + sProb + '%)</span>' +
            '</div>' +
            posHtml +
        '</div>';
    }

    function buildCard(p) {
        var conf    = p.confidence || 'low';
        var home    = esc(cleanName(p.home_team));
        var away    = esc(cleanName(p.away_team));
        var date    = fmtDate(p.date);
        var lg      = leagueAbbr(p.league);
        var isVal   = p.value_bet ? ' has-value' : '';
        var cardId  = 'card-' + p.match_id;

        var formHtml = '';
        if (p.home_form || p.away_form) {
            formHtml = '<div class="form-box"><div class="sec-label">Last 5 Games</div>' +
                formRow(p.home_form) + formRow(p.away_form) + '</div>';
        }

        var venueHtml = '';
        if (p.home_split || p.away_split) {
            venueHtml = '<div class="form-box"><div class="sec-label">Venue Record (This Season)</div>';
            if (p.home_split) {
                venueHtml += '<div class="venue-row">' +
                    '<span class="venue-team">' + esc(cleanName(p.home_team)) + ' at home</span>' +
                    '<span class="venue-stat">' +
                        sf(p.home_split.avg_scored,2) + ' scored &nbsp;/&nbsp; ' +
                        sf(p.home_split.avg_conceded,2) + ' conceded per game' +
                    '</span></div>';
            }
            if (p.away_split) {
                venueHtml += '<div class="venue-row">' +
                    '<span class="venue-team">' + esc(cleanName(p.away_team)) + ' away</span>' +
                    '<span class="venue-stat">' +
                        sf(p.away_split.avg_scored,2) + ' scored &nbsp;/&nbsp; ' +
                        sf(p.away_split.avg_conceded,2) + ' conceded per game' +
                    '</span></div>';
            }
            venueHtml += '</div>';
        }

        var h2hHtml = '';
        if (p.h2h && p.h2h.matches_played > 0) {
            var dom = p.h2h.home_dominance;
            var domBar = '';
            var leftPct  = Math.round(p.h2h.home_win_rate * 100);
            var rightPct = Math.round(p.h2h.away_win_rate * 100);
            var drawPct  = 100 - leftPct - rightPct;
            h2hHtml = '<div class="form-box">' +
                '<div class="sec-label">Head to Head &mdash; Last ' + p.h2h.matches_played + ' Meetings</div>' +
                '<div class="h2h-bar-wrap">' +
                    '<div class="h2h-bar">' +
                        '<div class="h2h-seg h2h-home" style="width:' + leftPct + '%">' +
                            (leftPct > 8 ? leftPct + '%' : '') +
                        '</div>' +
                        '<div class="h2h-seg h2h-draw" style="width:' + drawPct + '%">' +
                            (drawPct > 8 ? drawPct + '%' : '') +
                        '</div>' +
                        '<div class="h2h-seg h2h-away" style="width:' + rightPct + '%">' +
                            (rightPct > 8 ? rightPct + '%' : '') +
                        '</div>' +
                    '</div>' +
                    '<div class="h2h-labels">' +
                        '<span style="color:var(--green)">' + esc(cleanName(p.home_team)) + '</span>' +
                        '<span style="color:var(--t3)">Draw</span>' +
                        '<span style="color:var(--red)">' + esc(cleanName(p.away_team)) + '</span>' +
                    '</div>' +
                '</div>' +
                '<div class="h2h-goals">Avg ' + sf(p.h2h.avg_goals_total,1) + ' goals per game' +
                    (p.h2h.is_high_scoring ? ' &nbsp;&#x1F525; High scoring fixture' : '') +
                '</div>' +
            '</div>';
        }

        var valHtml = '';
        if (p.value_bet) {
            valHtml = '<div class="value-box"><div class="value-icon">&#x1F4B0;</div>' +
                '<div class="value-info">' +
                '<div class="value-rec">' + esc(p.value_bet.recommendation) + '</div>' +
                '<div class="value-stat">Edge: ' + sf(p.value_bet.edge*100,1) +
                '% | Kelly: ' + sf(p.value_bet.kelly_stake,1) + '%</div>' +
                '</div></div>';
        }

        return '<div class="card conf-' + conf + isVal + '" id="' + cardId + '">' +
            '<div class="card-head">' +
                '<div class="meta-row">' +
                    '<span class="league-tag">' + lg + '</span>' +
                    '<span class="conf-tag tag-' + conf + '">' + conf + '</span>' +
                    '<span class="date-tag">' + date + '</span>' +
                '</div>' +
                '<div class="teams-row">' +
                    '<div class="team team-h">' + home + '</div>' +
                    '<div class="vs-col"><span class="vs-text">vs</span>' +
                        '<span class="xg-line">' + sf(p.expected_home_goals,1) +
                        ' - ' + sf(p.expected_away_goals,1) + '</span></div>' +
                    '<div class="team team-a">' + away + '</div>' +
                '</div>' +
            '</div>' +
            '<div class="card-body">' +
                '<div class="goals-big">' + sf(p.predicted_goals,1) + '</div>' +
                '<div class="goals-sub">Predicted Total Goals</div>' +
                '<div class="pbar-row"><div class="pbar-label">Over 2.5</div>' +
                    '<div class="pbar-bg"><div class="pbar-fill fill-over" style="width:' +
                    sf((p.over_prob||0)*100,0) + '%"></div></div>' +
                    '<div class="pbar-val">' + sf((p.over_prob||0)*100,0) + '%</div></div>' +
                '<div class="pbar-row"><div class="pbar-label">Under 2.5</div>' +
                    '<div class="pbar-bg"><div class="pbar-fill fill-under" style="width:' +
                    sf((p.under_prob||0)*100,0) + '%"></div></div>' +
                    '<div class="pbar-val">' + sf((p.under_prob||0)*100,0) + '%</div></div>' +
                build1X2(p) +
                formHtml +
                venueHtml +
                h2hHtml +
                buildDist(p.goal_distribution_items, p.predicted_goals) +
                valHtml +
            '</div>' +
        '</div>';
    }

    /* ── Progress bar ───────────────────────────────────────────────────── */
    function showProgress(loaded, total) {
        var wrap  = document.getElementById('progress-wrap');
        var fill  = document.getElementById('progress-fill');
        var label = document.getElementById('progress-label');
        if (total <= 0) { wrap.classList.remove('show'); return; }
        wrap.classList.add('show');
        var pct = Math.min(100, Math.round((loaded/total)*100));
        fill.style.width = pct + '%';
        label.textContent = loaded + ' / ' + total + ' matches loaded';
        if (pct >= 100) {
            setTimeout(function() { wrap.classList.remove('show'); }, 1500);
        }
    }

    /* ── SSE stream ─────────────────────────────────────────────────────── */
    function startStream() {
        if (!window.EventSource) {
            fallbackFetch();
            return;
        }

        var src = new EventSource('/api/stream');

        src.addEventListener('phase', function(e) {
            try { handlePhase(JSON.parse(e.data)); }
            catch(err) { console.error('stream phase err', err); }
        });

        src.addEventListener('match', function(e) {
            try {
                var p = JSON.parse(e.data);
                /* First real card — clear skeletons and dismiss overlay */
                if (Object.keys(ALL).length === 0) {
                    var grid = document.getElementById('grid');
                    if (grid) grid.innerHTML = '';
                    dismissOverlay();
                }
                ALL[p.match_id] = p;
                appendCard(p);
                scheduleFilterBar();
                renderStats(null);
                var cnt = Object.keys(ALL).length;
                showProgress(cnt, Math.max(cnt, TOTAL_EXPECTED));
            } catch(err) { console.error('stream match err', err); }
        });

        src.addEventListener('update', function(e) {
            try {
                var p = JSON.parse(e.data);
                ALL[p.match_id] = p;
                appendCard(p);
            } catch(err) { console.error('stream update err', err); }
        });

        src.addEventListener('stats', function(e) {
            try {
                var s = JSON.parse(e.data);
                TOTAL_EXPECTED = s.total || 0;
                renderStats(null);
                buildFilterBar();
                var cnt = Object.keys(ALL).length;
                showProgress(cnt, TOTAL_EXPECTED);
                if (s.last_updated && s.last_updated !== 'Never') {
                    document.getElementById('status-text').textContent = s.last_updated;
                }
                document.getElementById('footer').innerHTML =
                    'Live stream &nbsp;&middot;&nbsp; ' + TOTAL_EXPECTED + ' matches across 5 leagues';
            } catch(err) { console.error('stream stats err', err); }
        });

        src.addEventListener('done', function() {
            src.close();
            dismissOverlay();
            showProgress(100, 100);
            var rbtn = document.getElementById('refresh-btn');
            if (rbtn) rbtn.disabled = false;
            setInterval(fallbackFetch, 30000);
        });

        src.onerror = function() {
            src.close();
            dismissOverlay();
            fallbackFetch();
            setInterval(fallbackFetch, 30000);
        };
    }

    function fallbackFetch() {
        fetch('/api/predictions')
            .then(function(r){ return r.json(); })
            .then(function(d){
                var i;
                /* Clear skeletons on first real data */
                if (Object.keys(ALL).length === 0) {
                    var grid = document.getElementById('grid');
                    if (grid) grid.innerHTML = '';
                    dismissOverlay();
                }
                for (i = 0; i < d.predictions.length; i++) {
                    ALL[d.predictions[i].match_id] = d.predictions[i];
                }
                TOTAL_EXPECTED = d.stats.total || 0;
                renderStats(d.stats);
                buildFilterBar();
                renderGrid(getFiltered());
                document.getElementById('footer').innerHTML =
                    'Auto-refreshes every 30s &nbsp;&middot;&nbsp; ' +
                    d.stats.total + ' matches across 5 leagues';
            })
            .catch(function(){ document.getElementById('status-text').textContent = 'Reconnecting...'; });
    }

    /* ── Boot ───────────────────────────────────────────────────────────── */
    showSkeletons(10);   /* show skeleton cards immediately while connecting */
    startStream();

})();
</script>
</body>
</html>
"""

# ── History page template ─────────────────────────────────────────────────────

HISTORY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Match History \u2014 Soccer Predictions</title>
    <style>
        :root { --bg:#090e1a; --card:#101825; --border:rgba(255,255,255,0.07); --t1:#f0f4f8; --t2:#7a8fa6; --t3:#3d5068; --green:#10b981; --yellow:#f59e0b; --red:#ef4444; --purple:#a78bfa; }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:-apple-system,'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--t1); min-height:100vh; }
        .hdr { background:linear-gradient(160deg,#0d1a35,#131b30,#1a1040); border-bottom:1px solid rgba(139,92,246,0.2); padding:.9rem 1.25rem 0; position:sticky; top:0; z-index:200; box-shadow:0 4px 24px rgba(0,0,0,0.5); }
        .hdr-row { display:flex; align-items:center; justify-content:space-between; max-width:1200px; margin:0 auto; padding-bottom:.6rem; }
        .hdr-row h1 { font-size:1.2rem; font-weight:700; }
        .hdr-nav { display:flex; gap:.15rem; margin-left:1.2rem; }
        .hdr-nav a { font-size:.72rem; font-weight:600; padding:.28rem .7rem; border-radius:6px; text-decoration:none; color:var(--t2); transition:background .15s,color .15s; }
        .hdr-nav a:hover { background:rgba(255,255,255,.07); color:var(--t1); }
        .hdr-nav a.active { background:rgba(139,92,246,.18); color:var(--purple); }
        .acc-strip { display:flex; max-width:1200px; margin:0 auto; border-top:1px solid var(--border); }
        .acc-pill { flex:1; text-align:center; padding:.4rem .25rem; border-right:1px solid var(--border); }
        .acc-pill:last-child { border-right:none; }
        .acc-label { font-size:.58rem; text-transform:uppercase; letter-spacing:.07em; color:var(--t3); }
        .acc-num { font-size:1.05rem; font-weight:700; }
        .acc-num.positive { color:var(--green); }
        .acc-num.negative { color:var(--red); }
        .page-wrap { max-width:1200px; margin:0 auto; padding:1rem 1rem 3rem; }
        .filter-bar { display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin-bottom:1rem; }
        .fsep { width:1px; height:20px; background:var(--border); flex-shrink:0; }
        .filter-group { display:flex; flex-wrap:wrap; gap:.4rem; }
        .fbtn { background:rgba(255,255,255,.05); border:1px solid var(--border); color:var(--t2); padding:.28rem .65rem; border-radius:6px; font-size:.7rem; font-weight:600; cursor:pointer; transition:all .15s; }
        .fbtn:hover { background:rgba(255,255,255,.09); color:var(--t1); }
        .fbtn.active { background:rgba(139,92,246,.18); border-color:rgba(139,92,246,.4); color:var(--purple); }
        .count-tag { font-size:.68rem; color:var(--t3); margin-left:.25rem; }
        .breakdown-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:.65rem; margin-bottom:1.5rem; }
        .bk-card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:.7rem .85rem; }
        .bk-title { font-size:.63rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--t3); margin-bottom:.4rem; }
        .bk-stat { font-size:.73rem; color:var(--t2); margin-bottom:.18rem; }
        .bk-stat span { font-weight:700; color:var(--t1); }
        .bk-acc { font-size:1.05rem; font-weight:800; margin-top:.35rem; }
        .bk-acc.good { color:var(--green); }
        .bk-acc.ok { color:var(--yellow); }
        .bk-acc.bad { color:var(--red); }
        .hist-table { width:100%; border-collapse:collapse; font-size:.72rem; }
        .hist-table th { text-align:left; padding:.45rem .6rem; border-bottom:2px solid var(--border); color:var(--t3); font-size:.6rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; white-space:nowrap; }
        .hist-table td { padding:.5rem .6rem; border-bottom:1px solid var(--border); vertical-align:middle; }
        .hist-table tr:hover td { background:rgba(255,255,255,.025); }
        .match-teams { font-weight:600; font-size:.74rem; }
        .match-meta { font-size:.62rem; color:var(--t3); margin-top:.1rem; }
        .score-cell { font-weight:800; font-size:.9rem; text-align:center; white-space:nowrap; }
        .pred-cell { text-align:center; }
        .pred-goals { font-size:.78rem; font-weight:700; }
        .pred-ou { font-size:.62rem; color:var(--t3); }
        .pred-vb { font-size:.62rem; color:var(--green); margin-top:.1rem; }
        .outcome-badge { display:inline-flex; align-items:center; gap:.3rem; padding:.2rem .5rem; border-radius:5px; font-size:.65rem; font-weight:700; white-space:nowrap; }
        .badge-correct { background:rgba(16,185,129,.15); color:var(--green); border:1px solid rgba(16,185,129,.25); }
        .badge-wrong { background:rgba(239,68,68,.12); color:var(--red); border:1px solid rgba(239,68,68,.2); }
        .badge-pending { background:rgba(100,116,139,.12); color:var(--t3); border:1px solid var(--border); }
        .conf-tag { display:inline-block; padding:.15rem .4rem; border-radius:4px; font-size:.6rem; font-weight:700; }
        .tag-high { background:rgba(16,185,129,.15); color:var(--green); }
        .tag-medium { background:rgba(245,158,11,.12); color:var(--yellow); }
        .tag-low { background:rgba(100,116,139,.12); color:var(--t3); }
        .pnl-cell { text-align:right; font-weight:700; font-size:.78rem; white-space:nowrap; }
        .pnl-pos { color:var(--green); }
        .pnl-neg { color:var(--red); }
        .pnl-nil { color:var(--t3); }
        .no-results { text-align:center; padding:3.5rem 1rem; color:var(--t2); }
        .no-results-icon { font-size:2.5rem; margin-bottom:.6rem; }
        .no-results p { margin-bottom:.3rem; }
        .no-results .hint { font-size:.78rem; color:var(--t3); }
        @media(max-width:700px){
            .hist-table th:nth-child(4),.hist-table td:nth-child(4),
            .hist-table th:nth-child(6),.hist-table td:nth-child(6) { display:none; }
        }
    </style>
</head>
<body>

<div class="hdr">
    <div class="hdr-row">
        <div style="display:flex;align-items:center;gap:.5rem;">
            <h1>&#x26BD; Soccer Predictions</h1>
            <nav class="hdr-nav">
                <a href="/">Fixtures</a>
                <a href="/history" class="active">History</a>
            </nav>
        </div>
    </div>
    <div class="acc-strip">
        <div class="acc-pill"><div class="acc-label">Resolved</div><div class="acc-num" id="s-total">&#x2014;</div></div>
        <div class="acc-pill"><div class="acc-label">O/U Accuracy</div><div class="acc-num" id="s-accuracy">&#x2014;</div></div>
        <div class="acc-pill"><div class="acc-label">Value Bets</div><div class="acc-num" id="s-vbwon">&#x2014;</div></div>
        <div class="acc-pill"><div class="acc-label">P&amp;L</div><div class="acc-num" id="s-pnl">&#x2014;</div></div>
        <div class="acc-pill"><div class="acc-label">ROI</div><div class="acc-num" id="s-roi">&#x2014;</div></div>
    </div>
</div>

<div class="page-wrap">

    <div class="filter-bar" id="filter-bar">
        <div class="filter-group" id="grp-outcome">
            <button class="fbtn active" data-filter="outcome" data-val="all">All Results</button>
            <button class="fbtn" data-filter="outcome" data-val="correct">&#x2705; Correct</button>
            <button class="fbtn" data-filter="outcome" data-val="wrong">&#x274C; Wrong</button>
        </div>
        <div class="fsep"></div>
        <div class="filter-group" id="grp-conf">
            <button class="fbtn active" data-filter="conf" data-val="all">All Conf</button>
            <button class="fbtn" data-filter="conf" data-val="high">&#x1F7E2; High</button>
            <button class="fbtn" data-filter="conf" data-val="medium">&#x1F7E1; Med</button>
            <button class="fbtn" data-filter="conf" data-val="low">&#x1F534; Low</button>
        </div>
        <div class="fsep"></div>
        <div class="filter-group" id="grp-league"></div>
        <span class="count-tag" id="count-tag"></span>
    </div>

    <div class="breakdown-grid" id="breakdown-grid"></div>

    <table class="hist-table">
        <thead>
            <tr>
                <th>Match</th>
                <th style="text-align:center">Score</th>
                <th style="text-align:center">Predicted</th>
                <th style="text-align:center">Confidence</th>
                <th style="text-align:center">Outcome</th>
                <th style="text-align:right">P&amp;L</th>
            </tr>
        </thead>
        <tbody id="hist-body"></tbody>
    </table>

    <div class="no-results" id="no-results" style="display:none">
        <div class="no-results-icon">&#x1F4CA;</div>
        <p>No resolved matches yet.</p>
        <p class="hint">Results appear here automatically after matches finish. Check back tomorrow!</p>
    </div>

</div>

<script>
(function() {
    var ALL      = [];
    var FOUTCOME = 'all';
    var FCONF    = 'all';
    var FLEAGUE  = 'all';

    function esc(s) {
        return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
    function sf(v, d) { var n=parseFloat(v); return isNaN(n)?'?':n.toFixed(d); }
    function pct(v)   { return sf((v||0)*100,1)+'%'; }
    function lgAbbr(n) {
        var m={'Premier League':'PL','La Liga':'LaLiga','Bundesliga':'Buli','Serie A':'SA','Ligue 1':'L1'};
        return m[n]||(n||'').substring(0,5);
    }
    function fmtDate(s) {
        if (!s) return '';
        var d = new Date(s+'T12:00:00Z');
        return d.toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'});
    }

    function loadData() {
        fetch('/api/results/recent')
            .then(function(r){ return r.json(); })
            .then(function(d){
                ALL = d.results || [];
                buildLeagueBtns();
                render();
            })
            .catch(function(){ showMsg('Could not load history data.'); });

        fetch('/api/performance')
            .then(function(r){ return r.json(); })
            .then(function(d){
                renderStats(d.performance||{});
                renderBreakdown(d.performance||{});
            })
            .catch(function(){});
    }

    function renderStats(perf) {
        var ov  = perf.overall || {};
        var pnl = ov.total_pnl || 0;
        var roi = ov.roi || 0;
        document.getElementById('s-total').textContent    = ov.total_results || 0;
        document.getElementById('s-accuracy').textContent = pct(ov.over_under_accuracy||0);
        document.getElementById('s-vbwon').textContent    = (ov.value_bets_won||0)+' / '+(ov.value_bets_total||0);
        var pEl = document.getElementById('s-pnl');
        pEl.textContent = (pnl>=0?'+':'')+sf(pnl,2)+'u';
        pEl.className   = 'acc-num'+(pnl>0?' positive':pnl<0?' negative':'');
        var rEl = document.getElementById('s-roi');
        rEl.textContent = (roi>=0?'+':'')+sf(roi,1)+'%';
        rEl.className   = 'acc-num'+(roi>0?' positive':roi<0?' negative':'');
    }

    function renderBreakdown(perf) {
        var html = '';
        function card(title, e) {
            if (!e||!e.total_results) return '';
            var acc = (e.over_under_accuracy||0)*100;
            var cls = acc>=60?'good':acc>=50?'ok':'bad';
            var pnl = e.total_pnl||0;
            return '<div class="bk-card">'+
                '<div class="bk-title">'+esc(title)+'</div>'+
                '<div class="bk-stat">Matches: <span>'+e.total_results+'</span></div>'+
                (e.value_bets_total?'<div class="bk-stat">Value bets: <span>'+(e.value_bets_won||0)+'/'+e.value_bets_total+'</span></div>':'')+
                '<div class="bk-stat">P&amp;L: <span style="color:'+(pnl>=0?'var(--green)':'var(--red)')+'">'+
                    (pnl>=0?'+':'')+sf(pnl,2)+'u</span></div>'+
                '<div class="bk-acc '+cls+'">'+sf(acc,1)+'% O/U acc</div>'+
            '</div>';
        }
        var bc = perf.by_confidence||{};
        ['high','medium','low'].forEach(function(c){
            html += card(c.charAt(0).toUpperCase()+c.slice(1)+' Conf', bc[c]);
        });
        var bl = perf.by_league||{};
        Object.keys(bl).forEach(function(l){ html += card(l, bl[l]); });
        document.getElementById('breakdown-grid').innerHTML = html;
    }

    function buildLeagueBtns() {
        var seen = {};
        ALL.forEach(function(r){ if(r.league) seen[r.league]=true; });
        var grp  = document.getElementById('grp-league');
        var html = '<button class="fbtn active" data-filter="league" data-val="all">All Leagues</button>';
        Object.keys(seen).forEach(function(lg){
            html += '<button class="fbtn" data-filter="league" data-val="'+esc(lg)+'">'+lgAbbr(lg)+'</button>';
        });
        grp.innerHTML = html;
        grp.querySelectorAll('.fbtn').forEach(function(b){
            b.addEventListener('click', function(){
                FLEAGUE = this.getAttribute('data-val');
                grp.querySelectorAll('.fbtn').forEach(function(x){ x.classList.remove('active'); });
                this.classList.add('active');
                render();
            });
        });
    }

    function getFiltered() {
        return ALL.filter(function(r){
            if (FOUTCOME==='correct' && r.predicted_correct!==1) return false;
            if (FOUTCOME==='wrong'   && r.predicted_correct!==0) return false;
            if (FCONF!=='all' && r.confidence!==FCONF) return false;
            if (FLEAGUE!=='all' && r.league!==FLEAGUE) return false;
            return true;
        });
    }

    function render() {
        var rows   = getFiltered();
        var body   = document.getElementById('hist-body');
        var noRes  = document.getElementById('no-results');
        var ctEl   = document.getElementById('count-tag');
        ctEl.textContent = rows.length+' of '+ALL.length+' shown';
        if (!rows.length) { body.innerHTML=''; noRes.style.display=''; return; }
        noRes.style.display = 'none';
        body.innerHTML = rows.map(buildRow).join('');
    }

    function buildRow(r) {
        var over  = parseFloat(r.over_prob)||0;
        var under = parseFloat(r.under_prob)||0;
        var ouLbl = over>under ? 'Over 2.5 ('+pct(over)+')' : 'Under 2.5 ('+pct(under)+')';
        var conf  = r.confidence||'low';
        var score = (r.home_goals!==null&&r.home_goals!==undefined)
            ? r.home_goals+' &ndash; '+r.away_goals : '&#x2014;';
        var outHtml = r.predicted_correct===1
            ? '<span class="outcome-badge badge-correct">&#x2705; Correct</span>'
            : r.predicted_correct===0
                ? '<span class="outcome-badge badge-wrong">&#x274C; Wrong</span>'
                : '<span class="outcome-badge badge-pending">&#x23F3; Pending</span>';
        var pnl    = (r.pnl!==null&&r.pnl!==undefined) ? parseFloat(r.pnl) : null;
        var pnlCls = pnl>0?'pnl-pos':pnl<0?'pnl-neg':'pnl-nil';
        var pnlTxt = (r.value_rec&&pnl!==null)
            ? (pnl>=0?'+':'')+sf(pnl,2)+'u'
            : '<span style="color:var(--t3)">&#x2014;</span>';

        return '<tr>'+
            '<td><div class="match-teams">'+esc(r.home_team)+' vs '+esc(r.away_team)+'</div>'+
                '<div class="match-meta">'+lgAbbr(r.league)+' &middot; '+fmtDate(r.match_date)+'</div></td>'+
            '<td><div class="score-cell">'+score+'</div></td>'+
            '<td><div class="pred-cell">'+
                '<div class="pred-goals">'+sf(r.predicted_goals,1)+' goals</div>'+
                '<div class="pred-ou">'+ouLbl+'</div>'+
                (r.value_rec?'<div class="pred-vb">&#x1F4B0; Value bet</div>':'')+
            '</div></td>'+
            '<td style="text-align:center"><span class="conf-tag tag-'+conf+'">'+conf+'</span></td>'+
            '<td style="text-align:center">'+outHtml+'</td>'+
            '<td><div class="pnl-cell '+pnlCls+'">'+pnlTxt+'</div></td>'+
        '</tr>';
    }

    function showMsg(msg) {
        var el = document.getElementById('no-results');
        el.style.display = '';
        el.innerHTML = '<div class="no-results-icon">&#x26A0;&#xFE0F;</div><p>'+esc(msg)+'</p>';
    }

    document.getElementById('filter-bar').addEventListener('click', function(e){
        var btn = e.target.closest('.fbtn[data-filter]');
        if (!btn) return;
        var f = btn.getAttribute('data-filter');
        var v = btn.getAttribute('data-val');
        btn.closest('.filter-group').querySelectorAll('.fbtn').forEach(function(b){ b.classList.remove('active'); });
        btn.classList.add('active');
        if (f==='outcome') FOUTCOME=v;
        if (f==='conf')    FCONF=v;
        render();
    });

    loadData();
})();
</script>
</body>
</html>
"""

# ── Local development entry point ────────────────────────────────────────────
# Gunicorn calls create_app() directly — this block is only for
# running locally via: python server.py [--live]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Soccer prediction web server.")
    parser.add_argument("--live", action="store_true",
                        help="Fetch real API data (needs keys in config.py)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # Mirror the env variable gunicorn uses so create_app() works the same
    if args.live:
        os.environ["LIVE_MODE"] = "true"

    flask_app = create_app()

    port = args.port
    logger.info("")
    logger.info("=" * 50)
    logger.info("  Dashboard → http://localhost:%d", port)
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 50)
    logger.info("")

    flask_app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )
