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

from prediction_dashboard import ModelOutput, PredictionDashboard, TeamForm, ValueBet, SummaryStats

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

    @app.route("/api/refresh")
    @requires_auth
    def api_refresh() -> Response:
        with _lock:
            count = len(_predictions)
        return jsonify({"status": "ok", "predictions_loaded": count})

    @app.route("/health")
    def health() -> Response:
        """
        Public health-check endpoint (no auth).
        Render uses this to confirm the app is running.
        """
        with _lock:
            count = len(_predictions)
        return jsonify({"status": "healthy", "predictions": count})


def _start_live_mode(app: Flask) -> None:
    """Fetch live data and start the scheduler in a background thread."""
    from client_football_data import FootballDataClient
    from client_odds_api import OddsAPIClient
    from scheduler import PredictionScheduler
    from config import FOOTBALL_DATA

    dashboard = PredictionDashboard(output_dir="predictions")

    def _initialise() -> None:
        logger.info("Live mode: fetching initial data...")
        football_client = FootballDataClient()
        predictions: List[ModelOutput] = []

        for league_code in FOOTBALL_DATA.league_codes:
            try:
                preds = football_client.build_model_outputs(league_code)
                predictions.extend(preds)
                logger.info("  %s: %d fixtures", league_code, len(preds))
            except RuntimeError as exc:
                logger.warning("  %s skipped — %s", league_code, exc)

        try:
            predictions = OddsAPIClient().attach_value_bets(predictions)
        except RuntimeError as exc:
            logger.warning("Odds API skipped — %s", exc)

        update_predictions(predictions)

        # Scheduler — hook into state updater so browser stays fresh
        scheduler = PredictionScheduler(dashboard, output_dir="predictions")
        original_update = scheduler._update_predictions

        def hooked_update(preds: List[ModelOutput]) -> None:
            original_update(preds)
            update_predictions(preds)

        scheduler._update_predictions = hooked_update
        scheduler.start()
        logger.info("Scheduler started.")

    # Run initial fetch on a daemon thread so gunicorn doesn't time out
    thread = threading.Thread(target=_initialise, daemon=True, name="LiveInit")
    thread.start()


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
        "home_form":               _serialise_form(pred.home_form),
        "away_form":               _serialise_form(pred.away_form),
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


# ── HTML template (self-contained, auto-refreshes via JS polling) ─────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚽ Soccer Predictions</title>
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 2rem;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.4);
        }

        .header h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }

        .live-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255,255,255,0.15);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            margin-top: 0.4rem;
        }

        .pulse {
            width: 8px; height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50%       { opacity: 0.4; transform: scale(0.8); }
        }

        .container { max-width: 1200px; margin: 0 auto; padding: 1.5rem; }

        /* ── Stats row ── */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: var(--bg-card);
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border-left: 4px solid var(--accent-blue);
            text-align: center;
        }

        .stat-card h3 {
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .stat-value { font-size: 1.75rem; font-weight: bold; }

        /* ── Match cards ── */
        .matches-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 1.25rem;
        }

        .match-card {
            background: var(--bg-card);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }

        .match-card:hover { transform: translateY(-3px); }

        .confidence-high   { border-top: 4px solid var(--accent-green); }
        .confidence-medium { border-top: 4px solid var(--accent-yellow); }
        .confidence-low    { border-top: 4px solid var(--accent-red); }

        .match-header {
            background: linear-gradient(135deg, #1e293b, #334155);
            padding: 1.25rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }

        .teams {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 1.1rem;
            font-weight: 600;
        }

        .vs { color: var(--text-secondary); font-size: 0.8rem; margin: 0 0.75rem; }
        .meta { color: var(--text-secondary); font-size: 0.8rem; margin-top: 0.4rem; }

        .match-body { padding: 1.25rem; }

        .goal-prediction {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(135deg, var(--accent-green), #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .prediction-sub {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }

        /* ── Probability bars ── */
        .prob-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.6rem;
        }

        .prob-label { width: 90px; font-size: 0.8rem; color: var(--text-secondary); }

        .prob-bar-bg {
            flex: 1;
            height: 20px;
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            overflow: hidden;
            margin: 0 0.75rem;
        }

        .prob-bar-fill {
            height: 100%;
            border-radius: 10px;
            animation: growBar 0.6s ease forwards;
        }

        @keyframes growBar { from { width: 0; } }

        .over-fill  { background: linear-gradient(90deg, var(--accent-green), #059669); }
        .under-fill { background: linear-gradient(90deg, var(--accent-blue), #2563eb); }

        .prob-value { width: 40px; text-align: right; font-size: 0.85rem; font-weight: 600; }

        /* ── Form section ── */
        .form-section {
            margin: 1rem 0;
            padding: 0.75rem 1rem;
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.07);
        }

        .form-title {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .form-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.4rem;
            flex-wrap: wrap;
        }

        .form-row:last-child { margin-bottom: 0; }

        .form-team {
            font-size: 0.78rem;
            font-weight: 600;
            min-width: 110px;
        }

        .form-badges { display: flex; gap: 3px; }

        .form-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
        }

        .form-W { background: rgba(16,185,129,0.2);  color: var(--accent-green);  border: 1px solid var(--accent-green); }
        .form-D { background: rgba(245,158,11,0.2);  color: var(--accent-yellow); border: 1px solid var(--accent-yellow); }
        .form-L { background: rgba(239,68,68,0.2);   color: var(--accent-red);    border: 1px solid var(--accent-red); }

        .form-stats {
            font-size: 0.7rem;
            color: var(--text-secondary);
            margin-left: auto;
        }

        .momentum-bar-wrap { display: flex; align-items: center; gap: 4px; margin-top: 2px; }

        .momentum-bar-bg {
            width: 60px;
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            overflow: hidden;
        }

        .momentum-bar-fill {
            height: 100%;
            border-radius: 2px;
            background: linear-gradient(90deg, var(--accent-yellow), var(--accent-green));
        }

        /* ── Distribution bars ── */
        .dist-title { font-size: 0.8rem; color: var(--text-secondary); margin: 1rem 0 0.4rem; }

        .dist-bars {
            display: flex;
            gap: 4px;
            height: 55px;
            align-items: flex-end;
        }

        .dist-bar {
            flex: 1;
            background: linear-gradient(to top, var(--accent-blue), #60a5fa);
            border-radius: 3px 3px 0 0;
            position: relative;
            min-height: 2px;
            cursor: pointer;
        }

        .dist-bar:hover::after {
            content: attr(data-prob);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            color: #fff;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7rem;
            white-space: nowrap;
        }

        .dist-label-row {
            display: flex;
            gap: 4px;
            margin-top: 4px;
        }

        .dist-label-item {
            flex: 1;
            text-align: center;
            font-size: 0.65rem;
            color: var(--text-secondary);
        }

        /* ── Value badge ── */
        .value-badge {
            display: inline-block;
            padding: 0.4rem 0.85rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 0.75rem;
            width: 100%;
            text-align: center;
        }

        .value-yes {
            background: rgba(16,185,129,0.15);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
        }

        .value-no {
            background: rgba(148,163,184,0.08);
            color: var(--text-secondary);
        }

        /* ── Footer ── */
        .footer {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 2.5rem;
            padding: 1.5rem;
            border-top: 1px solid rgba(255,255,255,0.08);
        }

        /* ── Loading overlay ── */
        #loading {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(15,23,42,0.7);
            align-items: center;
            justify-content: center;
            z-index: 999;
            font-size: 1.2rem;
        }

        #loading.show { display: flex; }

        /* ── Mobile tweaks ── */
        @media (max-width: 600px) {
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .matches-grid { grid-template-columns: 1fr; }
            .teams { font-size: 0.95rem; }
            .header h1 { font-size: 1.4rem; }
        }
    </style>
</head>
<body>

<div id="loading">⏳ Refreshing predictions…</div>

<div class="header">
    <h1>⚽ Soccer Predictions</h1>
    <div class="live-badge">
        <div class="pulse"></div>
        <span id="last-updated">Loading…</span>
    </div>
</div>

<div class="container">
    <div class="stats-grid" id="stats-grid"></div>
    <div class="matches-grid" id="matches-grid"></div>
    <div class="footer" id="footer"></div>
</div>

<script>
    var REFRESH_INTERVAL_MS = 30000;

    // ── Fetch predictions from the API endpoint ────────────────────────────
    function fetchPredictions() {
        fetch('/api/predictions')
            .then(function(res) { return res.json(); })
            .then(function(data) {
                try {
                    render(data);
                } catch(renderErr) {
                    document.getElementById('last-updated').textContent =
                        'Render error: ' + renderErr.message;
                    console.error('Render error:', renderErr);
                }
            })
            .catch(function(err) {
                document.getElementById('last-updated').textContent =
                    'Fetch error: ' + err.message;
                console.error('Fetch failed:', err);
            });
    }

    // ── Render helpers ────────────────────────────────────────────────────
    function render(data) {
        renderStats(data.stats);
        // Render cards in batches so the browser doesn't freeze on 100+ cards
        renderMatchesBatched(data.predictions);
        document.getElementById('last-updated').textContent =
            'Updated ' + data.last_updated;
        document.getElementById('footer').textContent =
            'Auto-refreshes every 30s · ' + data.stats.total + ' matches loaded';
    }

    function renderStats(stats) {
        document.getElementById('stats-grid').innerHTML =
            '<div class="stat-card"><h3>Matches</h3><div class="stat-value">' + stats.total + '</div></div>' +
            '<div class="stat-card"><h3>High Confidence</h3><div class="stat-value" style="color:var(--accent-green)">' + stats.high_confidence + '</div></div>' +
            '<div class="stat-card"><h3>Avg Goals</h3><div class="stat-value" style="color:var(--accent-blue)">' + stats.avg_goals.toFixed(2) + '</div></div>' +
            '<div class="stat-card"><h3>Value Bets</h3><div class="stat-value" style="color:var(--accent-yellow)">' + stats.value_bets + '</div></div>';
    }

    function renderMatchesBatched(predictions) {
        // Render first 10 immediately, then add the rest in chunks
        // so the phone browser doesn't block for 2+ seconds on 100 cards
        var grid = document.getElementById('matches-grid');
        grid.innerHTML = '';
        var BATCH = 10;
        var idx = 0;
        function renderBatch() {
            var slice = predictions.slice(idx, idx + BATCH);
            slice.forEach(function(p) {
                var div = document.createElement('div');
                try {
                    div.innerHTML = matchCard(p);
                    grid.appendChild(div.firstChild);
                } catch(e) {
                    console.error('Card error for', p.home_team, e);
                }
            });
            idx += BATCH;
            if (idx < predictions.length) {
                setTimeout(renderBatch, 0); // yield to browser between batches
            }
        }
        renderBatch();
    }

    // formRow defined at module scope — avoids mobile browser strict-mode scoping bugs
    function formRow(form) {
        if (!form) return "";
        var badges = form.results.map(function(r) {
            var cls = r === "W" ? "form-W" : r === "D" ? "form-D" : "form-L";
            return "<span class=\"form-badge " + cls + "\">" + r + "</span>";
        }).join("");
        var momentum = (form.momentum * 100).toFixed(0);
        return "<div class=\"form-row\">" +
            "<div class=\"form-team\">" + form.team_name + "</div>" +
            "<div class=\"form-badges\">" + badges + "</div>" +
            "<div class=\"form-stats\">GF:" + form.avg_scored +
            " GA:" + form.avg_conceded +
            " &nbsp;&middot;&nbsp; Momentum: " + momentum + "%" +
            "<div class=\"momentum-bar-wrap\"><div class=\"momentum-bar-bg\">" +
            "<div class=\"momentum-bar-fill\" style=\"width:" + momentum + "%\"></div>" +
            "</div></div></div></div>";
    }

    function matchCard(p) {
        // Distribution bars
        var distBars = p.goal_distribution_items.map(function(item) {
            var h = Math.max(item[1] * 100, 2);
            return "<div class=\"dist-bar\" style=\"height:" + h + "%\" data-prob=\"" + (item[1]*100).toFixed(1) + "%\"></div>";
        }).join("");

        var distLabels = p.goal_distribution_items.map(function(item) {
            return "<div class=\"dist-label-item\">" + item[0] + "</div>";
        }).join("");

        // Form section
        var formSection = (p.home_form || p.away_form)
            ? "<div class=\"form-section\"><div class=\"form-title\">Last 5 Games</div>" +
              formRow(p.home_form) + formRow(p.away_form) + "</div>"
            : "";

        // Value badge — no backticks for Huawei browser compatibility
        var badge = p.value_bet
            ? "<div class=\"value-badge value-yes\"> VALUE BET: " +
              p.value_bet.recommendation +
              " &nbsp;|&nbsp; Edge: " + (p.value_bet.edge*100).toFixed(1) + "%" +
              " &nbsp;|&nbsp; Stake: " + p.value_bet.kelly_stake.toFixed(1) + "%" +
              "</div>"
            : "<div class=\"value-badge value-no\">No Value Detected</div>";

        return "<div class=\"match-card confidence-" + p.confidence + "\">" +
            "<div class=\"match-header\">" +
                "<div class=\"teams\">" +
                    "<span>" + p.home_team + "</span>" +
                    "<span class=\"vs\">VS</span>" +
                    "<span>" + p.away_team + "</span>" +
                "</div>" +
                "<div class=\"meta\">" + p.league + " &nbsp;&middot;&nbsp; " + p.date + "</div>" +
            "</div>" +
            "<div class=\"match-body\">" +
                "<div class=\"goal-prediction\">" + p.predicted_goals.toFixed(1) + "</div>" +
                "<div class=\"prediction-sub\">Predicted Total Goals</div>" +
                "<div class=\"prob-row\">" +
                    "<div class=\"prob-label\">Over 2.5</div>" +
                    "<div class=\"prob-bar-bg\"><div class=\"prob-bar-fill over-fill\" style=\"width:" + (p.over_prob*100).toFixed(0) + "%\"></div></div>" +
                    "<div class=\"prob-value\">" + (p.over_prob*100).toFixed(0) + "%</div>" +
                "</div>" +
                "<div class=\"prob-row\">" +
                    "<div class=\"prob-label\">Under 2.5</div>" +
                    "<div class=\"prob-bar-bg\"><div class=\"prob-bar-fill under-fill\" style=\"width:" + (p.under_prob*100).toFixed(0) + "%\"></div></div>" +
                    "<div class=\"prob-value\">" + (p.under_prob*100).toFixed(0) + "%</div>" +
                "</div>" +
                formSection +
                "<div class=\"dist-title\">Goal Distribution</div>" +
                "<div class=\"dist-bars\">" + distBars + "</div>" +
                "<div class=\"dist-label-row\">" + distLabels + "</div>" +
                badge +
            "</div>" +
        "</div>";
    }

    // ── Bootstrap ─────────────────────────────────────────────────────────
    fetchPredictions();
    setInterval(fetchPredictions, REFRESH_INTERVAL_MS);
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
