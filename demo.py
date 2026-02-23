"""
Full pipeline demo — shows how all components connect together.

Modes:
    1. LIVE   — fetches real data from APIs (requires valid API keys in config.py)
    2. MOCK   — uses sample data to demonstrate the dashboard without API calls
               (default, safe to run without keys)

Usage:
    python demo.py          # mock mode
    python demo.py --live   # live mode (needs real API keys)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from prediction_dashboard import ModelOutput, PredictionDashboard, ValueBet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")


# ══════════════════════════════════════════════════════════════════════════════
# Mock data
# ══════════════════════════════════════════════════════════════════════════════

MOCK_PREDICTIONS = [
    ModelOutput(
        match_id="EPL_001", date="2024-03-15", league="Premier League",
        home_team="Arsenal", away_team="Chelsea",
        expected_home_goals=1.8, expected_away_goals=1.3,
        home_strength=1.45, away_strength=1.20,
        goal_distribution={
            "exactly_0": 0.05, "exactly_1": 0.14, "exactly_2": 0.22,
            "exactly_3": 0.25, "exactly_4": 0.18, "exactly_5": 0.10,
            "exactly_6": 0.06,
        },
        value_bet=ValueBet(
            recommendation="Over 2.5 Goals @ 1.85 (Pinnacle)", edge=0.14, kelly_stake=8.5,
        ),
    ),
    ModelOutput(
        match_id="LAL_001", date="2024-03-15", league="La Liga",
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
        match_id="BUN_001", date="2024-03-16", league="Bundesliga",
        home_team="Bayern Munich", away_team="Borussia Dortmund",
        expected_home_goals=2.2, expected_away_goals=1.6,
        home_strength=1.80, away_strength=1.42,
        goal_distribution={
            "exactly_0": 0.03, "exactly_1": 0.09, "exactly_2": 0.18,
            "exactly_3": 0.24, "exactly_4": 0.22, "exactly_5": 0.14,
            "exactly_6": 0.10,
        },
        value_bet=ValueBet(
            recommendation="Over 3.5 Goals @ 2.10 (Bet365)", edge=0.08, kelly_stake=4.2,
        ),
    ),
    ModelOutput(
        match_id="SA_001", date="2024-03-16", league="Serie A",
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


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runners
# ══════════════════════════════════════════════════════════════════════════════

def run_mock_pipeline() -> None:
    logger.info("Running in MOCK mode — no API calls will be made.")
    dashboard = PredictionDashboard(output_dir="predictions")

    dashboard.display_console(MOCK_PREDICTIONS, title="TODAY'S PREDICTIONS (MOCK)")
    html_path = dashboard.generate_html_dashboard(MOCK_PREDICTIONS)
    csv_path  = dashboard.export_csv(MOCK_PREDICTIONS)
    json_path = dashboard.export_json(MOCK_PREDICTIONS)

    logger.info("HTML dashboard : %s", html_path)
    logger.info("CSV export     : %s", csv_path)
    logger.info("JSON export    : %s", json_path)

    df = dashboard.to_dataframe(MOCK_PREDICTIONS)
    print("\n📋 DataFrame preview:")
    print(df[["home_team", "away_team", "predicted_goals", "confidence"]].to_string(index=False))


def run_live_pipeline() -> None:
    logger.info("Running in LIVE mode.")

    from client_football_data import FootballDataClient
    from client_odds_api import OddsAPIClient
    from scheduler import PredictionScheduler
    from config import FOOTBALL_DATA

    dashboard = PredictionDashboard(output_dir="predictions")

    # Step 1 — fixtures
    logger.info("Step 1/3: Fetching fixtures from API-Football...")
    football_client = FootballDataClient()
    predictions: list[ModelOutput] = []
    for league_code in FOOTBALL_DATA.league_codes:
        try:
            preds = football_client.build_model_outputs(league_code)
            predictions.extend(preds)
            logger.info("  League %s: %d fixtures", league_code, len(preds))
        except RuntimeError as exc:
            logger.warning("  League %s skipped — %s", league_code, exc)

    if not predictions:
        logger.error("No fixtures fetched. Check your API_FOOTBALL_KEY in config.py.")
        sys.exit(1)

    # Step 2 — odds / value bets
    logger.info("Step 2/3: Attaching value bets from The Odds API...")
    try:
        predictions = OddsAPIClient().attach_value_bets(predictions)
    except RuntimeError as exc:
        logger.warning("Odds API failed (%s) — continuing without value bets.", exc)

    # Step 3 — render
    logger.info("Step 3/3: Rendering dashboard...")
    dashboard.display_console(predictions, title="TODAY'S PREDICTIONS (LIVE)")
    dashboard.generate_html_dashboard(predictions)
    dashboard.export_csv(predictions)
    dashboard.export_json(predictions)

    # Start scheduler
    scheduler = PredictionScheduler(dashboard, output_dir="predictions")
    scheduler.start()
    logger.info(
        "\n✅ Scheduler running. Odds refresh every %d min. Press Ctrl+C to stop.\n",
        scheduler._config.odds_fetch_interval_minutes,
    )
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        scheduler.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer prediction pipeline demo.")
    parser.add_argument("--live", action="store_true",
                        help="Use real API data (requires keys in config.py)")
    args = parser.parse_args()

    if args.live:
        run_live_pipeline()
    else:
        run_mock_pipeline()


if __name__ == "__main__":
    main()
