"""
Central configuration for all API integrations.
Replace placeholder values with your real keys before running.

Recommended: load secrets from environment variables so keys
never live inside your code:
    export FOOTBALL_DATA_KEY="your_real_key"    (Mac/Linux/Termux)
    $env:FOOTBALL_DATA_KEY="your_real_key"      (Windows PowerShell)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FootballDataConfig:
    """
    https://www.football-data.org — fixtures, results, standings.
    Free forever for the top competitions. Supports the live 2025/26 season.
    Authentication: X-Auth-Token request header (not a query parameter).
    Rate limit: 10 requests/minute on the free tier.
    """
    base_url: str = "https://api.football-data.org/v4"
    key: str = field(
        default_factory=lambda: os.getenv(
            "FOOTBALL_DATA_KEY", "d038cf5d6a0f45e88e6a8715e6169549"
        )
    )
    # League codes used by football-data.org (string codes, not numeric IDs)
    league_codes: tuple = (
        "PL",   # Premier League
        "PD",   # La Liga (Primera Division)
        "BL1",  # Bundesliga
        "SA",   # Serie A
        "FL1",  # Ligue 1
    )
    timeout_seconds: int = 10


@dataclass(frozen=True)
class OddsAPIConfig:
    """https://the-odds-api.com — bookmaker odds for value bet detection."""
    base_url: str = "https://api.the-odds-api.com/v4"
    key: str = field(
        default_factory=lambda: os.getenv("ODDS_API_KEY", "01cb4e593c5f5b2d3e7ee61c77595323")
    )
    # Sport keys for all 5 leagues we track
    sport_keys: tuple = (
        "soccer_epl",               # Premier League
        "soccer_spain_la_liga",     # La Liga
        "soccer_germany_bundesliga", # Bundesliga
        "soccer_italy_serie_a",     # Serie A
        "soccer_france_ligue_one",  # Ligue 1
    )
    regions: str = "uk,eu"
    markets: str = "totals"
    odds_format: str = "decimal"
    timeout_seconds: int = 10


@dataclass(frozen=True)
class StatsBombConfig:
    """
    https://github.com/statsbomb/open-data
    No API key required — fetched directly from the public GitHub repo.
    """
    base_url: str = (
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    )
    timeout_seconds: int = 15


@dataclass(frozen=True)
class UnderstatConfig:
    """
    https://understat.com — xG data scraped from embedded page JSON.
    No API key required.
    """
    base_url: str = "https://understat.com"
    league_slugs: dict = field(default_factory=lambda: {
        "Premier League": "EPL",
        "La Liga":        "La_liga",
        "Bundesliga":     "Bundesliga",
        "Serie A":        "Serie_A",
        "Ligue 1":        "Ligue_1",
    })
    timeout_seconds: int = 15


@dataclass(frozen=True)
class CacheConfig:
    """Local disk cache settings."""
    cache_dir: Path = Path(".cache")
    ttl: dict = field(default_factory=lambda: {
        "fixtures":  3_600,
        "odds":        900,
        "statsbomb": 86_400,
        "understat":  3_600,
    })


@dataclass(frozen=True)
class SchedulerConfig:
    """Automatic fetch intervals."""
    fixtures_fetch_times: tuple = ((6, 0), (18, 0))
    odds_fetch_interval_minutes: int = 15
    training_data_fetch_day: str = "monday"


# ── Singleton instances used throughout the project ───────────────────────────
FOOTBALL_DATA = FootballDataConfig()
ODDS_API       = OddsAPIConfig()
STATSBOMB      = StatsBombConfig()
UNDERSTAT      = UnderstatConfig()
CACHE          = CacheConfig()
SCHEDULER      = SchedulerConfig()
