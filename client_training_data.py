"""
Training data clients:
  - StatsBombClient  — free open data (event-level, shot maps, xG)
  - UnderstatClient  — xG & match stats scraped from embedded page JSON

Both clients return pandas DataFrames suitable for model training.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import pandas as pd

from cache import DiskCache
from config import STATSBOMB, UNDERSTAT, StatsBombConfig, UnderstatConfig

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# StatsBomb — free open data from GitHub
# ══════════════════════════════════════════════════════════════════════════════

class StatsBombClient:
    """
    Accesses StatsBomb's public open-data repository on GitHub.
    No API key required.

    Key datasets:
        competitions  — list of available competitions/seasons
        matches       — fixture list per competition/season
        events        — full event stream per match (shots, passes, etc.)
        lineups       — squad lineups per match
    """

    def __init__(
        self,
        config: StatsBombConfig = STATSBOMB,
        cache: Optional[DiskCache] = None,
    ) -> None:
        self._config = config
        self._cache  = cache or DiskCache()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_competitions(self) -> pd.DataFrame:
        """Return all available competitions as a DataFrame."""
        data = self._fetch_json("competitions", "competitions.json")
        df = pd.DataFrame(data)
        logger.info("StatsBomb: %d competitions available", len(df))
        return df

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Return all matches for a competition/season."""
        path = f"matches/{competition_id}/{season_id}.json"
        data = self._fetch_json(f"matches_{competition_id}_{season_id}", path)
        df = pd.DataFrame(data)
        logger.info(
            "StatsBomb: %d matches for comp=%d season=%d",
            len(df), competition_id, season_id,
        )
        return df

    def get_events(self, match_id: int) -> pd.DataFrame:
        """
        Return the full event stream for a match.
        Each row is one event (shot, pass, dribble, etc.).
        """
        path = f"events/{match_id}.json"
        data = self._fetch_json(f"events_{match_id}", path)
        df = pd.json_normalize(data)
        return df

    def get_shots(self, match_id: int) -> pd.DataFrame:
        """Convenience wrapper — returns only shot events with xG values."""
        events = self.get_events(match_id)
        if events.empty or "type.name" not in events.columns:
            return pd.DataFrame()
        shots = events[events["type.name"] == "Shot"].copy()
        return shots

    def build_xg_training_data(
        self, competition_id: int, season_id: int
    ) -> pd.DataFrame:
        """
        Aggregate per-match xG totals across a full season.
        Returns one row per match with columns:
            match_id, home_team, away_team, home_xg, away_xg,
            home_goals, away_goals, date
        """
        matches = self.get_matches(competition_id, season_id)
        rows: List[Dict] = []

        for _, match in matches.iterrows():
            match_id = int(match["match_id"])
            shots    = self.get_shots(match_id)

            if shots.empty:
                continue

            home_team = match["home_team"]["home_team_name"]
            away_team = match["away_team"]["away_team_name"]

            home_xg = shots.loc[
                shots["possession_team.name"] == home_team,
                "shot.statsbomb_xg"
            ].sum() if "shot.statsbomb_xg" in shots.columns else 0.0

            away_xg = shots.loc[
                shots["possession_team.name"] == away_team,
                "shot.statsbomb_xg"
            ].sum() if "shot.statsbomb_xg" in shots.columns else 0.0

            rows.append({
                "match_id":    match_id,
                "date":        match.get("match_date", ""),
                "home_team":   home_team,
                "away_team":   away_team,
                "home_xg":     round(float(home_xg), 3),
                "away_xg":     round(float(away_xg), 3),
                "home_goals":  match.get("home_score", 0),
                "away_goals":  match.get("away_score", 0),
            })

        df = pd.DataFrame(rows)
        logger.info(
            "StatsBomb: built xG training data — %d matches", len(df)
        )
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_json(self, cache_key: str, path: str) -> Any:
        cached = self._cache.get("statsbomb", cache_key)
        if cached is not None:
            return cached

        url = f"{self._config.base_url}/{path}"
        logger.debug("StatsBomb GET %s", url)

        try:
            with urllib.request.urlopen(url, timeout=self._config.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.error("StatsBomb HTTP %d for %s", exc.code, url)
            raise RuntimeError(f"StatsBomb returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            logger.error("StatsBomb network error: %s", exc.reason)
            raise RuntimeError(f"Network error: {exc.reason}") from exc

        self._cache.set("statsbomb", cache_key, data)
        return data


# ══════════════════════════════════════════════════════════════════════════════
# Understat — xG data scraped from embedded page JSON
# ══════════════════════════════════════════════════════════════════════════════

class UnderstatClient:
    """
    Scrapes xG and match data from Understat.com.
    Understat embeds all data as JSON inside <script> tags —
    no HTML parsing library required.

    Available leagues: EPL, La_liga, Bundesliga, Serie_A, Ligue_1, RFPL
    """

    # Regex to extract the embedded JSON from Understat pages
    _JSON_RE = re.compile(r"JSON\.parse\('(.+?)'\)", re.DOTALL)

    def __init__(
        self,
        config: UnderstatConfig = UNDERSTAT,
        cache: Optional[DiskCache] = None,
    ) -> None:
        self._config = config
        self._cache  = cache or DiskCache()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_league_matches(self, league: str, season: int) -> pd.DataFrame:
        """
        Return all matches for a league/season with xG values.
        `league` should be one of the keys in UnderstatConfig.league_slugs
        (e.g. "Premier League").

        DataFrame columns:
            id, datetime, home_team, away_team,
            home_goals, away_goals, home_xg, away_xg,
            home_result (W/D/L from home perspective)
        """
        slug      = self._config.league_slugs.get(league, league)
        cache_key = f"{slug}_{season}"

        cached = self._cache.get("understat", cache_key)
        if cached is not None:
            return pd.DataFrame(cached)

        url  = f"{self._config.base_url}/league/{slug}/{season}"
        html = self._fetch_html(url)
        data = self._extract_matches_json(html)

        df = self._normalise_matches(data)
        self._cache.set("understat", cache_key, df.to_dict("records"))
        logger.info(
            "Understat: %d matches for %s %d", len(df), league, season
        )
        return df

    def get_team_xg(self, league: str, season: int) -> pd.DataFrame:
        """
        Aggregate per-team xG totals for a season.
        Returns one row per team with: team, xg_for, xg_against,
        goals_for, goals_against, matches_played.
        """
        matches = self.get_league_matches(league, season)
        if matches.empty:
            return pd.DataFrame()

        home = matches.rename(columns={
            "home_team": "team", "home_xg": "xg_for",
            "away_xg": "xg_against", "home_goals": "goals_for",
            "away_goals": "goals_against",
        })[["team", "xg_for", "xg_against", "goals_for", "goals_against"]]

        away = matches.rename(columns={
            "away_team": "team", "away_xg": "xg_for",
            "home_xg": "xg_against", "away_goals": "goals_for",
            "home_goals": "goals_against",
        })[["team", "xg_for", "xg_against", "goals_for", "goals_against"]]

        combined = pd.concat([home, away], ignore_index=True)
        aggregated = (
            combined
            .groupby("team")
            .agg(
                xg_for        = ("xg_for",        "sum"),
                xg_against    = ("xg_against",    "sum"),
                goals_for     = ("goals_for",      "sum"),
                goals_against = ("goals_against",  "sum"),
                matches_played = ("xg_for",        "count"),
            )
            .reset_index()
        )
        aggregated["xg_per_match"]     = (
            aggregated["xg_for"] / aggregated["matches_played"]
        ).round(3)
        aggregated["xg_against_per_match"] = (
            aggregated["xg_against"] / aggregated["matches_played"]
        ).round(3)

        return aggregated.sort_values("xg_for", ascending=False)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_html(self, url: str) -> str:
        """Fetch raw HTML from Understat with a browser-like User-Agent."""
        cached = self._cache.get("understat", f"html_{url}")
        if cached is not None:
            return cached

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SoccerBot/1.0)"},
        )
        logger.debug("Understat GET %s", url)
        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                html = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            logger.error("Understat fetch error: %s", exc)
            raise RuntimeError(f"Understat fetch failed: {exc}") from exc

        self._cache.set("understat", f"html_{url}", html)
        return html

    def _extract_matches_json(self, html: str) -> List[Dict]:
        """Pull the datesData JSON array out of the embedded script tag."""
        match = self._JSON_RE.search(html)
        if not match:
            raise ValueError("Could not find embedded JSON in Understat page.")

        # Understat escapes the JSON with unicode escapes — decode carefully
        raw = match.group(1).encode("utf-8").decode("unicode_escape")
        return json.loads(raw)

    @staticmethod
    def _normalise_matches(raw: List[Dict]) -> pd.DataFrame:
        """Flatten Understat's nested match structure into a tidy DataFrame."""
        rows = []
        for entry in raw:
            for match in (entry if isinstance(entry, list) else [entry]):
                try:
                    rows.append({
                        "id":          match.get("id"),
                        "datetime":    match.get("datetime"),
                        "home_team":   match["h"]["title"],
                        "away_team":   match["a"]["title"],
                        "home_goals":  int(match["goals"]["h"] or 0),
                        "away_goals":  int(match["goals"]["a"] or 0),
                        "home_xg":     float(match["xG"]["h"] or 0),
                        "away_xg":     float(match["xG"]["a"] or 0),
                    })
                except (KeyError, TypeError, ValueError) as exc:
                    logger.debug("Skipping malformed match entry: %s", exc)

        return pd.DataFrame(rows)
