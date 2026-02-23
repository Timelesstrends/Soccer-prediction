"""
Football-Data.org client — fixtures, results, team statistics.
API docs: https://www.football-data.org/documentation/quickstart

Key differences from API-Football:
  - Auth via X-Auth-Token header (not a query param)
  - Leagues identified by string codes (PL, PD, BL1, SA, FL1)
  - Endpoint: /v4/competitions/{code}/matches
  - Match object uses "homeTeam"/"awayTeam" and "utcDate"
  - No season restriction on free tier — 2025/26 works out of the box
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from cache import DiskCache
from config import FOOTBALL_DATA, FootballDataConfig
from prediction_dashboard import FORM_WINDOW, ModelOutput, TeamForm

# Seconds between API calls to respect the 10 req/min free tier limit
_RATE_LIMIT_DELAY: float = 6.5
_last_request_time: float = 0.0

# How much form momentum influences the final xG (0 = ignore, 1 = all form)
FORM_WEIGHT: float = 0.30   # 30% form, 70% season strength

logger = logging.getLogger(__name__)


class FootballDataClient:
    """
    Fetches upcoming fixtures from Football-Data.org and maps them
    into the project's ModelOutput structure.

    Caches all responses locally to stay within the 10 req/min free limit.
    """

    def __init__(
        self,
        config: FootballDataConfig = FOOTBALL_DATA,
        cache: Optional[DiskCache] = None,
    ) -> None:
        self._config = config
        self._cache  = cache or DiskCache()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_matches(
        self,
        league_code: str,
        date_from: Optional[date] = None,
        date_to:   Optional[date] = None,
        status:    str = "SCHEDULED",
    ) -> List[Dict]:
        """
        Return upcoming matches for a league as raw API dicts.
        Defaults to a 14-day window starting today.

        status options: SCHEDULED, LIVE, IN_PLAY, PAUSED,
                        FINISHED, POSTPONED, CANCELLED, SUSPENDED
        """
        today     = date.today()
        date_from = date_from or today
        date_to   = date_to   or (today + timedelta(days=14))

        cache_key = f"{league_code}_{date_from}_{date_to}_{status}"
        cached = self._cache.get("fixtures", cache_key)
        if cached is not None:
            return cached

        params = {
            "dateFrom": str(date_from),
            "dateTo":   str(date_to),
            "status":   status,
        }
        data    = self._request(f"competitions/{league_code}/matches", params)
        matches = data.get("matches", [])

        self._cache.set("fixtures", cache_key, matches)
        logger.info(
            "Football-Data: %d matches fetched for %s (%s → %s)",
            len(matches), league_code, date_from, date_to,
        )
        return matches

    def get_standings(self, league_code: str) -> List[Dict]:
        """Return current standings table for strength calculations."""
        cache_key = f"standings_{league_code}"
        cached = self._cache.get("fixtures", cache_key)
        if cached is not None:
            return cached

        data     = self._request(f"competitions/{league_code}/standings", {})
        standings = data.get("standings", [])
        # We only want the TOTAL standing (not HOME/AWAY splits)
        total = next(
            (s["table"] for s in standings if s.get("type") == "TOTAL"), []
        )
        self._cache.set("fixtures", cache_key, total)
        return total

    def get_team_form(self, team_id: int, team_name: str) -> Optional[TeamForm]:
        """
        Fetch the last FORM_WINDOW finished matches for a team and
        return a TeamForm object summarising their recent results.
        Returns None if the API call fails or there is no data.
        """
        cache_key = f"form_{team_id}"
        cached = self._cache.get("fixtures", cache_key)
        if cached is not None:
            # Re-hydrate from cache dict
            return TeamForm(**cached)

        try:
            data = self._request(
                f"teams/{team_id}/matches",
                {"status": "FINISHED", "limit": str(FORM_WINDOW)},
            )
        except RuntimeError as exc:
            logger.warning("Could not fetch form for team %d: %s", team_id, exc)
            return None

        matches = data.get("matches", [])
        if not matches:
            return None

        results: List[str] = []
        goals_scored: List[int] = []
        goals_conceded: List[int] = []

        # API returns newest first — reverse so oldest → newest
        for match in reversed(matches[-FORM_WINDOW:]):
            try:
                home_id   = match["homeTeam"]["id"]
                home_goals = match["score"]["fullTime"]["home"] or 0
                away_goals = match["score"]["fullTime"]["away"] or 0
                is_home    = (home_id == team_id)

                scored    = home_goals if is_home else away_goals
                conceded  = away_goals if is_home else home_goals

                if scored > conceded:
                    result = "W"
                elif scored == conceded:
                    result = "D"
                else:
                    result = "L"

                results.append(result)
                goals_scored.append(scored)
                goals_conceded.append(conceded)
            except (KeyError, TypeError):
                continue

        if not results:
            return None

        form = TeamForm(
            team_name=team_name,
            results=results,
            goals_scored=goals_scored,
            goals_conceded=goals_conceded,
        )

        # Cache as plain dict (dataclass isn't JSON serialisable directly)
        self._cache.set("fixtures", cache_key, {
            "team_name":      form.team_name,
            "results":        form.results,
            "goals_scored":   form.goals_scored,
            "goals_conceded": form.goals_conceded,
        })

        logger.info(
            "Form for %s: %s (momentum=%.0f%%)",
            team_name, form.form_string, form.momentum * 100,
        )
        return form

    def build_model_outputs(self, league_code: str) -> List[ModelOutput]:
        """
        Fetch upcoming matches for a league and return ModelOutput objects.
        Each output includes season strength AND last-5-game form,
        blended together using FORM_WEIGHT.
        """
        matches      = self.get_matches(league_code)
        standings    = self.get_standings(league_code)
        strength_map = self._build_strength_map(standings)

        outputs: List[ModelOutput] = []
        for match in matches:
            try:
                outputs.append(
                    self._match_to_output(match, league_code, strength_map)
                )
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping match %s — mapping error: %s",
                    match.get("id", "?"), exc,
                )
        return outputs

    # ── Private helpers ───────────────────────────────────────────────────────

    def _match_to_output(
        self,
        match: Dict,
        league_code: str,
        strength_map: Dict[str, float],
    ) -> ModelOutput:
        """
        Map a single Football-Data.org match dict → ModelOutput.

        xG calculation blends two signals:
          1. Season strength  — goals-for/against ratio from standings (70%)
          2. Form momentum    — weighted W/D/L over last 5 games         (30%)

        Formula:
          base_xg    = season_strength * home/away_multiplier
          form_xg    = form_momentum   * league_avg_goals_per_team
          final_xg   = base_xg * (1 - FORM_WEIGHT) + form_xg * FORM_WEIGHT
        """
        home_team   = match["homeTeam"]
        away_team   = match["awayTeam"]
        home_name   = home_team["name"]
        away_name   = away_team["name"]
        home_id     = home_team["id"]
        away_id     = away_team["id"]

        home_strength = strength_map.get(home_name, 1.0)
        away_strength = strength_map.get(away_name, 1.0)

        # Fetch last-5 form (may be None if API call fails)
        home_form = self.get_team_form(home_id, home_name)
        away_form = self.get_team_form(away_id, away_name)

        # Base xG from season strength + home advantage
        base_xg_home = home_strength * 1.35
        base_xg_away = away_strength * 1.10

        # Form-adjusted xG — scale momentum (0–1) around a 1.3 goals/game average
        LEAGUE_AVG_GOALS_PER_TEAM = 1.3
        form_xg_home = (
            home_form.momentum * LEAGUE_AVG_GOALS_PER_TEAM * 2
            if home_form else base_xg_home
        )
        form_xg_away = (
            away_form.momentum * LEAGUE_AVG_GOALS_PER_TEAM * 2
            if away_form else base_xg_away
        )

        # Blend: 70% season strength, 30% recent form
        xg_home = round(
            base_xg_home * (1 - FORM_WEIGHT) + form_xg_home * FORM_WEIGHT, 2
        )
        xg_away = round(
            base_xg_away * (1 - FORM_WEIGHT) + form_xg_away * FORM_WEIGHT, 2
        )

        return ModelOutput(
            match_id            = str(match["id"]),
            date                = match["utcDate"][:10],
            league              = match["competition"]["name"],
            home_team           = home_name,
            away_team           = away_name,
            expected_home_goals = xg_home,
            expected_away_goals = xg_away,
            home_strength       = home_strength,
            away_strength       = away_strength,
            goal_distribution   = self._poisson_distribution(xg_home + xg_away),
            home_form           = home_form,
            away_form           = away_form,
        )

    def _build_strength_map(self, standings_table: List[Dict]) -> Dict[str, float]:
        """
        Derive a strength rating (0.5–2.0) for each team from their
        goals-for / goals-against ratio in the standings table.
        Returns an empty dict if standings are unavailable.
        """
        strength_map: Dict[str, float] = {}
        for row in standings_table:
            team_name  = row["team"]["name"]
            goals_for  = row.get("goalsFor",     1)
            goals_against = row.get("goalsAgainst", 1)
            played     = row.get("playedGames",  1) or 1

            avg_for     = goals_for     / played
            avg_against = goals_against / played

            # Ratio capped between 0.5 and 2.0 to avoid extreme outliers
            ratio = avg_for / max(avg_against, 0.5)
            strength_map[team_name] = round(min(max(ratio, 0.5), 2.0), 3)

        return strength_map

    @staticmethod
    def _poisson_distribution(expected_total: float) -> Dict[str, float]:
        """
        Build a Poisson goal probability distribution from an expected total.
        Used as a model placeholder until a proper ML model provides one.
        """
        lam = max(expected_total, 0.1)
        dist: Dict[str, float] = {}
        remaining = 1.0
        for k in range(6):
            p = (lam ** k * math.exp(-lam)) / math.factorial(k)
            dist[f"exactly_{k}"] = round(p, 4)
            remaining -= p
        dist["exactly_6"] = round(max(remaining, 0.0), 4)
        return dist

    def _request(self, endpoint: str, params: Dict[str, str]) -> Any:
        """
        Authenticated GET request to Football-Data.org.
        Uses X-Auth-Token header — the only auth method supported.
        Enforces a minimum delay between calls to respect the 10 req/min limit.
        Cached responses bypass this delay entirely.
        """
        global _last_request_time

        query = "&".join(f"{k}={v}" for k, v in params.items())
        url   = f"{self._config.base_url}/{endpoint}"
        if query:
            url = f"{url}?{query}"

        # Rate limiting — wait if the last request was too recent
        elapsed = time.time() - _last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            wait = _RATE_LIMIT_DELAY - elapsed
            logger.debug("Rate limit: waiting %.1fs before next request", wait)
            time.sleep(wait)

        req = urllib.request.Request(
            url,
            headers={
                "X-Auth-Token": self._config.key,
                "Accept":       "application/json",
            },
        )

        logger.debug("GET %s", url)
        _last_request_time = time.time()

        try:
            with urllib.request.urlopen(
                req, timeout=self._config.timeout_seconds
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error(
                "Football-Data HTTP %d for %s — %s", exc.code, url, body
            )
            if exc.code == 401:
                raise RuntimeError(
                    "Invalid API key. Check FOOTBALL_DATA_KEY in config.py."
                ) from exc
            if exc.code == 429:
                raise RuntimeError(
                    "Rate limit hit (10 req/min on free tier). "
                    "The cache will prevent this on repeated runs."
                ) from exc
            raise RuntimeError(
                f"Football-Data returned HTTP {exc.code}: {body}"
            ) from exc

        except urllib.error.URLError as exc:
            logger.error("Football-Data network error: %s", exc.reason)
            raise RuntimeError(f"Network error: {exc.reason}") from exc
