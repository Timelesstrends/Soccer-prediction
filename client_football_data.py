"""
Football-Data.org client — fixtures, results, team statistics.
API docs: https://www.football-data.org/documentation/quickstart

Key differences from API-Football:
  - Auth via X-Auth-Token header (not a query param)
  - Leagues identified by string codes (PL, PD, BL1, SA, FL1)
  - Endpoint: /v4/competitions/{code}/matches
  - Match object uses "homeTeam"/"awayTeam" and "utcDate"
  - No season restriction on free tier — 2025/26 works out of the box

xG Model signals (blended):
  1. Season strength  — goals-for/against ratio from TOTAL standings   (50%)
  2. Weighted form    — exponentially weighted W/D/L last 5 games       (20%)
  3. Home/away split  — venue-specific goals scored/conceded per game   (20%)
  4. Head-to-head     — historical results & avg goals in this fixture  (10%)
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from cache import DiskCache
from config import FOOTBALL_DATA, FootballDataConfig
from prediction_dashboard import (
    FORM_WINDOW, H2H_WINDOW,
    HeadToHead, HomeAwaySplit, ModelOutput, TeamForm,
)

# ── Rate limiting ──────────────────────────────────────────────────────────────
# Free tier: 10 requests/minute → 6 seconds minimum between calls
_RATE_LIMIT_DELAY: float = 6.5
_last_request_time: float = 0.0

# ── Blend weights (must sum to 1.0) ───────────────────────────────────────────
W_SEASON   = 0.50   # season-long goals-for/against ratio
W_FORM     = 0.20   # weighted recent form (last 5 games)
W_VENUE    = 0.20   # home/away split at this specific venue
W_H2H      = 0.10   # head-to-head history between these two teams

# League average goals per team per game — used when form fallback needed
LEAGUE_AVG_GOALS_PER_TEAM: float = 1.3

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
            "Football-Data: %d matches fetched for %s (%s to %s)",
            len(matches), league_code, date_from, date_to,
        )
        return matches

    def get_standings(self, league_code: str) -> Dict[str, List[Dict]]:
        """
        Return standings split into TOTAL, HOME and AWAY tables.
        Returns a dict with keys 'total', 'home', 'away'.
        Each value is a list of team row dicts.
        """
        cache_key = f"standings_all_{league_code}"
        cached = self._cache.get("fixtures", cache_key)
        if cached is not None:
            return cached

        data      = self._request(f"competitions/{league_code}/standings", {})
        standings = data.get("standings", [])

        result = {"total": [], "home": [], "away": []}
        for s in standings:
            stype = s.get("type", "").upper()
            if stype == "TOTAL":
                result["total"] = s.get("table", [])
            elif stype == "HOME":
                result["home"]  = s.get("table", [])
            elif stype == "AWAY":
                result["away"]  = s.get("table", [])

        self._cache.set("fixtures", cache_key, result)
        return result

    def get_team_form(self, team_id: int, team_name: str) -> Optional[TeamForm]:
        """
        Fetch the last FORM_WINDOW finished matches for a team.
        Returns a TeamForm with exponentially weighted momentum.
        """
        cache_key = f"form_{team_id}"
        cached = self._cache.get("fixtures", cache_key)
        if cached is not None:
            return TeamForm(**cached)

        try:
            data = self._request(
                f"teams/{team_id}/matches",
                {"status": "FINISHED", "limit": str(FORM_WINDOW)},
            )
        except Exception as exc:
            logger.warning("Could not fetch form for %s: %s", team_name, exc)
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
                home_id    = match["homeTeam"]["id"]
                home_goals = match["score"]["fullTime"]["home"] or 0
                away_goals = match["score"]["fullTime"]["away"] or 0
                is_home    = (home_id == team_id)

                scored   = home_goals if is_home else away_goals
                conceded = away_goals if is_home else home_goals

                result = "W" if scored > conceded else ("D" if scored == conceded else "L")
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

    def get_head_to_head(
        self,
        match_id: int,
        home_team_name: str,
        away_team_name: str,
    ) -> Optional[HeadToHead]:
        """
        Fetch H2H history for a specific fixture using football-data.org's
        /matches/{id}/head2head endpoint.

        Returns None if the call fails or there's no history.
        """
        cache_key = f"h2h_{match_id}"
        cached = self._cache.get("fixtures", cache_key)
        if cached is not None:
            return HeadToHead(**cached)

        try:
            data = self._request(
                f"matches/{match_id}/head2head",
                {"limit": str(H2H_WINDOW)},
            )
        except Exception as exc:
            logger.warning("H2H fetch failed for match %s: %s", match_id, exc)
            return None

        matches = data.get("matches", [])
        if not matches:
            return None

        # Identify home team ID from the aggregates block if available,
        # otherwise infer from the first match
        agg = data.get("aggregates", {})
        home_team_id = None
        away_team_id = None
        try:
            home_team_id = agg["homeTeam"]["id"]
            away_team_id = agg["awayTeam"]["id"]
        except (KeyError, TypeError):
            # Fall back: find team IDs from match records
            for m in matches:
                try:
                    if m["homeTeam"]["name"] == home_team_name:
                        home_team_id = m["homeTeam"]["id"]
                        away_team_id = m["awayTeam"]["id"]
                        break
                    elif m["awayTeam"]["name"] == home_team_name:
                        home_team_id = m["awayTeam"]["id"]
                        away_team_id = m["homeTeam"]["id"]
                        break
                except (KeyError, TypeError):
                    continue

        home_wins = 0
        draws     = 0
        away_wins = 0
        total_goals_home_team = 0   # goals scored by home_team across all H2H
        total_goals_away_team = 0   # goals scored by away_team across all H2H
        total_goals_all       = 0
        played = 0

        for m in matches:
            try:
                ft = m["score"]["fullTime"]
                mh = ft["home"]
                ma = ft["away"]
                if mh is None or ma is None:
                    continue

                match_home_id = m["homeTeam"]["id"]
                # Determine which team was which in this historical match
                if match_home_id == home_team_id:
                    ht_scored = mh
                    at_scored = ma
                elif match_home_id == away_team_id:
                    ht_scored = ma
                    at_scored = mh
                else:
                    # Can't map — skip this match
                    continue

                total_goals_home_team += ht_scored
                total_goals_away_team += at_scored
                total_goals_all       += mh + ma
                played += 1

                if ht_scored > at_scored:
                    home_wins += 1
                elif ht_scored == at_scored:
                    draws += 1
                else:
                    away_wins += 1
            except (KeyError, TypeError):
                continue

        if played == 0:
            return None

        h2h = HeadToHead(
            home_team       = home_team_name,
            away_team       = away_team_name,
            matches_played  = played,
            home_wins       = home_wins,
            draws           = draws,
            away_wins       = away_wins,
            avg_goals_total = round(total_goals_all / played, 2),
            home_avg_scored = round(total_goals_home_team / played, 2),
            away_avg_scored = round(total_goals_away_team / played, 2),
        )
        self._cache.set("fixtures", cache_key, {
            "home_team":       h2h.home_team,
            "away_team":       h2h.away_team,
            "matches_played":  h2h.matches_played,
            "home_wins":       h2h.home_wins,
            "draws":           h2h.draws,
            "away_wins":       h2h.away_wins,
            "avg_goals_total": h2h.avg_goals_total,
            "home_avg_scored": h2h.home_avg_scored,
            "away_avg_scored": h2h.away_avg_scored,
        })
        logger.info(
            "H2H %s vs %s: %d games, home wins %d, draws %d, away wins %d, avg %.1f goals",
            home_team_name, away_team_name,
            played, home_wins, draws, away_wins, h2h.avg_goals_total,
        )
        return h2h

    def build_model_outputs(self, league_code: str) -> List[ModelOutput]:
        """
        Fetch upcoming matches for a league and return ModelOutput objects.
        Blends season strength, recent form, home/away splits and H2H history.
        """
        matches      = self.get_matches(league_code)
        standings    = self.get_standings(league_code)
        strength_map = self._build_strength_map(standings["total"])
        home_map     = self._build_venue_split_map(standings["home"],  "home")
        away_map     = self._build_venue_split_map(standings["away"],  "away")

        outputs: List[ModelOutput] = []
        for match in matches:
            try:
                outputs.append(
                    self._match_to_output(match, league_code, strength_map, home_map, away_map)
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
        home_map: Dict[str, HomeAwaySplit],
        away_map: Dict[str, HomeAwaySplit],
    ) -> ModelOutput:
        """
        Map a single match dict to ModelOutput using the 4-signal blend.

        Signal 1 — Season strength (50%)
            Derived from goals-for / goals-against across the full season.
            Captures overall quality regardless of venue.

        Signal 2 — Weighted recent form (20%)
            Momentum score from last 5 games with exponential recency weighting.
            A win last week counts ~5x more than a win 5 weeks ago.

        Signal 3 — Home/away venue split (20%)
            How many goals the home team scores/concedes at home, and how
            many the away team scores/concedes away. This captures teams
            like Atletico Madrid who defend tightly at home but attack away.

        Signal 4 — Head-to-head history (10%)
            Average goals scored by each team in their last 10 meetings.
            Fixtures between specific clubs often have a consistent pattern
            regardless of current form (e.g. low-scoring derbies).
        """
        home_team  = match["homeTeam"]
        away_team  = match["awayTeam"]
        home_name  = home_team["name"]
        away_name  = away_team["name"]
        home_id    = home_team["id"]
        away_id    = away_team["id"]
        match_id   = match["id"]

        # ── Signal 1: season strength ─────────────────────────────────────
        home_strength = strength_map.get(home_name, 1.0)
        away_strength = strength_map.get(away_name, 1.0)
        xg_season_home = home_strength * 1.35   # home advantage multiplier
        xg_season_away = away_strength * 1.10   # slight penalty playing away

        # ── Signal 2: weighted recent form ───────────────────────────────
        try:
            home_form = self.get_team_form(home_id, home_name)
        except Exception as exc:
            logger.warning("Home form failed for %s: %s", home_name, exc)
            home_form = None

        try:
            away_form = self.get_team_form(away_id, away_name)
        except Exception as exc:
            logger.warning("Away form failed for %s: %s", away_name, exc)
            away_form = None

        # Momentum (0–1) → scale to goals per game range
        xg_form_home = (
            home_form.momentum * LEAGUE_AVG_GOALS_PER_TEAM * 2
            if home_form else xg_season_home
        )
        xg_form_away = (
            away_form.momentum * LEAGUE_AVG_GOALS_PER_TEAM * 2
            if away_form else xg_season_away
        )

        # ── Signal 3: venue split ─────────────────────────────────────────
        home_split = home_map.get(home_name)
        away_split = away_map.get(away_name)

        xg_venue_home = (
            home_split.xg_estimate if home_split else xg_season_home
        )
        xg_venue_away = (
            away_split.xg_estimate if away_split else xg_season_away
        )

        # ── Signal 4: head-to-head ────────────────────────────────────────
        h2h = None
        xg_h2h_home = xg_season_home
        xg_h2h_away = xg_season_away
        try:
            h2h = self.get_head_to_head(match_id, home_name, away_name)
            if h2h and h2h.matches_played >= 3:
                xg_h2h_home = h2h.home_avg_scored
                xg_h2h_away = h2h.away_avg_scored
        except Exception as exc:
            logger.warning("H2H failed for %s vs %s: %s", home_name, away_name, exc)

        # ── Blend all 4 signals ───────────────────────────────────────────
        xg_home = round(
            xg_season_home * W_SEASON +
            xg_form_home   * W_FORM   +
            xg_venue_home  * W_VENUE  +
            xg_h2h_home    * W_H2H,
            2,
        )
        xg_away = round(
            xg_season_away * W_SEASON +
            xg_form_away   * W_FORM   +
            xg_venue_away  * W_VENUE  +
            xg_h2h_away    * W_H2H,
            2,
        )

        # Sanity floor — never predict less than 0.3 goals per team
        xg_home = max(xg_home, 0.3)
        xg_away = max(xg_away, 0.3)

        logger.debug(
            "%s vs %s | season(%.2f,%.2f) form(%.2f,%.2f) "
            "venue(%.2f,%.2f) h2h(%.2f,%.2f) → xG(%.2f,%.2f)",
            home_name, away_name,
            xg_season_home, xg_season_away,
            xg_form_home,   xg_form_away,
            xg_venue_home,  xg_venue_away,
            xg_h2h_home,    xg_h2h_away,
            xg_home,        xg_away,
        )

        return ModelOutput(
            match_id            = str(match_id),
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
            home_split          = home_split,
            away_split          = away_split,
            h2h                 = h2h,
        )

    def _build_strength_map(self, standings_table: List[Dict]) -> Dict[str, float]:
        """
        Derive a strength rating (0.5–2.0) for each team from their
        season goals-for / goals-against ratio.
        """
        strength_map: Dict[str, float] = {}
        for row in standings_table:
            try:
                team_name     = row["team"]["name"]
                goals_for     = row.get("goalsFor",     1) or 1
                goals_against = row.get("goalsAgainst", 1) or 1
                played        = row.get("playedGames",  1) or 1

                avg_for     = goals_for     / played
                avg_against = goals_against / played
                ratio = avg_for / max(avg_against, 0.5)
                strength_map[team_name] = round(min(max(ratio, 0.5), 2.0), 3)
            except (KeyError, TypeError, ZeroDivisionError):
                continue
        return strength_map

    def _build_venue_split_map(
        self,
        standings_table: List[Dict],
        venue: str,
    ) -> Dict[str, HomeAwaySplit]:
        """
        Build a lookup of team_name → HomeAwaySplit from HOME or AWAY standings.
        """
        split_map: Dict[str, HomeAwaySplit] = {}
        for row in standings_table:
            try:
                team_name = row["team"]["name"]
                split_map[team_name] = HomeAwaySplit(
                    team_name     = team_name,
                    venue         = venue,
                    played        = row.get("playedGames",  0) or 0,
                    wins          = row.get("won",          0) or 0,
                    draws         = row.get("draw",         0) or 0,
                    losses        = row.get("lost",         0) or 0,
                    goals_for     = row.get("goalsFor",     0) or 0,
                    goals_against = row.get("goalsAgainst", 0) or 0,
                )
            except (KeyError, TypeError):
                continue
        return split_map

    @staticmethod
    def _poisson_distribution(expected_total: float) -> Dict[str, float]:
        """
        Build a Poisson goal probability distribution from an expected total.
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
        Enforces the 10 req/min rate limit and caches responses.
        """
        global _last_request_time

        query = "&".join(f"{k}={v}" for k, v in params.items())
        url   = f"{self._config.base_url}/{endpoint}"
        if query:
            url = f"{url}?{query}"

        elapsed = time.time() - _last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            wait = _RATE_LIMIT_DELAY - elapsed
            logger.debug("Rate limit: waiting %.1fs", wait)
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
            logger.error("Football-Data HTTP %d for %s — %s", exc.code, url, body)
            if exc.code == 401:
                raise RuntimeError("Invalid API key. Check FOOTBALL_DATA_KEY.") from exc
            if exc.code == 429:
                raise RuntimeError("Rate limit hit. Cache will prevent this on re-run.") from exc
            raise RuntimeError(f"Football-Data HTTP {exc.code}: {body}") from exc

        except urllib.error.URLError as exc:
            logger.error("Network error: %s", exc.reason)
            raise RuntimeError(f"Network error: {exc.reason}") from exc
