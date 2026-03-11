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
    HeadToHead, HomeAwaySplit, MatchResult, ModelOutput, TeamForm,
)

# ── Fitted model coefficients (loaded once, refreshed nightly) ───────────────
# If model_trainer has produced fitted coefficients they are used in place of
# the hand-tuned strength-ratio approach. Falls back silently if unavailable.
def _load_fitted_coefficients() -> Optional[Dict[str, Any]]:
    try:
        from model_trainer import load as _trainer_load
        return _trainer_load()
    except Exception:
        return None

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
        position_map = self._build_position_map(standings["total"])

        outputs: List[ModelOutput] = []
        for match in matches:
            try:
                outputs.append(
                    self._match_to_output(match, league_code, strength_map, home_map, away_map, position_map)
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
        position_map: Optional[Dict[str, int]] = None,
    ) -> ModelOutput:
        """
        Map a single match dict to ModelOutput using the 5-signal blend.

        Signal 1 — Season strength (45%)
            Derived from goals-for / goals-against across the full season.

        Signal 2 — Weighted recent form (20%)
            Exponentially weighted W/D/L across last 5 games.

        Signal 3 — Home/away venue split (20%)
            Venue-specific goals scored/conceded per game.

        Signal 4 — Head-to-head history (10%)
            Average goals each team has scored in their last 10 meetings.

        Signal 5 — League table position (5%)
            Top-half vs bottom-half position bonus/penalty.
            Rewards teams punching above or below their season average.

        After blending xG, builds a Dixon-Coles goal matrix to derive
        Home Win / Draw / Away Win probabilities and the most likely scoreline.
        """
        home_team  = match["homeTeam"]
        away_team  = match["awayTeam"]
        home_name  = home_team["name"]
        away_name  = away_team["name"]
        home_id    = home_team["id"]
        away_id    = away_team["id"]
        match_id   = match["id"]

        # ── Signal 1: season strength ─────────────────────────────────────
        # If fitted Dixon-Coles coefficients exist, use attack/defence ratings
        # derived from MLE. Otherwise fall back to the hand-tuned strength ratio.
        _fitted = _load_fitted_coefficients()
        if _fitted and "teams" in _fitted:
            _teams     = _fitted["teams"]
            _home_adv  = _fitted.get("home_advantage", 1.2)
            _home_coef = _teams.get(home_name, {})
            _away_coef = _teams.get(away_name, {})
            # λ = attack_home * defence_away * home_advantage
            home_strength = _home_coef.get("attack", 1.0)
            away_strength = _away_coef.get("attack", 1.0)
            xg_season_home = (
                _home_coef.get("attack",  1.0) *
                _away_coef.get("defence", 1.0) *
                _home_adv
            )
            xg_season_away = (
                _away_coef.get("attack",  1.0) *
                _home_coef.get("defence", 1.0)
            )
            logger.debug(
                "Using fitted coefficients for %s vs %s (home_adv=%.3f)",
                home_name, away_name, _home_adv,
            )
        else:
            home_strength  = strength_map.get(home_name, 1.0)
            away_strength  = strength_map.get(away_name, 1.0)
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

        # ── Signal 5: league table position ──────────────────────────────
        # Top teams get a small attacking bonus; bottom teams a penalty.
        # Capped at ±8% so position never overwhelms form or strength.
        total_teams  = 20
        pos_map      = position_map or {}
        home_pos     = pos_map.get(home_name, total_teams // 2)
        away_pos     = pos_map.get(away_name, total_teams // 2)

        # Position 1 → +0.08 boost, position 20 → -0.08 penalty
        home_pos_factor = ((total_teams / 2) - home_pos) / (total_teams / 2) * 0.08
        away_pos_factor = ((total_teams / 2) - away_pos) / (total_teams / 2) * 0.08

        xg_pos_home = xg_season_home * (1 + home_pos_factor)
        xg_pos_away = xg_season_away * (1 + away_pos_factor)

        # ── Blend all 5 signals (weights sum to 1.0) ──────────────────────
        W_POS        = 0.05
        w_season_adj = W_SEASON - W_POS   # 0.45

        xg_home = round(
            xg_season_home * w_season_adj +
            xg_form_home   * W_FORM       +
            xg_venue_home  * W_VENUE      +
            xg_h2h_home    * W_H2H        +
            xg_pos_home    * W_POS,
            2,
        )
        xg_away = round(
            xg_season_away * w_season_adj +
            xg_form_away   * W_FORM       +
            xg_venue_away  * W_VENUE      +
            xg_h2h_away    * W_H2H        +
            xg_pos_away    * W_POS,
            2,
        )

        # Floor: never predict less than 0.3 goals per team
        xg_home = max(xg_home, 0.3)
        xg_away = max(xg_away, 0.3)

        # ── Dixon-Coles 1X2 probabilities ─────────────────────────────────
        match_result = self._compute_match_result(
            xg_home, xg_away,
            home_position=home_pos,
            away_position=away_pos,
        )

        logger.debug(
            "%s (pos %d) vs %s (pos %d) | xG(%.2f,%.2f) "
            "1X2(%.0f%%/%.0f%%/%.0f%%) [%s]",
            home_name, home_pos, away_name, away_pos,
            xg_home, xg_away,
            match_result.home_prob * 100,
            match_result.draw_prob * 100,
            match_result.away_prob * 100,
            match_result.outcome,
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
            match_result        = match_result,
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

    def _build_position_map(self, standings_table: List[Dict]) -> Dict[str, int]:
        """
        Build a {team_name: position} map from the total standings table.
        Position 1 = top of the table.
        """
        pos_map: Dict[str, int] = {}
        for row in standings_table:
            try:
                team_name = row["team"]["name"]
                position  = row.get("position", 0)
                if position:
                    pos_map[team_name] = int(position)
            except (KeyError, TypeError):
                continue
        return pos_map

    @staticmethod
    def _dixon_coles_matrix(
        xg_home: float,
        xg_away: float,
        max_goals: int = 8,
    ) -> List[List[float]]:
        """
        Build the Dixon-Coles joint goal probability matrix.

        P(home=i, away=j) = Poisson(i|xg_home) * Poisson(j|xg_away) * rho(i,j)

        The rho correction adjusts the probability of low-scoring results
        (0-0, 1-0, 0-1, 1-1) which Poisson slightly misprices.

        Returns a (max_goals+1) x (max_goals+1) matrix where
        matrix[i][j] = probability of home scoring i, away scoring j.
        """
        rho = -0.13   # Dixon-Coles correction parameter (industry standard)

        def poisson_pmf(k: int, lam: float) -> float:
            return (lam ** k) * math.exp(-lam) / math.factorial(k)

        def dc_correction(i: int, j: int, lam1: float, lam2: float) -> float:
            if i == 0 and j == 0:
                return 1 - lam1 * lam2 * rho
            if i == 1 and j == 0:
                return 1 + lam2 * rho
            if i == 0 and j == 1:
                return 1 + lam1 * rho
            if i == 1 and j == 1:
                return 1 - rho
            return 1.0

        lam_h = max(xg_home, 0.1)
        lam_a = max(xg_away, 0.1)

        matrix = []
        for i in range(max_goals + 1):
            row = []
            for j in range(max_goals + 1):
                p = (
                    poisson_pmf(i, lam_h)
                    * poisson_pmf(j, lam_a)
                    * dc_correction(i, j, lam_h, lam_a)
                )
                row.append(max(p, 0.0))
            matrix.append(row)
        return matrix

    @staticmethod
    def _compute_match_result(
        xg_home: float,
        xg_away: float,
        home_position: int = 10,
        away_position: int = 10,
    ) -> "MatchResult":
        """
        Derive 1X2 probabilities and most likely scoreline from the
        Dixon-Coles goal matrix.
        """
        matrix  = FootballDataClient._dixon_coles_matrix(xg_home, xg_away)
        n       = len(matrix)

        home_prob = 0.0
        draw_prob = 0.0
        away_prob = 0.0
        best_prob = 0.0
        best_score = (1, 1)

        for i in range(n):
            for j in range(n):
                p = matrix[i][j]
                if i > j:
                    home_prob += p
                elif i == j:
                    draw_prob += p
                else:
                    away_prob += p
                if p > best_prob:
                    best_prob  = p
                    best_score = (i, j)

        # Normalise to sum to 1.0 (small floating point drift from rho correction)
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total

        return MatchResult(
            home_prob          = round(home_prob, 4),
            draw_prob          = round(draw_prob, 4),
            away_prob          = round(away_prob, 4),
            most_likely_score  = best_score,
            most_likely_prob   = round(best_prob, 4),
            home_position      = home_position,
            away_position      = away_position,
        )


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
