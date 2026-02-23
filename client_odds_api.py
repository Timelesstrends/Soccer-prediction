"""
The Odds API client — fetches bookmaker odds and detects value bets.
Docs: https://the-odds-api.com/lol-of-the-api/

Value bet logic:
    edge = model_probability - implied_bookmaker_probability
    If edge > threshold, we have a value bet.
    Kelly stake = edge / (decimal_odds - 1)
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cache import DiskCache
from config import ODDS_API, OddsAPIConfig
from prediction_dashboard import ModelOutput, ValueBet

logger = logging.getLogger(__name__)

# Minimum edge to consider a bet worth flagging
VALUE_THRESHOLD: float = 0.05


@dataclass(frozen=True)
class BookmakerOdds:
    """Decimal odds for a single over/under market from one bookmaker."""
    bookmaker: str
    over_odds:  float
    under_odds: float

    @property
    def over_implied_prob(self) -> float:
        return 1.0 / self.over_odds

    @property
    def under_implied_prob(self) -> float:
        return 1.0 / self.under_odds


class OddsAPIClient:
    """
    Fetches live bookmaker odds across all configured leagues and
    attaches ValueBet objects to existing ModelOutput instances.
    """

    def __init__(
        self,
        config: OddsAPIConfig = ODDS_API,
        cache: Optional[DiskCache] = None,
    ) -> None:
        self._config = config
        self._cache  = cache or DiskCache()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_odds(self, sport_key: str) -> List[Dict]:
        """Return raw odds dicts for all upcoming events in one sport/league."""
        cache_key = f"odds_{sport_key}"
        cached = self._cache.get("odds", cache_key)
        if cached is not None:
            return cached

        params = {
            "apiKey":     self._config.key,
            "regions":    self._config.regions,
            "markets":    self._config.markets,
            "oddsFormat": self._config.odds_format,
        }
        url = (
            f"{self._config.base_url}/sports/{sport_key}/odds"
            f"?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        )
        data = self._request(url)
        self._cache.set("odds", cache_key, data)
        logger.info("Fetched odds for %d events (%s)", len(data), sport_key)
        return data

    def get_all_odds(self) -> Dict[str, Dict]:
        """
        Fetch odds for every configured league and return a single
        lookup dict: normalised "home|away" → event dict.
        """
        odds_map: Dict[str, Dict] = {}
        for sport_key in self._config.sport_keys:
            try:
                events = self.get_odds(sport_key)
                for ev in events:
                    key = (
                        self._normalise_name(ev["home_team"])
                        + "|"
                        + self._normalise_name(ev["away_team"])
                    )
                    odds_map[key] = ev
            except RuntimeError as exc:
                logger.warning("Skipping odds for %s — %s", sport_key, exc)
        return odds_map

    def attach_value_bets(self, predictions: List[ModelOutput]) -> List[ModelOutput]:
        """
        For each ModelOutput, search across ALL configured league odds
        and attach a ValueBet if edge exceeds VALUE_THRESHOLD.
        Mutates value_bet in-place and returns the same list.
        """
        odds_map = self.get_all_odds()
        logger.info("Odds map contains %d matchups across all leagues", len(odds_map))

        for pred in predictions:
            lookup_key = (
                self._normalise_name(pred.home_team)
                + "|"
                + self._normalise_name(pred.away_team)
            )
            event = odds_map.get(lookup_key)
            if event:
                value_bet = self._evaluate_value(pred, event)
                pred.value_bet = value_bet
                if value_bet:
                    logger.info(
                        "Value bet: %s vs %s — edge=%.1f%%",
                        pred.home_team, pred.away_team, value_bet.edge * 100,
                    )
            else:
                logger.debug("No odds found for %s vs %s", pred.home_team, pred.away_team)

        return predictions

    # ── Private helpers ───────────────────────────────────────────────────────

    def _evaluate_value(self, pred: ModelOutput, event: Dict) -> Optional[ValueBet]:
        """
        Compare model probability against best available bookmaker odds.
        Returns a ValueBet if edge exceeds VALUE_THRESHOLD, else None.
        """
        best_odds = self._best_totals_odds(event)
        if best_odds is None:
            return None

        over_edge  = pred.over_prob  - best_odds.over_implied_prob
        under_edge = pred.under_prob - best_odds.under_implied_prob

        if over_edge >= VALUE_THRESHOLD:
            kelly = over_edge / (best_odds.over_odds - 1)
            return ValueBet(
                recommendation=f"Over 2.5 @ {best_odds.over_odds} ({best_odds.bookmaker})",
                edge=round(over_edge, 4),
                kelly_stake=round(kelly * 100, 2),
            )

        if under_edge >= VALUE_THRESHOLD:
            kelly = under_edge / (best_odds.under_odds - 1)
            return ValueBet(
                recommendation=f"Under 2.5 @ {best_odds.under_odds} ({best_odds.bookmaker})",
                edge=round(under_edge, 4),
                kelly_stake=round(kelly * 100, 2),
            )

        return None

    def _best_totals_odds(self, event: Dict) -> Optional[BookmakerOdds]:
        """Find the bookmaker offering the best over odds for the 2.5 totals market."""
        best: Optional[BookmakerOdds] = None
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "totals":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                over_odds  = outcomes.get("Over",  0.0)
                under_odds = outcomes.get("Under", 0.0)
                if over_odds and under_odds:
                    candidate = BookmakerOdds(
                        bookmaker=bookmaker["title"],
                        over_odds=float(over_odds),
                        under_odds=float(under_odds),
                    )
                    if best is None or candidate.over_odds > best.over_odds:
                        best = candidate
        return best

    @staticmethod
    def _normalise_name(name: str) -> str:
        """Lowercase and strip punctuation for fuzzy team name matching."""
        return name.lower().replace(".", "").replace("-", " ").strip()

    def _request(self, url: str) -> Any:
        logger.debug("GET %s", url)
        try:
            with urllib.request.urlopen(url, timeout=self._config.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.error("Odds API HTTP %d", exc.code)
            raise RuntimeError(f"Odds API returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            logger.error("Odds API network error: %s", exc.reason)
            raise RuntimeError(f"Network error: {exc.reason}") from exc
