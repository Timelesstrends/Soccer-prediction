"""
Enhanced Display & Reporting Module for Soccer Prediction System
Supports: Console (colorama), HTML Dashboard (Jinja2), CSV, JSON
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from colorama import Fore, Style, init as colorama_init
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
GOAL_THRESHOLD: float = 2.5
MAX_DISTRIBUTION_GOALS: int = 7
CONSOLE_BAR_WIDTH: int = 30
FORM_WINDOW: int = 5                          # number of recent games to consider
H2H_WINDOW: int = 10                          # number of H2H games to consider
TEMPLATE_DIR: Path = Path(__file__).parent / "templates"


# ── Enums ──────────────────────────────────────────────────────────────────────
class ConfidenceLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ValueBet:
    """Structured value bet recommendation."""
    recommendation: str
    edge: float          # model probability minus implied bookmaker probability
    kelly_stake: float   # Kelly criterion stake as a percentage


@dataclass(frozen=True)
class TeamForm:
    """
    A team's form across their last FORM_WINDOW matches.
    Results are stored oldest → newest so the last element is most recent.
    Each result is 'W', 'D', or 'L' from that team's perspective.
    """
    team_name: str
    results: List[str]          # e.g. ['W', 'D', 'L', 'W', 'W']
    goals_scored: List[int]     # goals this team scored in each game
    goals_conceded: List[int]   # goals this team conceded in each game

    @property
    def wins(self) -> int:
        return self.results.count("W")

    @property
    def draws(self) -> int:
        return self.results.count("D")

    @property
    def losses(self) -> int:
        return self.results.count("L")

    @property
    def points(self) -> int:
        """Total points from last N games (W=3, D=1, L=0)."""
        return self.wins * 3 + self.draws

    @property
    def avg_scored(self) -> float:
        return round(sum(self.goals_scored) / max(len(self.goals_scored), 1), 2)

    @property
    def avg_conceded(self) -> float:
        return round(sum(self.goals_conceded) / max(len(self.goals_conceded), 1), 2)

    @property
    def form_rating(self) -> float:
        """
        Normalised form rating between 0.0 (terrible) and 1.0 (perfect).
        Based on points per game over the last N matches (max = 3 pts/game).
        """
        max_points = len(self.results) * 3
        return round(self.points / max_points, 3) if max_points else 0.5

    @property
    def momentum(self) -> float:
        """
        Weighted form rating — recent games count more than older ones.
        Weights: oldest=1, …, newest=N (exponential-style linear ramp).
        Returns a value between 0.0 and 1.0.
        """
        if not self.results:
            return 0.5
        point_map = {"W": 3, "D": 1, "L": 0}
        weights = list(range(1, len(self.results) + 1))
        weighted_pts  = sum(point_map[r] * w for r, w in zip(self.results, weights))
        max_weighted  = sum(w * 3 for w in weights)
        return round(weighted_pts / max_weighted, 3)

    @property
    def form_string(self) -> str:
        """Human-readable form string e.g. 'W W D L W'."""
        return " ".join(self.results)


@dataclass(frozen=True)
class HomeAwaySplit:
    """
    A team's record split by home or away games only.
    Used to capture teams that are dominant at home but weak away (or vice versa).
    """
    team_name: str
    venue: str              # 'home' or 'away'
    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int

    @property
    def avg_scored(self) -> float:
        return round(self.goals_for / max(self.played, 1), 3)

    @property
    def avg_conceded(self) -> float:
        return round(self.goals_against / max(self.played, 1), 3)

    @property
    def xg_estimate(self) -> float:
        """
        Simple xG estimate from home/away split.
        Returns goals scored per game at this venue, floored at 0.3.
        """
        return max(self.avg_scored, 0.3)

    @property
    def strength_rating(self) -> float:
        """
        Venue-specific strength between 0.5 and 2.0.
        Based on goals-for / goals-against ratio at this venue.
        """
        ratio = self.goals_for / max(self.goals_against, 0.5 * max(self.played, 1))
        return round(min(max(ratio, 0.5), 2.0), 3)


@dataclass(frozen=True)
class HeadToHead:
    """
    Head-to-head record between two specific teams.
    home_team and away_team here refer to the UPCOMING match, not H2H venues.
    """
    home_team: str
    away_team: str
    matches_played: int
    home_wins: int          # times home_team won in H2H
    draws: int
    away_wins: int          # times away_team won in H2H
    avg_goals_total: float  # average total goals in H2H matches
    home_avg_scored: float  # avg goals scored by home_team in H2H
    away_avg_scored: float  # avg goals scored by away_team in H2H

    @property
    def home_win_rate(self) -> float:
        """Home team H2H win rate (0.0–1.0)."""
        return round(self.home_wins / max(self.matches_played, 1), 3)

    @property
    def away_win_rate(self) -> float:
        """Away team H2H win rate (0.0–1.0)."""
        return round(self.away_wins / max(self.matches_played, 1), 3)

    @property
    def draw_rate(self) -> float:
        return round(self.draws / max(self.matches_played, 1), 3)

    @property
    def is_high_scoring(self) -> bool:
        """True if H2H fixtures tend to go over 2.5 goals."""
        return self.avg_goals_total > 2.5

    @property
    def home_dominance(self) -> float:
        """
        How much the home team dominates this H2H fixture.
        Positive = home team tends to win, negative = away team tends to win.
        Range roughly -1.0 to +1.0.
        """
        return round(self.home_win_rate - self.away_win_rate, 3)


@dataclass
class ModelOutput:
    """Raw output from the prediction model — no presentation concerns."""
    match_id: str
    date: str
    league: str
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    home_strength: float
    away_strength: float
    goal_distribution: Dict[str, float]   # keys: "exactly_0", "exactly_1", …
    value_bet: Optional[ValueBet] = None
    home_form: Optional[TeamForm] = None       # last FORM_WINDOW games for home team
    away_form: Optional[TeamForm] = None       # last FORM_WINDOW games for away team
    home_split: Optional[HomeAwaySplit] = None # home team's home-venue record
    away_split: Optional[HomeAwaySplit] = None # away team's away-venue record
    h2h: Optional[HeadToHead] = None           # head-to-head history

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def predicted_goals(self) -> float:
        return self.expected_home_goals + self.expected_away_goals

    @property
    def over_prob(self) -> float:
        under_keys = {f"exactly_{i}" for i in range(int(GOAL_THRESHOLD) + 1)}
        return 1.0 - sum(
            v for k, v in self.goal_distribution.items() if k in under_keys
        )

    @property
    def under_prob(self) -> float:
        return 1.0 - self.over_prob

    @property
    def confidence(self) -> ConfidenceLevel:
        if self.over_prob >= 0.65 or self.under_prob >= 0.65:
            return ConfidenceLevel.HIGH
        if self.over_prob >= 0.55 or self.under_prob >= 0.55:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @property
    def goal_distribution_items(self) -> List[Tuple[int, float]]:
        """Sorted (goal_count, probability) pairs for display."""
        return [
            (i, self.goal_distribution.get(f"exactly_{i}", 0.0))
            for i in range(MAX_DISTRIBUTION_GOALS)
        ]


# ── Summary stats ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SummaryStats:
    total: int
    high_confidence: int
    avg_goals: float
    value_bets: int

    @classmethod
    def from_predictions(cls, predictions: List[ModelOutput]) -> "SummaryStats":
        if not predictions:
            return cls(total=0, high_confidence=0, avg_goals=0.0, value_bets=0)
        return cls(
            total=len(predictions),
            high_confidence=sum(
                1 for p in predictions if p.confidence is ConfidenceLevel.HIGH
            ),
            avg_goals=float(np.mean([p.predicted_goals for p in predictions])),
            value_bets=sum(1 for p in predictions if p.value_bet is not None),
        )


# ── Renderers ─────────────────────────────────────────────────────────────────
class ConsoleRenderer:
    """Renders predictions to stdout using colorama for cross-platform colour."""

    _RESULT_COLOURS = {
        "W": Fore.GREEN,
        "D": Fore.YELLOW,
        "L": Fore.RED,
    }

    def __init__(self) -> None:
        colorama_init(autoreset=True)

    def render(
        self, predictions: List[ModelOutput], title: str = "MATCH PREDICTIONS"
    ) -> None:
        if not predictions:
            print("No predictions to display.")
            return

        print("\n" + "=" * 80)
        print(f"{title:^80}")
        print("=" * 80)

        for pred in predictions:
            self._render_match(pred)

        print("=" * 80)
        self._render_summary(SummaryStats.from_predictions(predictions))

    def _render_match(self, pred: ModelOutput) -> None:
        print(f"\n{pred.home_team} vs {pred.away_team}")
        print(f"  League: {pred.league}  |  Date: {pred.date}")
        print(f"  Predicted goals: {pred.predicted_goals:.1f}")
        print(f"  Over 2.5: {pred.over_prob:.0%}  |  Under 2.5: {pred.under_prob:.0%}")
        print(f"  Confidence: {pred.confidence.value.upper()}")

        if pred.home_form:
            print(f"  {pred.home_team} form: {pred.home_form.form_string} "
                  f"(momentum {pred.home_form.momentum:.0%})")
        if pred.away_form:
            print(f"  {pred.away_team} form: {pred.away_form.form_string} "
                  f"(momentum {pred.away_form.momentum:.0%})")

        if pred.home_split:
            print(f"  {pred.home_team} at home: "
                  f"{pred.home_split.avg_scored:.2f} scored / "
                  f"{pred.home_split.avg_conceded:.2f} conceded per game")
        if pred.away_split:
            print(f"  {pred.away_team} away: "
                  f"{pred.away_split.avg_scored:.2f} scored / "
                  f"{pred.away_split.avg_conceded:.2f} conceded per game")

        if pred.h2h and pred.h2h.matches_played > 0:
            print(f"  H2H ({pred.h2h.matches_played} games): "
                  f"{pred.h2h.home_team} wins {pred.h2h.home_win_rate:.0%} | "
                  f"draws {pred.h2h.draw_rate:.0%} | "
                  f"avg {pred.h2h.avg_goals_total:.1f} goals")

        if pred.value_bet:
            print(f"  VALUE BET: {pred.value_bet.recommendation} "
                  f"(edge {pred.value_bet.edge:.0%}, "
                  f"Kelly {pred.value_bet.kelly_stake:.1f}%)")

        self._render_distribution(pred.goal_distribution_items)

    def _render_distribution(self, items: List[Tuple[int, float]]) -> None:
        print("  Goal distribution:")
        for goal_count, prob in items:
            bar = "█" * int(prob * CONSOLE_BAR_WIDTH)
            print(f"    {goal_count} goals: {bar:<{CONSOLE_BAR_WIDTH}} {prob:.1%}")

    def _render_summary(self, stats: SummaryStats) -> None:
        pct = (
            f"{stats.high_confidence / stats.total:.0%}"
            if stats.total else "n/a"
        )
        print(f"\n📊 SUMMARY")
        print(f"   Total Matches:           {stats.total}")
        print(f"   High Confidence:         {stats.high_confidence} ({pct})")
        print(f"   Average Predicted Goals: {stats.avg_goals:.2f}")
        print(f"   Value Opportunities:     {stats.value_bets}")


class HtmlRenderer:
    """Renders predictions to a self-contained HTML file using Jinja2."""

    def __init__(self, template_dir: Path = TEMPLATE_DIR) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html"]),
        )

    def render(
        self,
        predictions: List[ModelOutput],
        output_path: Path,
        title: str = "Soccer Prediction Dashboard",
    ) -> None:
        template = self._env.get_template("dashboard.html")
        stats = SummaryStats.from_predictions(predictions)

        html = template.render(
            predictions=predictions,
            stats=stats,
            goal_threshold=GOAL_THRESHOLD,
            form_window=FORM_WINDOW,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            title=title,
        )

        try:
            output_path.write_text(html, encoding="utf-8")
            logger.info("HTML dashboard written to %s", output_path)
        except OSError as exc:
            logger.error("Failed to write HTML dashboard: %s", exc)
            raise


class DataExporter:
    """Exports predictions to CSV or JSON."""

    def to_csv(self, predictions: List[ModelOutput], output_path: Path) -> None:
        rows = [self._flatten(p) for p in predictions]
        if not rows:
            logger.warning("No predictions to export.")
            return
        try:
            with output_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info("CSV exported to %s (%d rows)", output_path, len(rows))
        except OSError as exc:
            logger.error("Failed to write CSV: %s", exc)
            raise

    def to_dataframe(self, predictions: List[ModelOutput]) -> pd.DataFrame:
        return pd.DataFrame([self._flatten(p) for p in predictions])

    def to_json(self, predictions: List[ModelOutput], output_path: Path) -> None:
        payload = {
            "generated_at": datetime.now().isoformat(),
            "count": len(predictions),
            "predictions": [self._to_dict(p) for p in predictions],
        }
        try:
            output_path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            logger.info("JSON exported to %s", output_path)
        except OSError as exc:
            logger.error("Failed to write JSON: %s", exc)
            raise

    def _flatten(self, pred: ModelOutput) -> Dict:
        row: Dict = {
            "match_id":             pred.match_id,
            "date":                 pred.date,
            "league":               pred.league,
            "home_team":            pred.home_team,
            "away_team":            pred.away_team,
            "expected_home_goals":  pred.expected_home_goals,
            "expected_away_goals":  pred.expected_away_goals,
            "predicted_goals":      round(pred.predicted_goals, 4),
            "over_prob":            round(pred.over_prob, 4),
            "under_prob":           round(pred.under_prob, 4),
            "confidence":           pred.confidence.value,
            "home_strength":        pred.home_strength,
            "away_strength":        pred.away_strength,
        }
        if pred.home_form:
            row["home_form"]          = pred.home_form.form_string
            row["home_momentum"]      = pred.home_form.momentum
            row["home_avg_scored"]    = pred.home_form.avg_scored
            row["home_avg_conceded"]  = pred.home_form.avg_conceded
        else:
            row["home_form"] = row["home_momentum"] = ""
            row["home_avg_scored"] = row["home_avg_conceded"] = None

        if pred.away_form:
            row["away_form"]          = pred.away_form.form_string
            row["away_momentum"]      = pred.away_form.momentum
            row["away_avg_scored"]    = pred.away_form.avg_scored
            row["away_avg_conceded"]  = pred.away_form.avg_conceded
        else:
            row["away_form"] = row["away_momentum"] = ""
            row["away_avg_scored"] = row["away_avg_conceded"] = None

        if pred.home_split:
            row["home_venue_scored"]   = pred.home_split.avg_scored
            row["home_venue_conceded"] = pred.home_split.avg_conceded
        else:
            row["home_venue_scored"] = row["home_venue_conceded"] = None

        if pred.away_split:
            row["away_venue_scored"]   = pred.away_split.avg_scored
            row["away_venue_conceded"] = pred.away_split.avg_conceded
        else:
            row["away_venue_scored"] = row["away_venue_conceded"] = None

        if pred.h2h:
            row["h2h_played"]    = pred.h2h.matches_played
            row["h2h_home_wins"] = pred.h2h.home_wins
            row["h2h_draws"]     = pred.h2h.draws
            row["h2h_away_wins"] = pred.h2h.away_wins
            row["h2h_avg_goals"] = pred.h2h.avg_goals_total
        else:
            row["h2h_played"] = row["h2h_home_wins"] = row["h2h_draws"] = None
            row["h2h_away_wins"] = row["h2h_avg_goals"] = None

        if pred.value_bet:
            row["value_recommendation"] = pred.value_bet.recommendation
            row["value_edge"]    = round(pred.value_bet.edge, 4)
            row["kelly_stake"]   = round(pred.value_bet.kelly_stake, 4)
        else:
            row["value_recommendation"] = ""
            row["value_edge"] = row["kelly_stake"] = None

        return row

    def _to_dict(self, pred: ModelOutput) -> Dict:
        def _form_to_dict(f: Optional[TeamForm]) -> Optional[Dict]:
            if f is None:
                return None
            return {
                "team_name":      f.team_name,
                "results":        f.results,
                "goals_scored":   f.goals_scored,
                "goals_conceded": f.goals_conceded,
                "form_string":    f.form_string,
                "form_rating":    f.form_rating,
                "momentum":       f.momentum,
                "avg_scored":     f.avg_scored,
                "avg_conceded":   f.avg_conceded,
                "points":         f.points,
            }

        def _split_to_dict(s: Optional[HomeAwaySplit]) -> Optional[Dict]:
            if s is None:
                return None
            return {
                "team_name":    s.team_name,
                "venue":        s.venue,
                "played":       s.played,
                "wins":         s.wins,
                "draws":        s.draws,
                "losses":       s.losses,
                "goals_for":    s.goals_for,
                "goals_against":s.goals_against,
                "avg_scored":   s.avg_scored,
                "avg_conceded": s.avg_conceded,
            }

        def _h2h_to_dict(h: Optional[HeadToHead]) -> Optional[Dict]:
            if h is None:
                return None
            return {
                "home_team":       h.home_team,
                "away_team":       h.away_team,
                "matches_played":  h.matches_played,
                "home_wins":       h.home_wins,
                "draws":           h.draws,
                "away_wins":       h.away_wins,
                "avg_goals_total": h.avg_goals_total,
                "home_avg_scored": h.home_avg_scored,
                "away_avg_scored": h.away_avg_scored,
                "home_win_rate":   h.home_win_rate,
                "away_win_rate":   h.away_win_rate,
                "home_dominance":  h.home_dominance,
                "is_high_scoring": h.is_high_scoring,
            }

        return {
            "match_id":            pred.match_id,
            "date":                pred.date,
            "league":              pred.league,
            "home_team":           pred.home_team,
            "away_team":           pred.away_team,
            "expected_home_goals": pred.expected_home_goals,
            "expected_away_goals": pred.expected_away_goals,
            "predicted_goals":     pred.predicted_goals,
            "over_prob":           pred.over_prob,
            "under_prob":          pred.under_prob,
            "confidence":          pred.confidence.value,
            "home_strength":       pred.home_strength,
            "away_strength":       pred.away_strength,
            "goal_distribution":   pred.goal_distribution,
            "home_form":           _form_to_dict(pred.home_form),
            "away_form":           _form_to_dict(pred.away_form),
            "home_split":          _split_to_dict(pred.home_split),
            "away_split":          _split_to_dict(pred.away_split),
            "h2h":                 _h2h_to_dict(pred.h2h),
            "value_bet":           asdict(pred.value_bet) if pred.value_bet else None,
        }


# ── Façade ────────────────────────────────────────────────────────────────────
class PredictionDashboard:
    """Thin façade that wires renderers and exporters together."""

    def __init__(self, output_dir: str = "predictions") -> None:
        self.output_dir = Path(output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"Cannot create output directory '{output_dir}': {exc}"
            ) from exc

        self._console  = ConsoleRenderer()
        self._html     = HtmlRenderer()
        self._exporter = DataExporter()

    def display_console(
        self, predictions: List[ModelOutput], title: str = "MATCH PREDICTIONS"
    ) -> None:
        self._console.render(predictions, title)

    def generate_html_dashboard(
        self, predictions: List[ModelOutput], filename: str = "dashboard.html"
    ) -> Path:
        output_path = self.output_dir / filename
        self._html.render(predictions, output_path)
        return output_path

    def export_csv(
        self, predictions: List[ModelOutput], filename: str = "predictions.csv"
    ) -> Path:
        output_path = self.output_dir / filename
        self._exporter.to_csv(predictions, output_path)
        return output_path

    def export_json(
        self, predictions: List[ModelOutput], filename: str = "predictions.json"
    ) -> Path:
        output_path = self.output_dir / filename
        self._exporter.to_json(predictions, output_path)
        return output_path

    def to_dataframe(self, predictions: List[ModelOutput]) -> pd.DataFrame:
        return self._exporter.to_dataframe(predictions)
