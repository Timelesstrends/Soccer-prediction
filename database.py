"""
database.py — SQLite persistence layer for the soccer prediction tracker.

Three tables:
    predictions  — every prediction logged the moment it is generated
    results      — actual final scores fetched after matches finish
    performance  — pre-computed accuracy stats (rebuilt nightly)

All writes are atomic via context managers.
Thread-safe: uses a module-level lock around every write.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = Path("predictions.db")
_lock   = threading.Lock()

# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS predictions (
    match_id            TEXT PRIMARY KEY,
    logged_at           TEXT NOT NULL,
    match_date          TEXT NOT NULL,
    league              TEXT NOT NULL,
    home_team           TEXT NOT NULL,
    away_team           TEXT NOT NULL,
    expected_home_goals REAL NOT NULL,
    expected_away_goals REAL NOT NULL,
    predicted_goals     REAL NOT NULL,
    over_prob           REAL NOT NULL,
    under_prob          REAL NOT NULL,
    confidence          TEXT NOT NULL,
    value_rec           TEXT,
    value_edge          REAL,
    value_kelly         REAL,
    home_strength       REAL,
    away_strength       REAL,
    home_momentum       REAL,
    away_momentum       REAL,
    h2h_avg_goals       REAL,
    home_venue_scored   REAL,
    away_venue_scored   REAL
);

CREATE TABLE IF NOT EXISTS results (
    match_id          TEXT PRIMARY KEY,
    fetched_at        TEXT NOT NULL,
    home_goals        INTEGER NOT NULL,
    away_goals        INTEGER NOT NULL,
    total_goals       INTEGER NOT NULL,
    over_correct      INTEGER NOT NULL,
    under_correct     INTEGER NOT NULL,
    predicted_correct INTEGER NOT NULL,
    value_bet_won     INTEGER,
    pnl               REAL,
    FOREIGN KEY (match_id) REFERENCES predictions(match_id)
);

CREATE TABLE IF NOT EXISTS performance (
    computed_at          TEXT NOT NULL,
    scope                TEXT NOT NULL,
    scope_type           TEXT NOT NULL,
    total_predictions    INTEGER NOT NULL,
    total_results        INTEGER NOT NULL,
    over_under_correct   INTEGER NOT NULL,
    over_under_accuracy  REAL NOT NULL,
    value_bets_total     INTEGER NOT NULL,
    value_bets_won       INTEGER NOT NULL,
    value_bet_win_rate   REAL NOT NULL,
    total_pnl            REAL NOT NULL,
    roi                  REAL NOT NULL,
    PRIMARY KEY (computed_at, scope, scope_type)
);

CREATE INDEX IF NOT EXISTS idx_predictions_date   ON predictions(match_date);
CREATE INDEX IF NOT EXISTS idx_predictions_league ON predictions(league);
CREATE INDEX IF NOT EXISTS idx_results_fetched    ON results(fetched_at);
"""


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def _connect(path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db(path: Path = DB_PATH) -> None:
    """Create all tables and indexes if they do not already exist."""
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)
    logger.info("Database initialised at %s", path)


# ── Predictions ───────────────────────────────────────────────────────────────

def log_prediction(pred: Any, path: Path = DB_PATH) -> bool:
    """
    Insert a ModelOutput into the predictions table.
    Skips silently if that match_id already exists (idempotent).
    Returns True if inserted, False if already present.
    """
    value_rec   = pred.value_bet.recommendation if pred.value_bet else None
    value_edge  = pred.value_bet.edge           if pred.value_bet else None
    value_kelly = pred.value_bet.kelly_stake    if pred.value_bet else None
    home_mom    = pred.home_form.momentum       if pred.home_form  else None
    away_mom    = pred.away_form.momentum       if pred.away_form  else None
    h2h_avg     = pred.h2h.avg_goals_total      if pred.h2h        else None
    home_vs     = pred.home_split.avg_scored    if pred.home_split else None
    away_vs     = pred.away_split.avg_scored    if pred.away_split else None

    sql = """
        INSERT OR IGNORE INTO predictions (
            match_id, logged_at, match_date, league,
            home_team, away_team,
            expected_home_goals, expected_away_goals, predicted_goals,
            over_prob, under_prob, confidence,
            value_rec, value_edge, value_kelly,
            home_strength, away_strength,
            home_momentum, away_momentum,
            h2h_avg_goals, home_venue_scored, away_venue_scored
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    params = (
        pred.match_id,
        datetime.utcnow().isoformat(timespec="seconds"),
        pred.date,
        pred.league,
        pred.home_team,
        pred.away_team,
        round(pred.expected_home_goals, 4),
        round(pred.expected_away_goals, 4),
        round(pred.predicted_goals, 4),
        round(pred.over_prob, 4),
        round(pred.under_prob, 4),
        pred.confidence.value,
        value_rec, value_edge, value_kelly,
        pred.home_strength, pred.away_strength,
        home_mom, away_mom, h2h_avg, home_vs, away_vs,
    )

    with _lock:
        with _connect(path) as conn:
            cur = conn.execute(sql, params)
            inserted = cur.rowcount > 0

    if inserted:
        logger.debug("Logged: %s vs %s", pred.home_team, pred.away_team)
    return inserted


def log_predictions_bulk(preds: List[Any], path: Path = DB_PATH) -> int:
    """Log a list of ModelOutputs. Returns count of newly inserted rows."""
    count = 0
    for pred in preds:
        try:
            if log_prediction(pred, path):
                count += 1
        except Exception as exc:
            logger.warning("Failed to log %s: %s", pred.match_id, exc)
    logger.info("Logged %d new predictions to DB", count)
    return count


def get_unresolved_predictions(
    before_date: Optional[str] = None,
    path: Path = DB_PATH,
) -> List[Dict]:
    """
    Return predictions that have no result yet.
    Optionally filtered to matches whose date is before `before_date`.
    """
    sql = """
        SELECT p.*
        FROM   predictions p
        LEFT   JOIN results r ON p.match_id = r.match_id
        WHERE  r.match_id IS NULL
    """
    params: tuple = ()
    if before_date:
        sql   += " AND p.match_date < ?"
        params = (before_date,)
    sql += " ORDER BY p.match_date ASC"

    with _connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_all_predictions(
    league:     Optional[str] = None,
    confidence: Optional[str] = None,
    limit:      int = 500,
    path: Path = DB_PATH,
) -> List[Dict]:
    """Return logged predictions with results joined in where available."""
    sql = """
        SELECT
            p.*,
            r.home_goals, r.away_goals, r.total_goals,
            r.over_correct, r.predicted_correct,
            r.value_bet_won, r.pnl
        FROM   predictions p
        LEFT   JOIN results r ON p.match_id = r.match_id
        WHERE  1=1
    """
    params: List[Any] = []
    if league:
        sql += " AND p.league = ?"
        params.append(league)
    if confidence:
        sql += " AND p.confidence = ?"
        params.append(confidence)
    sql += " ORDER BY p.match_date DESC LIMIT ?"
    params.append(limit)

    with _connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ── Results ───────────────────────────────────────────────────────────────────

def log_result(
    match_id:   str,
    home_goals: int,
    away_goals: int,
    path: Path = DB_PATH,
) -> bool:
    """
    Record the actual final score and compute outcome/P&L fields.
    Returns True if inserted, False if already present or no prediction found.
    """
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE match_id = ?", (match_id,)
        ).fetchone()

    if row is None:
        logger.warning("No prediction for match_id %s — skipping", match_id)
        return False

    pred        = dict(row)
    total_goals = home_goals + away_goals
    actual_over = total_goals > 2.5

    predicted_over    = pred["over_prob"] > pred["under_prob"]
    over_correct      = int(predicted_over == actual_over)
    under_correct     = int(not predicted_over == actual_over)
    predicted_correct = over_correct

    value_bet_won = None
    pnl           = 0.0

    if pred["value_rec"] and pred["value_kelly"]:
        implied_odds = _parse_odds(pred["value_rec"])
        kelly        = pred["value_kelly"] / 100.0
        if implied_odds and kelly > 0:
            rec_over = "Over" in pred["value_rec"]
            won      = (rec_over and actual_over) or (not rec_over and not actual_over)
            value_bet_won = int(won)
            pnl = kelly * (implied_odds - 1) if won else -kelly

    sql = """
        INSERT OR IGNORE INTO results (
            match_id, fetched_at,
            home_goals, away_goals, total_goals,
            over_correct, under_correct, predicted_correct,
            value_bet_won, pnl
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """
    with _lock:
        with _connect(path) as conn:
            cur = conn.execute(sql, (
                match_id,
                datetime.utcnow().isoformat(timespec="seconds"),
                home_goals, away_goals, total_goals,
                over_correct, under_correct, predicted_correct,
                value_bet_won, round(pnl, 4),
            ))
            inserted = cur.rowcount > 0

    if inserted:
        logger.info(
            "Result: %s %d-%d [%s] pnl=%.2f",
            match_id, home_goals, away_goals,
            "CORRECT" if predicted_correct else "WRONG", pnl,
        )
    return inserted


def _parse_odds(rec: str) -> Optional[float]:
    """Extract decimal odds from 'Over 2.5 @ 1.85 (Pinnacle)' style strings."""
    m = re.search(r"@\s*([\d.]+)", rec or "")
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ── Performance ───────────────────────────────────────────────────────────────

def rebuild_performance(path: Path = DB_PATH) -> None:
    """
    Recompute all performance rows from scratch.
    Writes one row each for: overall, every league, every confidence level.
    Called nightly after results are fetched.
    """
    computed_at = datetime.utcnow().isoformat(timespec="seconds")

    with _connect(path) as conn:
        conn.execute("DELETE FROM performance")

        leagues = [r[0] for r in conn.execute(
            "SELECT DISTINCT league FROM predictions"
        ).fetchall()]

        scopes: List[Tuple[str, str, str]] = [("overall", "overall", "1=1")]
        for lg in leagues:
            scopes.append((lg, "league", f"p.league = ?"))
        for conf in ("high", "medium", "low"):
            scopes.append((conf, "confidence", f"p.confidence = ?"))

        for scope, scope_type, where in scopes:
            # Build params for the WHERE clause
            if scope_type == "league":
                wp = (scope,)
            elif scope_type == "confidence":
                wp = (scope,)
            else:
                wp = ()

            sql = f"""
                SELECT
                    COUNT(p.match_id)                                  AS total_predictions,
                    COUNT(r.match_id)                                  AS total_results,
                    SUM(COALESCE(r.predicted_correct, 0))              AS over_under_correct,
                    COUNT(CASE WHEN p.value_rec IS NOT NULL THEN 1 END) AS value_bets_total,
                    SUM(COALESCE(r.value_bet_won, 0))                  AS value_bets_won,
                    SUM(COALESCE(r.pnl, 0))                            AS total_pnl
                FROM predictions p
                LEFT JOIN results r ON p.match_id = r.match_id
                WHERE {where}
            """
            row = conn.execute(sql, wp).fetchone()
            if not row or (row["total_results"] or 0) == 0:
                continue

            total_res = row["total_results"]
            ou_cor    = row["over_under_correct"] or 0
            vb_tot    = row["value_bets_total"]   or 0
            vb_won    = row["value_bets_won"]      or 0
            tot_pnl   = row["total_pnl"]           or 0.0

            conn.execute("""
                INSERT INTO performance (
                    computed_at, scope, scope_type,
                    total_predictions, total_results,
                    over_under_correct, over_under_accuracy,
                    value_bets_total, value_bets_won, value_bet_win_rate,
                    total_pnl, roi
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                computed_at, scope, scope_type,
                row["total_predictions"], total_res,
                ou_cor, round(ou_cor / total_res, 4),
                vb_tot, vb_won,
                round(vb_won / vb_tot, 4) if vb_tot > 0 else 0.0,
                round(tot_pnl, 4),
                round((tot_pnl / vb_tot) * 100, 2) if vb_tot > 0 else 0.0,
            ))

    logger.info("Performance stats rebuilt at %s", computed_at)


def get_performance(path: Path = DB_PATH) -> Dict[str, Any]:
    """Return latest performance stats structured for the API."""
    sql = """
        SELECT * FROM performance
        WHERE computed_at = (SELECT MAX(computed_at) FROM performance)
        ORDER BY scope_type, scope
    """
    with _connect(path) as conn:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]

    out: Dict[str, Any] = {"overall": {}, "by_league": {}, "by_confidence": {}}
    for row in rows:
        entry = {k: row[k] for k in (
            "total_predictions", "total_results",
            "over_under_accuracy", "value_bets_total", "value_bets_won",
            "value_bet_win_rate", "total_pnl", "roi",
        )}
        if row["scope_type"] == "overall":
            out["overall"] = entry
        elif row["scope_type"] == "league":
            out["by_league"][row["scope"]] = entry
        elif row["scope_type"] == "confidence":
            out["by_confidence"][row["scope"]] = entry
    return out


def get_recent_results(limit: int = 100, path: Path = DB_PATH) -> List[Dict]:
    """Return the most recent resolved predictions for the history browser."""
    sql = """
        SELECT
            p.match_id, p.match_date, p.league,
            p.home_team, p.away_team,
            p.predicted_goals, p.over_prob, p.under_prob, p.confidence,
            p.value_rec, p.value_edge, p.value_kelly,
            p.expected_home_goals, p.expected_away_goals,
            r.home_goals, r.away_goals, r.total_goals,
            r.predicted_correct, r.value_bet_won, r.pnl
        FROM   predictions p
        JOIN   results r ON p.match_id = r.match_id
        ORDER  BY p.match_date DESC, p.home_team ASC
        LIMIT  ?
    """
    with _connect(path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_db_summary(path: Path = DB_PATH) -> Dict[str, int]:
    """Row counts for each table — used by the health endpoint."""
    with _connect(path) as conn:
        preds   = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        perf    = conn.execute("SELECT COUNT(*) FROM performance").fetchone()[0]
    return {"predictions": preds, "results": results, "performance": perf}
