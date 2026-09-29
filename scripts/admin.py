"""
Admin panel for Dizzy (RDM Chatbot).

Self-contained module: local SQLite storage for chat history / errors,
malicious-request detection, stats, and the HTML dashboard + JSON API.
`main.py` only needs to `include_router(admin.router)` and call the
`log_chat` / `log_error` / `register_status_provider` hooks below.
"""
from __future__ import annotations

import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import psutil
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# ──────────────────────────────────────────────
#  Local storage (SQLite)
# ──────────────────────────────────────────────
DB_PATH = Path(
    os.getenv(
        "ADMIN_DB_PATH",
        Path(__file__).resolve().parent.parent / "data" / "admin.db",
    )
)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_db_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row


def _init_db() -> None:
    with _db_lock, _conn:
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ip TEXT,
                flagged INTEGER NOT NULL DEFAULT 0,
                matched_keywords TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id)")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_created ON chats(created_at)")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_flagged ON chats(flagged)")


_init_db()


def log_chat(
    user_id: str,
    role: str,
    content: str,
    ip: Optional[str] = None,
    matched_keywords: Optional[list[str]] = None,
) -> int:
    """Insert a chat row and return its id (used by append_flag for late flags)."""
    flagged = 1 if matched_keywords else 0
    keywords_str = ",".join(matched_keywords) if matched_keywords else None
    with _db_lock, _conn:
        cur = _conn.execute(
            "INSERT INTO chats (user_id, role, content, ip, flagged, matched_keywords, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, role, content, ip, flagged, keywords_str, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def append_flag(chat_id: int, keywords: list[str]) -> None:
    """Flag an already-logged chat row, e.g. once the model's own security
    filter (a separate detection path from `detect_malicious`) trips
    partway through generation."""
    if not keywords:
        return
    with _db_lock, _conn:
        row = _conn.execute(
            "SELECT matched_keywords FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
        if row is None:
            return
        existing = {kw for kw in (row["matched_keywords"] or "").split(",") if kw}
        existing.update(keywords)
        _conn.execute(
            "UPDATE chats SET flagged = 1, matched_keywords = ? WHERE id = ?",
            (",".join(sorted(existing)), chat_id),
        )


def log_error(source: str, message: str) -> None:
    with _db_lock, _conn:
        _conn.execute(
            "INSERT INTO errors (source, message, created_at) VALUES (?, ?, ?)",
            (source, message, datetime.utcnow().isoformat()),
        )
    print(f"[admin] error logged from {source}: {message}")


def get_recent_chats(limit: int = 200) -> list[dict]:
    with _db_lock:
        cur = _conn.execute(
            "SELECT id, user_id, role, content, ip, flagged, matched_keywords, created_at "
            "FROM chats ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_flagged_chats(limit: int = 200) -> list[dict]:
    with _db_lock:
        cur = _conn.execute(
            "SELECT id, user_id, role, content, ip, matched_keywords, created_at "
            "FROM chats WHERE flagged = 1 ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_recent_errors(limit: int = 100) -> list[dict]:
    with _db_lock:
        cur = _conn.execute(
            "SELECT id, source, message, created_at FROM errors ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def get_active_user_count(window_minutes: int = 5) -> int:
    cutoff = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat()
    with _db_lock:
        cur = _conn.execute(
            "SELECT COUNT(DISTINCT user_id) AS n FROM chats WHERE created_at >= ?",
            (cutoff,),
        )
        return cur.fetchone()["n"]


def get_daily_active_users(days: int = 14) -> list[dict]:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with _db_lock:
        cur = _conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, COUNT(DISTINCT user_id) AS users
            FROM chats
            WHERE created_at >= ?
            GROUP BY day
            ORDER BY day
            """,
            (cutoff,),
        )
        return [dict(row) for row in cur.fetchall()]


# ──────────────────────────────────────────────
#  Malicious-request detection
# ──────────────────────────────────────────────
MALICIOUS_KEYWORDS = [
    # prompt injection / jailbreak attempts
    "ignore previous instructions", "ignore all previous", "disregard previous",
    "disregard all previous", "you are now", "jailbreak", "dan mode",
    "developer mode", "reveal your prompt", "reveal your system prompt",
    "reveal your instructions", "print your instructions", "repeat your instructions",
    "pretend you are", "act as if", "bypass your", "override your",
    "forget your instructions", "system prompt",
    # injection / exploitation attempts
    "drop table", "select * from", "union select", "'; --", "or 1=1", "1=1--",
    "<script", "javascript:", "onerror=", "onload=", "../../../etc/passwd",
    "; rm -rf", "&& rm -rf", "rm -rf /", "wget http", "curl http",
    # credential / data exfiltration attempts
    "api key", "api_key", "give me the password", "credit card number",
    "social security number",
    # harmful-content requests
    "how to make a bomb", "how to build a bomb", "how to hack", "malware",
    "ransomware", "ddos attack",
]


def detect_malicious(text: str) -> list[str]:
    """Return the list of configured keywords found in `text` (case-insensitive)."""
    if not text:
        return []
    lowered = text.lower()
    return [kw for kw in MALICIOUS_KEYWORDS if kw in lowered]


# ──────────────────────────────────────────────
#  Server / Dizzy status
# ──────────────────────────────────────────────
_process = psutil.Process(os.getpid())
_start_time = time.time()


def get_server_status() -> dict:
    try:
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(str(Path(__file__).resolve().parent.parent))
        return {
            "status": "online",
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": vm.percent,
            "memory_used_gb": round(vm.used / (1024 ** 3), 2),
            "memory_total_gb": round(vm.total / (1024 ** 3), 2),
            "disk_percent": disk.percent,
            "process_memory_mb": round(_process.memory_info().rss / (1024 ** 2), 2),
            "uptime_seconds": round(time.time() - _start_time),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


_dizzy_status_provider: Optional[Callable[[], dict]] = None


def register_status_provider(fn: Callable[[], dict]) -> None:
    """main.py registers a callable that returns the backend/model status dict."""
    global _dizzy_status_provider
    _dizzy_status_provider = fn


def get_dizzy_status() -> dict:
    if _dizzy_status_provider is None:
        return {"state": "unknown", "message": "no status provider registered"}
    try:
        return _dizzy_status_provider()
    except Exception as e:
        return {"state": "error", "message": str(e)}


# ──────────────────────────────────────────────
#  Auth
# ──────────────────────────────────────────────
_security = HTTPBasic()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(12)
    print(
        f"[admin] ADMIN_PASSWORD not set — generated a random password for this run: "
        f"{ADMIN_PASSWORD}\n"
        f"[admin] Set ADMIN_USERNAME / ADMIN_PASSWORD in .env for a stable login."
    )


def verify_admin(credentials: HTTPBasicCredentials = Depends(_security)) -> str:
    valid_user = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    valid_pass = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (valid_user and valid_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ──────────────────────────────────────────────
#  Router
# ──────────────────────────────────────────────
router = APIRouter()

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dizzy Admin Panel</title>
<meta name="robots" content="noindex, nofollow">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: #0f1115; color: #e6e6e6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  h1 { margin: 0 0 4px; font-size: 22px; }
  .sub { color: #8a8f98; margin-bottom: 24px; font-size: 13px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 24px; }
  .card {
    background: #161a21; border: 1px solid #262b35; border-radius: 10px; padding: 16px;
  }
  .card h3 { margin: 0 0 10px; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #8a8f98; }
  .metric { font-size: 26px; font-weight: 600; }
  .metric small { font-size: 12px; color: #8a8f98; font-weight: 400; }
  .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .pill.ok { background: #143d24; color: #4ade80; }
  .pill.warn { background: #4d3a12; color: #facc15; }
  .pill.bad { background: #4d1414; color: #f87171; }
  section { margin-bottom: 28px; }
  section h2 { font-size: 15px; margin: 0 0 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid #21252e; vertical-align: top; }
  th { color: #8a8f98; font-weight: 600; position: sticky; top: 0; background: #0f1115; }
  tbody tr:hover { background: #171b22; }
  .scroll { max-height: 360px; overflow-y: auto; border: 1px solid #262b35; border-radius: 10px; }
  .tag { background: #4d1414; color: #f87171; border-radius: 6px; padding: 1px 6px; font-size: 11px; margin-right: 4px; }
  .empty { color: #5b616c; padding: 14px; font-size: 13px; }
  .bars { display: flex; align-items: flex-end; gap: 6px; height: 90px; }
  .bar-wrap { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; }
  .bar { width: 100%; background: #3b82f6; border-radius: 3px 3px 0 0; min-height: 2px; }
  .bar-label { font-size: 10px; color: #8a8f98; margin-top: 4px; white-space: nowrap; }
  .refresh-note { font-size: 11px; color: #5b616c; }
</style>
</head>
<body>
  <h1>Dizzy Admin Panel</h1>
  <div class="sub">RDM Chatbot monitoring &middot; <span class="refresh-note">auto-refreshes every 5s</span></div>

  <div class="grid" id="statusGrid"></div>

  <section>
    <h2>Users per day</h2>
    <div class="card"><div class="bars" id="userBars"></div></div>
  </section>

  <section>
    <h2>Flagged / suspicious requests</h2>
    <div class="scroll"><table>
      <thead><tr><th>Time (UTC)</th><th>User</th><th>Keywords</th><th>Message</th></tr></thead>
      <tbody id="flaggedBody"></tbody>
    </table></div>
  </section>

  <section>
    <h2>Error log</h2>
    <div class="scroll"><table>
      <thead><tr><th>Time (UTC)</th><th>Source</th><th>Message</th></tr></thead>
      <tbody id="errorBody"></tbody>
    </table></div>
  </section>

  <section>
    <h2>Chat history (all users)</h2>
    <div class="scroll"><table>
      <thead><tr><th>Time (UTC)</th><th>User</th><th>IP</th><th>Role</th><th>Message</th></tr></thead>
      <tbody id="chatBody"></tbody>
    </table></div>
  </section>

<script>
function esc(s) {
  const d = document.createElement('div');
  d.innerText = (s ?? '');
  return d.innerHTML;
}

function statusPill(state) {
  const s = (state || '').toLowerCase();
  let cls = 'warn';
  if (['online', 'ready', 'ok'].includes(s)) cls = 'ok';
  if (['error', 'down', 'offline'].includes(s)) cls = 'bad';
  return `<span class="pill ${cls}">${esc(state || 'unknown')}</span>`;
}

async function refreshStats() {
  const res = await fetch('/admin/api/stats');
  if (!res.ok) return;
  const data = await res.json();
  const s = data.server, d = data.dizzy, u = data.users;

  document.getElementById('statusGrid').innerHTML = `
    <div class="card"><h3>Server status</h3><div class="metric">${statusPill(s.status)}</div>
      <div class="sub">CPU ${s.cpu_percent ?? '–'}% &middot; RAM ${s.memory_percent ?? '–'}% &middot; uptime ${Math.round((s.uptime_seconds||0)/60)}m</div></div>
    <div class="card"><h3>Dizzy status</h3><div class="metric">${statusPill(d.state)}</div>
      <div class="sub">${esc(d.message || '')}</div></div>
    <div class="card"><h3>Active now</h3><div class="metric">${u.active_now}<small> users (5 min)</small></div></div>
    <div class="card"><h3>Today</h3><div class="metric">${(u.daily.find(x => x.day === new Date().toISOString().slice(0,10)) || {users:0}).users}<small> unique users</small></div></div>
  `;

  const bars = document.getElementById('userBars');
  if (!u.daily.length) {
    bars.innerHTML = '<div class="empty">No activity yet.</div>';
  } else {
    const max = Math.max(...u.daily.map(x => x.users), 1);
    bars.innerHTML = u.daily.map(x => `
      <div class="bar-wrap">
        <div class="bar" style="height:${Math.max((x.users / max) * 100, 3)}%"></div>
        <div class="bar-label">${x.day.slice(5)}<br>${x.users}</div>
      </div>`).join('');
  }
}

async function refreshFlagged() {
  const res = await fetch('/admin/api/flagged');
  if (!res.ok) return;
  const { flagged } = await res.json();
  const body = document.getElementById('flaggedBody');
  body.innerHTML = flagged.length ? flagged.map(f => `
    <tr>
      <td>${esc(f.created_at)}</td>
      <td>${esc(f.user_id)}</td>
      <td>${(f.matched_keywords || '').split(',').filter(Boolean).map(k => `<span class="tag">${esc(k)}</span>`).join('')}</td>
      <td>${esc(f.content)}</td>
    </tr>`).join('') : '<tr><td colspan="4" class="empty">No flagged requests.</td></tr>';
}

async function refreshErrors() {
  const res = await fetch('/admin/api/errors');
  if (!res.ok) return;
  const { errors } = await res.json();
  const body = document.getElementById('errorBody');
  body.innerHTML = errors.length ? errors.map(e => `
    <tr><td>${esc(e.created_at)}</td><td>${esc(e.source)}</td><td>${esc(e.message)}</td></tr>
  `).join('') : '<tr><td colspan="3" class="empty">No errors logged.</td></tr>';
}

async function refreshChats() {
  const res = await fetch('/admin/api/chats');
  if (!res.ok) return;
  const { chats } = await res.json();
  const body = document.getElementById('chatBody');
  body.innerHTML = chats.length ? chats.map(c => `
    <tr>
      <td>${esc(c.created_at)}</td>
      <td>${esc(c.user_id)}</td>
      <td>${esc(c.ip)}</td>
      <td>${esc(c.role)}</td>
      <td>${esc(c.content)}</td>
    </tr>`).join('') : '<tr><td colspan="5" class="empty">No chat history yet.</td></tr>';
}

function refreshAll() {
  refreshStats();
  refreshFlagged();
  refreshErrors();
  refreshChats();
}

refreshAll();
setInterval(refreshAll, 5000);
</script>
</body>
</html>
"""


@router.get("/admin", response_class=HTMLResponse)
def admin_dashboard(user: str = Depends(verify_admin)) -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_HTML)


@router.get("/admin/api/stats")
def admin_stats(user: str = Depends(verify_admin)) -> dict:
    return {
        "server": get_server_status(),
        "dizzy": get_dizzy_status(),
        "users": {
            "active_now": get_active_user_count(),
            "daily": get_daily_active_users(),
        },
    }


@router.get("/admin/api/errors")
def admin_errors(user: str = Depends(verify_admin), limit: int = 100) -> dict:
    return {"errors": get_recent_errors(limit)}


@router.get("/admin/api/chats")
def admin_chats(user: str = Depends(verify_admin), limit: int = 200) -> dict:
    return {"chats": get_recent_chats(limit)}


@router.get("/admin/api/flagged")
def admin_flagged(user: str = Depends(verify_admin), limit: int = 200) -> dict:
    return {"flagged": get_flagged_chats(limit)}

