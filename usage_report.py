#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import logging
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta, date as date_cls
from typing import Dict, Iterable, List, Optional, Tuple
import requests
from dotenv import load_dotenv
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise RuntimeError(
        "pyarrow is required to write parquet output. Please install it via 'pip install pyarrow'."
    ) from exc
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

# --------------------------- Config & Constants ---------------------------

load_dotenv()

BASE_URL = os.environ.get("CAPI_BASE_URL", "https://api.chatgpt.com/v1").rstrip("/")
API_KEY = os.environ.get("CAPI_API_KEY")
WORKSPACE_ID = os.environ.get("WORKSPACE_ID")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
# Report columns (exact order) from your sample
REPORT_COLUMNS = [
    "period_start","period_end","account_id","public_id","name","email",
    "role","user_role","department","groups","user_status","created_or_invited_date",
    "is_active","first_day_active_in_period","last_day_active_in_period",
    "messages","messages_rank","model_to_messages","gpt_messages","gpts_messaged",
    "gpt_to_messages","tool_messages","tools_messaged","tool_to_messages",
    "project_messages","projects_messaged","project_to_messages","projects_created",
    "last_day_active"
]

REPORT_SCHEMA = pa.schema([
    ("period_start", pa.string()),
    ("period_end", pa.string()),
    ("account_id", pa.string()),
    ("public_id", pa.string()),
    ("name", pa.string()),
    ("email", pa.string()),
    ("role", pa.string()),
    ("user_role", pa.string()),
    ("department", pa.string()),
    ("groups", pa.string()),
    ("user_status", pa.string()),
    ("created_or_invited_date", pa.string()),
    ("is_active", pa.bool_()),
    ("first_day_active_in_period", pa.string()),
    ("last_day_active_in_period", pa.string()),
    ("messages", pa.int64()),
    ("messages_rank", pa.int64()),
    ("model_to_messages", pa.string()),
    ("gpt_messages", pa.int64()),
    ("gpts_messaged", pa.int64()),
    ("gpt_to_messages", pa.string()),
    ("tool_messages", pa.int64()),
    ("tools_messaged", pa.int64()),
    ("tool_to_messages", pa.string()),
    ("project_messages", pa.int64()),
    ("projects_messaged", pa.int64()),
    ("project_to_messages", pa.string()),
    ("projects_created", pa.int64()),
    ("last_day_active", pa.string()),
])

GPT_REPORT_COLUMNS = [
    "cadence","period_start","period_end","account_id","gpt_id","gpt_name",
    "config_type","gpt_description","gpt_url","gpt_creator_public_id",
    "gpt_creator_email","is_active","first_day_active_in_period",
    "last_day_active_in_period","messages_workspace",
    "unique_messagers_workspace"
]

GPT_REPORT_SCHEMA = pa.schema([
    ("cadence", pa.string()),
    ("period_start", pa.string()),
    ("period_end", pa.string()),
    ("account_id", pa.string()),
    ("gpt_id", pa.string()),
    ("gpt_name", pa.string()),
    ("config_type", pa.string()),
    ("gpt_description", pa.string()),
    ("gpt_url", pa.string()),
    ("gpt_creator_public_id", pa.string()),
    ("gpt_creator_email", pa.string()),
    ("is_active", pa.bool_()),
    ("first_day_active_in_period", pa.string()),
    ("last_day_active_in_period", pa.string()),
    ("messages_workspace", pa.int64()),
    ("unique_messagers_workspace", pa.int64()),
])

USERS_SUBDIR = "users"
GPTS_SUBDIR = "gpts"

# Safety defaults
CONVERSATION_LIMIT = 500   # max allowed by API (see spec)
USERS_LIMIT        = 200   # API default (spec)
PROJECTS_LIMIT     = 200   # reasonable page size
CONVERSATION_USERS_FILTER_LIMIT = 100  # max users param size per spec

# Rate limit: 50 rpm per endpoint => ~1.2 sec between calls per endpoint
PER_ENDPOINT_MIN_INTERVAL_SEC = 60.0 / 50.0

# --------------------------- Utilities ---------------------------

class RateLimiter:
    """Simple per-endpoint leaky-bucket style limiter."""
    def __init__(self, min_interval_sec: float):
        self.min_interval_sec = min_interval_sec
        self._last_call_at: Dict[str, float] = {}

    def wait(self, endpoint_key: str):
        now = time.monotonic()
        last = self._last_call_at.get(endpoint_key)
        if last is not None:
            delta = now - last
            wait = self.min_interval_sec - delta
            if wait > 0:
                time.sleep(wait)
        self._last_call_at[endpoint_key] = time.monotonic()

rl = RateLimiter(PER_ENDPOINT_MIN_INTERVAL_SEC)
logger = logging.getLogger(__name__)

@contextmanager
def progress_bar(iterable, description: str, total: Optional[int] = None, unit: str = "item"):
    """Yield iterable wrapped with tqdm when available, otherwise passthrough."""
    if tqdm and sys.stderr.isatty():
        bar = tqdm(iterable, desc=description, total=total, unit=unit)
        try:
            yield bar
        finally:
            bar.close()
    else:
        yield iterable

def utc_bounds_for(target_date: date_cls) -> Tuple[int, int, str, str]:
    """Return (start_ts, end_ts, start_str, end_str) for a UTC calendar day."""
    start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    iso = target_date.isoformat()
    return int(start.timestamp()), int(end.timestamp()), iso, iso

def yday_utc_bounds() -> Tuple[int, int, str, str]:
    """Return (start_ts, end_ts, start_str, end_str) for yesterday in UTC."""
    today_utc = datetime.now(timezone.utc).date()
    yday = today_utc - timedelta(days=1)
    return utc_bounds_for(yday)

def http_get(endpoint_key: str, url: str, params: dict) -> dict:
    """GET with per-endpoint throttle, 429 handling (Retry-After), and modest backoff."""
    max_attempts = 6
    backoff = 1.5
    attempt = 0
    while True:
        attempt += 1
        rl.wait(endpoint_key)
        try:
            resp = requests.get(url, headers={"Authorization": f"Bearer {API_KEY}"}, params=params, timeout=60)
        except Exception as e:
            if attempt >= max_attempts:
                raise
            time.sleep(min(60, (backoff ** attempt)))
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else max(PER_ENDPOINT_MIN_INTERVAL_SEC * 2, backoff ** attempt)
            time.sleep(sleep_s)
            continue

        if 200 <= resp.status_code < 300:
            return resp.json()

        if 500 <= resp.status_code < 600 and attempt < max_attempts:
            time.sleep(min(60, (backoff ** attempt)))
            continue

        # Surface other errors
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise RuntimeError(f"GET {url} failed ({resp.status_code}): {payload}")

def chunked(iterable: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# --------------------------- API Iterators ---------------------------

def iter_users(workspace_id: str) -> Iterable[dict]:
    """Yield WorkspaceUser dicts."""
    url = f"{BASE_URL}/compliance/workspaces/{workspace_id}/users"
    after = None
    while True:
        params = {"limit": USERS_LIMIT}
        if after:
            params["after"] = after
        data = http_get("users_list", url, params)
        users = data.get("data", [])
        for u in users:
            yield u
        if not data.get("has_more"):
            break
        after = data.get("last_id")

def iter_projects(workspace_id: str) -> Iterable[dict]:
    """Yield Project dicts (live projects; drafts are excluded by API)."""
    url = f"{BASE_URL}/compliance/workspaces/{workspace_id}/projects"
    after = None
    while True:
        params = {"limit": PROJECTS_LIMIT}
        if after:
            params["after"] = after
        data = http_get("projects_list", url, params)
        for p in data.get("data", []):
            yield p
        if not data.get("has_more"):
            break
        after = data.get("last_id")

def iter_gpts(workspace_id: str) -> Iterable[dict]:
    """Yield GPT dicts (live GPTs)."""
    url = f"{BASE_URL}/compliance/workspaces/{workspace_id}/gpts"
    after = None
    while True:
        params = {"limit": 200}
        if after:
            params["after"] = after
        data = http_get("gpts_list", url, params)
        for g in data.get("data", []):
            yield g
        if not data.get("has_more"):
            break
        after = data.get("last_id")

def iter_conversations_yday(workspace_id: str, start_ts: int, user_ids: Optional[List[str]] = None) -> Iterable[dict]:
    """
    Yield conversations updated since start_ts (yesterday 00:00:00 UTC).
    Uses 'since_timestamp' for the first page, then paginates via 'after'.
    """
    url = f"{BASE_URL}/compliance/workspaces/{workspace_id}/conversations"
    # First page with since_timestamp
    params = {
        "since_timestamp": start_ts,
        "limit": CONVERSATION_LIMIT,
        "file_format": "id"  # smaller payloads
    }
    if user_ids:
        params["users"] = user_ids
    data = http_get("conversations_list", url, params)
    for c in data.get("data", []):
        yield c

    after = data.get("last_id")
    while data.get("has_more"):
        # Per spec, since_timestamp and after cannot be sent together.
        # We continue pagination with 'after' only; downstream filters still keep us on yesterday's messages.
        params = {
            "after": after,
            "limit": CONVERSATION_LIMIT,
            "file_format": "id"
        }
        if user_ids:
            params["users"] = user_ids
        data = http_get("conversations_list", url, params)
        for c in data.get("data", []):
            yield c
        after = data.get("last_id")
        if not data.get("has_more"):
            break

# --------------------------- Aggregation ---------------------------

def build_daily_report(target_date: Optional[date_cls] = None):
    if not API_KEY or not WORKSPACE_ID:
        print("Please set CAPI_API_KEY and WORKSPACE_ID environment variables.", file=sys.stderr)
        sys.exit(2)

    if target_date is None:
        start_ts, end_ts, start_str, end_str = yday_utc_bounds()
        report_label = "yesterday"
    else:
        start_ts, end_ts, start_str, end_str = utc_bounds_for(target_date)
        report_label = target_date.isoformat()
    logger.info("Building daily report for %s (UTC %s to %s)", report_label, start_str, end_str)

    # Map user_id -> user record from Users API
    users_index: Dict[str, dict] = {}
    logger.info("Fetching workspace users...")
    with progress_bar(iter_users(WORKSPACE_ID), "Users", unit="user") as user_iter:
        for u in user_iter:
            # fields per WorkspaceUser schema
            users_index[u["id"]] = {
                "id": u.get("id"),
                "email": u.get("email"),
                "name": u.get("name"),
                "created_at": u.get("created_at"),
                "role": u.get("role"),
                "status": u.get("status"),
            }
    total_users = len(users_index)
    active_user_ids = [uid for uid, info in users_index.items() if info.get("status") == "active"]
    logger.info("Discovered %d users (%d active)", total_users, len(active_user_ids))

    # Map gpt_id -> GPT metadata
    gpt_index: Dict[str, dict] = {}
    logger.info("Fetching workspace GPTs...")
    with progress_bar(iter_gpts(WORKSPACE_ID), "GPTs", unit="gpt") as gpt_iter:
        for g in gpt_iter:
            gpt_id = g.get("id")
            if not gpt_id:
                continue
            latest_entry = {}
            latest_config = g.get("latest_config") or {}
            if isinstance(latest_config, dict):
                configs = latest_config.get("data") or []
                if configs:
                    latest_entry = configs[0]
            gpt_index[gpt_id] = {
                "id": gpt_id,
                "name": latest_entry.get("name") or g.get("name") or "",
                "description": latest_entry.get("description") or g.get("description") or "",
                "url": "",
                "owner_id": g.get("owner_id") or "",
                "owner_email": g.get("owner_email") or "",
                "config_type": "Live",
            }

    # Initialize counters (per user)
    counters = {}
    def ensure_user(uid: str):
        if uid not in counters:
            info = users_index.get(uid, {})
            counters[uid] = {
                "messages": 0,
                "gpt_messages": 0,
                "gpts_set": set(),
                "tool_messages": 0,
                "tools_set": set(),
                "project_messages": 0,
                "projects_set": set(),
                "projects_created": 0,
                "active": False,
                "user": info,
                "gpt_messages_map": {},
                "tool_messages_map": {},
                "project_messages_map": {},
                "model_messages_map": {},
                "first_active_ts": None,
                "last_active_ts": None,
            }

    gpt_counters: Dict[str, dict] = {}
    def ensure_gpt(gid: str):
        if gid not in gpt_counters:
            info = gpt_index.get(gid, {
                "id": gid,
                "name": "",
                "description": "",
                "url": "",
                "owner_id": "",
                "owner_email": "",
                "config_type": "",
            })
            gpt_counters[gid] = {
                "info": info,
                "messages": 0,
                "unique_users": set(),
                "first_active_ts": None,
                "last_active_ts": None,
            }

    for gid in gpt_index:
        ensure_gpt(gid)

    for uid in active_user_ids:
        ensure_user(uid)

    # Stream conversations updated since start_ts and count only messages in [start_ts, end_ts)
    processed_message_ids = set()
    conversation_count = 0
    message_count = 0
    total_chunks = (len(active_user_ids) + CONVERSATION_USERS_FILTER_LIMIT - 1) // CONVERSATION_USERS_FILTER_LIMIT if active_user_ids else 0
    if total_chunks == 0:
        logger.info("No active users to fetch conversations for.")
    else:
        logger.info("Fetching conversations for active users in %d chunk(s)...", total_chunks)
    with progress_bar(chunked(active_user_ids, CONVERSATION_USERS_FILTER_LIMIT), "Conversation Chunks", total=total_chunks if total_chunks else None, unit="chunk") as chunk_iter:
        for user_chunk in chunk_iter:
            if not user_chunk:
                continue
            for conv in iter_conversations_yday(WORKSPACE_ID, start_ts, user_ids=user_chunk):
                conversation_count += 1
                conv_user_id = conv.get("user_id")
                if not conv_user_id:
                    continue
                user_info = users_index.get(conv_user_id)
                if not user_info or user_info.get("status") != "active":
                    continue
                ensure_user(conv_user_id)

                messages = conv.get("messages") or []
                if isinstance(messages, dict):
                    # Compliance API MessageList schema: messages.data holds the actual entries
                    message_iterable = messages.get("data") or []
                else:
                    message_iterable = messages

                for msg in message_iterable:
                    if not isinstance(msg, dict):
                        logger.debug("Skipping non-dict message payload for conversation %s: %r", conv.get("id"), msg)
                        continue
                    msg_id = msg.get("id")
                    if msg_id:
                        dedupe_key = (conv.get("id"), msg_id)
                        if dedupe_key in processed_message_ids:
                            continue
                        processed_message_ids.add(dedupe_key)
                    ts = msg.get("created_at")
                    if ts is None:
                        continue
                    try:
                        ts_val = float(ts)
                    except (TypeError, ValueError):
                        logger.debug("Skipping message with non-numeric timestamp for conversation %s: %r", conv.get("id"), ts)
                        continue
                    if not (start_ts <= ts_val < end_ts):
                        continue

                    author = (msg.get("author") or {})
                    role = author.get("role")
                    tool_name = author.get("tool_name")
                    content = msg.get("content")
                    if content is None:
                        continue
                    message_count += 1

                    user_counter = counters[conv_user_id]

                    # Count user messages
                    if role == "user":
                        user_counter["messages"] += 1
                        user_counter["active"] = True
                        if user_counter["first_active_ts"] is None or ts_val < user_counter["first_active_ts"]:
                            user_counter["first_active_ts"] = ts_val
                        if user_counter["last_active_ts"] is None or ts_val > user_counter["last_active_ts"]:
                            user_counter["last_active_ts"] = ts_val

                        gpt_id = msg.get("gpt_id")
                        if gpt_id:
                            user_counter["gpt_messages"] += 1
                            user_counter["gpts_set"].add(gpt_id)
                            gpt_map = user_counter["gpt_messages_map"]
                            gpt_map[gpt_id] = gpt_map.get(gpt_id, 0) + 1
                            ensure_gpt(gpt_id)
                            gpt_counter = gpt_counters[gpt_id]
                            gpt_counter["messages"] += 1
                            gpt_counter["unique_users"].add(conv_user_id)
                            if gpt_counter["first_active_ts"] is None or ts_val < gpt_counter["first_active_ts"]:
                                gpt_counter["first_active_ts"] = ts_val
                            if gpt_counter["last_active_ts"] is None or ts_val > gpt_counter["last_active_ts"]:
                                gpt_counter["last_active_ts"] = ts_val

                        project_id = msg.get("project_id")
                        if project_id:
                            user_counter["project_messages"] += 1
                            user_counter["projects_set"].add(project_id)
                            proj_map = user_counter["project_messages_map"]
                            proj_map[project_id] = proj_map.get(project_id, 0) + 1

                    # Count assistant/non-GPT/non-project "model" replies for model_to_messages
                    if role == "assistant":
                        model_name = msg.get("model") or msg.get("model_name")
                        if model_name:
                            model_map = user_counter["model_messages_map"]
                            model_map[model_name] = model_map.get(model_name, 0) + 1

                    # Count assistant/non-GPT/non-project "model" replies for model_to_messages
                    # Tool messages & distinct tool names
                    if role == "tool":
                        user_counter["tool_messages"] += 1
                        if tool_name:
                            user_counter["tools_set"].add(tool_name)
                            tool_map = user_counter["tool_messages_map"]
                            tool_map[tool_name] = tool_map.get(tool_name, 0) + 1
    logger.info("Processed %d conversations and %d messages for active users", conversation_count, message_count)

    # Projects created yesterday
    logger.info("Scanning projects for creations during window...")
    project_created_count = 0
    with progress_bar(iter_projects(WORKSPACE_ID), "Projects", unit="project") as project_iter:
        for proj in project_iter:
            created_at = proj.get("created_at")
            owner_id = proj.get("owner_id")
            if created_at is None or owner_id is None:
                continue
            try:
                created_at_val = float(created_at)
            except (TypeError, ValueError):
                logger.debug("Skipping project with non-numeric created_at: %r", created_at)
                continue
            if start_ts <= created_at_val < end_ts:
                ensure_user(owner_id)
                counters[owner_id]["projects_created"] = counters[owner_id].get("projects_created", 0) + 1
                project_created_count += 1
    logger.info("Found %d project(s) created during the window", project_created_count)

    # Finalize rows for ALL known users (so inactive users appear with zeros)
    rows = []
    for uid, uinfo in users_index.items():
        c = counters.get(uid)
        if not c:
            # Initialize empty so the row exists
            c = {
                "messages": 0,
                "gpt_messages": 0,
                "gpts_set": set(),
                "tool_messages": 0,
                "tools_set": set(),
                "project_messages": 0,
                "projects_set": set(),
                "projects_created": 0,
                "active": False,
                "user": uinfo,
                "gpt_messages_map": {},
                "tool_messages_map": {},
                "project_messages_map": {},
                "model_messages_map": {},
                "first_active_ts": None,
                "last_active_ts": None,
            }

        messages = int(c.get("messages", 0))
        gpt_messages = int(c.get("gpt_messages", 0))
        tool_messages = int(c.get("tool_messages", 0))
        project_messages = int(c.get("project_messages", 0))
        projects_created = int(c.get("projects_created", 0))
        gpts_messaged = len(c.get("gpts_set", set()))
        tools_messaged = len(c.get("tools_set", set()))
        projects_messaged = len(c.get("projects_set", set()))

        first_active_ts = c.get("first_active_ts")
        last_active_ts = c.get("last_active_ts")
        first_active_date = (
            datetime.fromtimestamp(first_active_ts, tz=timezone.utc).date().isoformat()
            if first_active_ts is not None else ""
        )
        last_active_date = (
            datetime.fromtimestamp(last_active_ts, tz=timezone.utc).date().isoformat()
            if last_active_ts is not None else ""
        )

        model_map = c.get("model_messages_map", {}) or {}
        gpt_map = c.get("gpt_messages_map", {}) or {}
        tool_map = c.get("tool_messages_map", {}) or {}
        project_map = c.get("project_messages_map", {}) or {}

        model_to_messages_json = json.dumps(model_map, sort_keys=True)
        gpt_to_messages_json = json.dumps(gpt_map, sort_keys=True)
        tool_to_messages_json = json.dumps(tool_map, sort_keys=True)
        project_to_messages_json = json.dumps(project_map, sort_keys=True)

        is_active = messages > 0

        source_role = uinfo.get("role", "")
        user_role_value = {
            "account-owner": "owner",
            "account-admin": "admin",
            "standard-user": "user",
        }.get(source_role, "")
        status_raw = (uinfo.get("status") or "").lower()
        status_value = {
            "active": "enabled",
            "inactive": "deleted",
            "pending": "pending",
        }.get(status_raw, status_raw)
        created_at_ts = uinfo.get("created_at")
        created_or_invited_date = (
            datetime.fromtimestamp(created_at_ts, tz=timezone.utc).date().isoformat()
            if created_at_ts and status_value in {"enabled", "deleted"}
            else ""
        )
        groups_payload = json.dumps(uinfo.get("groups") or {}, sort_keys=True)
        department_value = uinfo.get("department") or ""

        row = {
            "period_start": start_str,
            "period_end": end_str,
            "account_id": WORKSPACE_ID,
            "public_id": uinfo.get("id", ""),
            "name": uinfo.get("name", ""),
            "email": uinfo.get("email", ""),
            "role": source_role,
            "user_role": user_role_value,
            "department": department_value,
            "groups": groups_payload,
            "user_status": status_value,
            "created_or_invited_date": created_or_invited_date,
            "is_active": is_active,
            "first_day_active_in_period": first_active_date,
            "last_day_active_in_period": last_active_date,
            "messages": messages,
            "messages_rank": None,  # filled later
            "model_to_messages": model_to_messages_json,
            "gpt_messages": gpt_messages,
            "gpts_messaged": gpts_messaged,
            "gpt_to_messages": gpt_to_messages_json,
            "tool_messages": tool_messages,
            "tools_messaged": tools_messaged,
            "tool_to_messages": tool_to_messages_json,
            "project_messages": project_messages,
            "projects_messaged": projects_messaged,
            "project_to_messages": project_to_messages_json,
            "projects_created": projects_created,
            "last_day_active": last_active_date,
        }
        rows.append(row)

    # Rank by messages (dense rank, descending). Users with 0 messages get no rank.
    rows_sorted = sorted(rows, key=lambda r: r["messages"], reverse=True)
    rank = 0
    last_val = None
    for r in rows_sorted:
        if r["messages"] <= 0:
            r["messages_rank"] = None
            continue
        if r["messages"] != last_val:
            rank += 1
            last_val = r["messages"]
        r["messages_rank"] = rank

    # Restore original order (by email asc for determinism)
    rows = sorted(rows_sorted, key=lambda r: (r["email"] or "", r["public_id"] or ""))
    row_count = len(rows)
    logger.info("Final dataset contains %d user rows", row_count)

    # Write Parquet in partitioned directory structure
    ordered_rows = [{col: r.get(col) for col in REPORT_COLUMNS} for r in rows]
    column_arrays = []
    for field in REPORT_SCHEMA:
        values = [row[field.name] for row in ordered_rows]
        array = pa.array(values, type=field.type, from_pandas=True)
        column_arrays.append(array)
    table = pa.Table.from_arrays(column_arrays, schema=REPORT_SCHEMA)

    user_partition_dir = os.path.join(
        OUTPUT_DIR,
        USERS_SUBDIR,
        f"date={start_str}",
    )
    os.makedirs(user_partition_dir, exist_ok=True)
    user_out_path = os.path.join(user_partition_dir, "part-00000.parquet")
    logger.info("Writing user Parquet dataset to %s", user_out_path)
    pq.write_table(table, user_out_path, use_dictionary=False)

    # Build GPT usage rows
    gpt_rows = []
    for gid, counter in gpt_counters.items():
        info = counter.get("info") or {}
        first_active_ts = counter.get("first_active_ts")
        last_active_ts = counter.get("last_active_ts")
        first_active_date = (
            datetime.fromtimestamp(first_active_ts, tz=timezone.utc).date().isoformat()
            if first_active_ts is not None else ""
        )
        last_active_date = (
            datetime.fromtimestamp(last_active_ts, tz=timezone.utc).date().isoformat()
            if last_active_ts is not None else ""
        )
        messages_workspace = int(counter.get("messages", 0))
        unique_messagers_workspace = len(counter.get("unique_users", set()))
        is_active_gpt = messages_workspace > 0
        row = {
            "cadence": "Daily",
            "period_start": start_str,
            "period_end": end_str,
            "account_id": WORKSPACE_ID,
            "gpt_id": info.get("id", gid),
            "gpt_name": info.get("name", ""),
            "config_type": info.get("config_type", ""),
            "gpt_description": info.get("description", ""),
            "gpt_url": info.get("url", ""),
            "gpt_creator_public_id": info.get("owner_id", ""),
            "gpt_creator_email": info.get("owner_email", ""),
            "is_active": is_active_gpt,
            "first_day_active_in_period": first_active_date,
            "last_day_active_in_period": last_active_date,
            "messages_workspace": messages_workspace,
            "unique_messagers_workspace": unique_messagers_workspace,
        }
        gpt_rows.append(row)

    # Include GPTs with metadata but no counter (should already exist, but safeguard)
    for gid, info in gpt_index.items():
        if gid not in gpt_counters:
            row = {
                "cadence": "Daily",
                "period_start": start_str,
                "period_end": end_str,
                "account_id": WORKSPACE_ID,
                "gpt_id": info.get("id", gid),
                "gpt_name": info.get("name", ""),
                "config_type": info.get("config_type", ""),
                "gpt_description": info.get("description", ""),
                "gpt_url": info.get("url", ""),
                "gpt_creator_public_id": info.get("owner_id", ""),
                "gpt_creator_email": info.get("owner_email", ""),
                "is_active": False,
                "first_day_active_in_period": "",
                "last_day_active_in_period": "",
                "messages_workspace": 0,
                "unique_messagers_workspace": 0,
            }
            gpt_rows.append(row)

    # Deterministic ordering by gpt_id
    gpt_rows = sorted(gpt_rows, key=lambda r: r["gpt_id"])
    logger.info("GPT dataset contains %d rows", len(gpt_rows))

    if gpt_rows:
        ordered_gpt_rows = [{col: r.get(col) for col in GPT_REPORT_COLUMNS} for r in gpt_rows]
        gpt_arrays = []
        for field in GPT_REPORT_SCHEMA:
            values = [row[field.name] for row in ordered_gpt_rows]
            array = pa.array(values, type=field.type, from_pandas=True)
            gpt_arrays.append(array)
        gpt_table = pa.Table.from_arrays(gpt_arrays, schema=GPT_REPORT_SCHEMA)
    else:
        gpt_table = pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in GPT_REPORT_SCHEMA],
            schema=GPT_REPORT_SCHEMA,
        )

    gpt_partition_dir = os.path.join(
        OUTPUT_DIR,
        GPTS_SUBDIR,
        f"date={start_str}",
    )
    os.makedirs(gpt_partition_dir, exist_ok=True)
    gpt_out_path = os.path.join(gpt_partition_dir, "part-00000.parquet")
    logger.info("Writing GPT Parquet dataset to %s", gpt_out_path)
    pq.write_table(gpt_table, gpt_out_path, use_dictionary=False)

    print(user_out_path)
    print(gpt_out_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily compliance usage report Parquet dataset.")
    parser.add_argument(
        "--date",
        help="UTC date to report in YYYY-MM-DD format; defaults to yesterday if omitted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not API_KEY or not WORKSPACE_ID:
        print("ERROR: CAPI_API_KEY and WORKSPACE_ID must be set.", file=sys.stderr)
        sys.exit(2)
    # Base URL must be api.chatgpt.com per Compliance API docs
    if not BASE_URL.startswith("https://api.chatgpt.com/"):
        print("WARNING: BASE_URL should be https://api.chatgpt.com/v1 per Compliance API docs.", file=sys.stderr)
    args = parse_args()
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("ERROR: --date must be in YYYY-MM-DD format (e.g. 2025-10-21).", file=sys.stderr)
            sys.exit(2)
    build_daily_report(target_date=target_date)
