"""Local mock authentication; opaque tokens, estate-scoped SQL and no real resident PII."""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from .domain import POCError, canonical, estate, utcnow


def token_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()


@dataclass(frozen=True)
class Identity:
    estate_id: str
    actor_ref: str
    role: str
    session_hash: str


class AuthStore:
    def __init__(self, path: Path, clock=utcnow):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS accounts (
              estate_id TEXT NOT NULL, username TEXT NOT NULL, actor_ref TEXT NOT NULL, role TEXT NOT NULL,
              salt TEXT NOT NULL, password_hash TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
              failures INTEGER NOT NULL DEFAULT 0, locked_until TEXT, PRIMARY KEY(estate_id,username));
            CREATE TABLE IF NOT EXISTS sessions (
              estate_id TEXT NOT NULL, token_hash TEXT NOT NULL, username TEXT NOT NULL,
              expires_at TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(estate_id,token_hash));
            CREATE TABLE IF NOT EXISTS confirmations (
              estate_id TEXT NOT NULL, ref_hash TEXT NOT NULL, actor_ref TEXT NOT NULL, session_hash TEXT NOT NULL,
              payload_json TEXT NOT NULL, expires_at TEXT NOT NULL, state TEXT NOT NULL,
              PRIMARY KEY(estate_id,ref_hash));
            CREATE TABLE IF NOT EXISTS audit (
              id INTEGER PRIMARY KEY, estate_id TEXT NOT NULL, at TEXT NOT NULL, actor_ref TEXT,
              action TEXT NOT NULL, outcome TEXT NOT NULL, request_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS business_events (
              estate_id TEXT NOT NULL, event_id TEXT NOT NULL, work_order_id TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY(estate_id,event_id));
            CREATE TABLE IF NOT EXISTS review_grants (
              estate_id TEXT NOT NULL, actor_ref TEXT NOT NULL, authorization_ref TEXT NOT NULL,
              active INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(estate_id,actor_ref,authorization_ref));
            CREATE TABLE IF NOT EXISTS knowledge_answer_audit (
              estate_id TEXT NOT NULL, request_id TEXT NOT NULL, at TEXT NOT NULL,
              actor_ref TEXT NOT NULL, result_json TEXT NOT NULL, PRIMARY KEY(estate_id,request_id));
            """)
        os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def seed_account(self, estate_id, username, actor_ref, role, password):
        estate(estate_id)
        if not actor_ref.startswith(estate_id + ":") or role not in {"RESIDENT", "BACKOFFICE_OPERATOR"}:
            raise ValueError("非法模拟账号")
        salt = secrets.token_hex(16)
        with self.connect() as db:
            existing = db.execute(
                "SELECT * FROM accounts WHERE estate_id=? AND username=?", (estate_id, username)
            ).fetchone()
            if existing and (
                existing["actor_ref"] != actor_ref
                or existing["role"] != role
                or not hmac.compare_digest(
                    password_hash(password, existing["salt"]), existing["password_hash"]
                )
            ):
                raise ValueError("模拟账号与已有凭据不一致，停止初始化；请保留原auth数据库和凭据文件")
            db.execute(
                "INSERT OR IGNORE INTO accounts(estate_id,username,actor_ref,role,salt,password_hash) "
                "VALUES(?,?,?,?,?,?)",
                (estate_id, username, actor_ref, role, salt, password_hash(password, salt)),
            )

    def audit(self, scope, actor, action, outcome, request_id):
        estate(scope)
        with self.connect() as db:
            db.execute(
                "INSERT INTO audit(estate_id,at,actor_ref,action,outcome,request_id) VALUES(?,?,?,?,?,?)",
                (scope, self.clock().isoformat(), actor, action, outcome, request_id),
            )

    def record_answer(self, scope, actor_ref, result):
        estate(scope)
        if result.get("estate_id") != scope:
            raise ValueError("答复审计不能跨楼盘")
        request_id = secrets.token_hex(12)
        result["debug_info"]["audit_id"] = request_id
        with self.connect() as db:
            db.execute("INSERT INTO knowledge_answer_audit VALUES(?,?,?,?,?)",
                       (scope, request_id, self.clock().isoformat(), actor_ref, canonical(result)))

    def login(self, estate_id, username, password):
        estate(estate_id)
        if (
            not isinstance(username, str)
            or not isinstance(password, str)
            or not (1 <= len(username) <= 64 and 1 <= len(password) <= 256)
        ):
            raise POCError("UNAUTHENTICATED", "登录验证失败", 401)
        now = self.clock()
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM accounts WHERE estate_id=? AND username=?", (estate_id, username)
            ).fetchone()
            salt = row["salt"] if row else "00" * 16
            valid = hmac.compare_digest(
                password_hash(password, salt), row["password_hash"] if row else "0" * 128
            )
            locked = bool(row and row["locked_until"] and row["locked_until"] > now.isoformat())
            if not row or not valid or not row["active"] or locked:
                if row:
                    failures = row["failures"] + 1
                    db.execute(
                        "UPDATE accounts SET failures=?,locked_until=? WHERE estate_id=? AND username=?",
                        (
                            failures,
                            (now + timedelta(seconds=30)).isoformat() if failures >= 5 else None,
                            estate_id,
                            username,
                        ),
                    )
                # Commit lockout/audit before returning the same external failure for unknown/wrong credentials.
                denied = True
            else:
                denied = False
                db.execute(
                    "UPDATE accounts SET failures=0,locked_until=NULL WHERE estate_id=? AND username=?",
                    (estate_id, username),
                )
                token = estate_id + "." + secrets.token_urlsafe(32)
                expires = (now + timedelta(hours=2)).isoformat()
                db.execute(
                    "INSERT INTO sessions(estate_id,token_hash,username,expires_at) VALUES(?,?,?,?)",
                    (estate_id, token_hash(token), username, expires),
                )
        self.audit(
            estate_id,
            row["actor_ref"] if row else None,
            "login",
            "DENIED" if denied else "OK",
            secrets.token_hex(12),
        )
        if denied:
            raise POCError("UNAUTHENTICATED", "登录验证失败", 401)
        return {"token": token, "estate_id": estate_id, "role": row["role"], "expires_at": expires}

    def resolve(self, token):
        scope = token.split(".", 1)[0] if isinstance(token, str) else ""
        if scope not in {"E001", "E002"}:
            raise POCError("UNAUTHENTICATED", "请重新登录", 401)
        with self.connect() as db:
            row = db.execute(
                """SELECT s.expires_at,s.revoked,a.* FROM sessions s JOIN accounts a
                ON a.estate_id=s.estate_id AND a.username=s.username
                WHERE s.estate_id=? AND s.token_hash=?""",
                (scope, token_hash(token)),
            ).fetchone()
        if not row or row["revoked"] or not row["active"] or row["expires_at"] <= self.clock().isoformat():
            raise POCError("UNAUTHENTICATED", "请重新登录", 401)
        return Identity(scope, row["actor_ref"], row["role"], token_hash(token))

    def logout(self, token):
        who = self.resolve(token)
        with self.connect() as db:
            db.execute(
                "UPDATE sessions SET revoked=1 WHERE estate_id=? AND token_hash=?",
                (who.estate_id, who.session_hash),
            )

    def prepare(self, who, payload):
        ref = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute(
                "INSERT INTO confirmations VALUES(?,?,?,?,?,?,?)",
                (
                    who.estate_id,
                    token_hash(ref),
                    who.actor_ref,
                    who.session_hash,
                    canonical(payload),
                    (self.clock() + timedelta(minutes=10)).isoformat(),
                    "AWAITING_CONFIRMATION",
                ),
            )
        return ref

    def confirmation(self, who, ref, required_state=None):
        if not isinstance(ref, str) or not 24 <= len(ref) <= 160:
            raise POCError("CONFIRMATION_REQUIRED", "请先核对并确认提交内容", 409)
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM confirmations WHERE estate_id=? AND ref_hash=? AND actor_ref=? "
                "AND session_hash=?",
                (who.estate_id, token_hash(ref), who.actor_ref, who.session_hash),
            ).fetchone()
        if not row:
            raise POCError("CONFIRMATION_REQUIRED", "请先核对并确认提交内容", 409)
        if row["expires_at"] <= self.clock().isoformat():
            raise POCError("CONFIRMATION_EXPIRED", "确认已过期，请重新核对", 410)
        if required_state and row["state"] != required_state:
            raise POCError("CONFIRMATION_REQUIRED", "请先核对并确认提交内容", 409)
        return json.loads(row["payload_json"])

    def confirm(self, who, ref, accept):
        self.confirmation(who, ref)
        if type(accept) is not bool:
            raise POCError("INVALID_INPUT", "必须明确确认或取消", 400)
        with self.connect() as db:
            db.execute(
                "UPDATE confirmations SET state=? WHERE estate_id=? AND ref_hash=? AND actor_ref=? "
                "AND session_hash=? AND state='AWAITING_CONFIRMATION'",
                (
                    "CONFIRMED" if accept else "CANCELLED",
                    who.estate_id,
                    token_hash(ref),
                    who.actor_ref,
                    who.session_hash,
                ),
            )
        self.confirmation(who, ref, "CONFIRMED" if accept else "CANCELLED")
        return {"confirmed": accept}

    def mirror_events(self, scope, work_order_id, events):
        with self.connect() as db:
            for event in events:
                if event["estate_id"] != estate(scope):
                    raise ValueError("日志跨楼盘")
                db.execute(
                    "INSERT OR IGNORE INTO business_events VALUES(?,?,?,?)",
                    (scope, event["event_id"], work_order_id, canonical(event)),
                )

    def reviewer_authorized(self, scope, actor_ref, authorization_ref):
        with self.connect() as db:
            return (
                db.execute(
                    "SELECT 1 FROM review_grants WHERE estate_id=? AND actor_ref=? AND authorization_ref=? "
                    "AND active=1",
                    (estate(scope), actor_ref, authorization_ref),
                ).fetchone()
                is not None
            )
