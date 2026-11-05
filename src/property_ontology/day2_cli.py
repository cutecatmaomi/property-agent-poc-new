"""Local administrator / simulated user CLI. Secrets stay in 0600 runtime files or getpass."""

import argparse
import getpass
import json
import os
import sys
import uuid
from pathlib import Path

from .actions import PropertyActions
from .domain import CORE_IDS, POCError
from .faq_importer import import_faq_candidates
from .importer import import_knowledge, load_packet, seed_mock
from .response_policy import classify_message
from .runtime import RuntimeSettings


def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):
        os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2))


def knowledge_text(result):
    """Public display is a single final answer. Full diagnostics remain available as JSON."""
    if "answer_text" in result:
        return (result['quote_notice'] + '\n\n' if result.get('quote_notice') else '') + result["answer_text"]
    if result.get("response_type") in {"FIXED_REPLY", "REVIEW_REQUIRED"}:
        return result["message"]
    raise ValueError("Knowledge display requires a unified AnswerResult")


def main():
    p = argparse.ArgumentParser(description="Day 2 Neo4j/Mock身份/受控业务Actions；没有LLM或聊天界面")
    p.add_argument("--profile", choices=["main", "test"], default="main")
    p.add_argument("--session-file", type=Path)
    p.add_argument("--preview-file", type=Path)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["init-db", "import-knowledge", "import-faq", "seed-mock", "logout", "create", "confirm", "verify-packet"]:
        sub.add_parser(name)
    stats = sub.add_parser("stats")
    stats.add_argument("--estate", required=True, choices=["E001", "E002"])
    login = sub.add_parser("login")
    login.add_argument("--estate", required=True, choices=["E001", "E002"])
    login.add_argument("--username", required=True)
    draft = sub.add_parser("prepare")
    draft.add_argument("--unit", required=True)
    draft.add_argument("--description", required=True)
    draft.add_argument("--key", default=None)
    human = sub.add_parser("handoff", help="仅回复人工服务热线话术，不登记或接通人工")
    human.add_argument("--question", default="转人工")
    status = sub.add_parser("status")
    status.add_argument("--number", required=True)
    update = sub.add_parser("update")
    update.add_argument("--number", required=True)
    update.add_argument("--target", required=True)
    update.add_argument("--expected-version", type=int, required=True)
    update.add_argument("--reason", required=True)
    update.add_argument("--key", required=True)
    consult = sub.add_parser("consult")
    consult.add_argument("--question", required=True)
    consult.add_argument("--service-code")
    consult.add_argument("--unit")
    consult.add_argument("--mode", choices=["current", "demo"], default="demo")
    consult.add_argument("--format", choices=["text", "json"], default="text")
    consult.add_argument("--renovation-activity", choices=["BARE", "MINOR", "DEMOLITION", "FLOOR_HEATING"])
    args = p.parse_args()
    app = None
    try:
        settings = RuntimeSettings.load(profile=args.profile)
        if args.command == "verify-packet":
            packet = load_packet(settings.root)
            result = {"validated_policies": len(packet["policies"]), "approved": 0}
        else:
            app = PropertyActions(settings)
            session_path = args.session_file or settings.auth_path.parent / "client-session.json"
            preview_path = args.preview_file or settings.auth_path.parent / "client-preview.json"
            if args.command == "init-db":
                app._repo.initialize_schema()
                result = {"schema_initialized": True, "core_entities": len(CORE_IDS)}
            elif args.command == "import-knowledge":
                result = import_knowledge(app._repo, settings.root)
            elif args.command == "import-faq":
                result = import_faq_candidates(app._repo, settings.root)
            elif args.command == "seed-mock":
                result = seed_mock(app._repo, app.auth, settings.root)
            elif args.command == "stats":
                result = app._repo.stats(args.estate)
            elif args.command == "login":
                # Default client session switching revokes the old context, including its confirmation tickets.
                if session_path.exists():
                    try:
                        app.logout(json.loads(session_path.read_text())["token"])
                    except POCError:
                        pass
                result = app.login(args.estate, args.username, getpass.getpass("模拟账号密码："))
                private_json(session_path, result)
                result = {k: v for k, v in result.items() if k != "token"}
            else:
                if not session_path.exists():
                    raise POCError("UNAUTHENTICATED", "请先运行login", 401)
                token = json.loads(session_path.read_text())["token"]
                if args.command == "logout":
                    result = app.logout(token)
                    session_path.unlink()
                elif args.command == "prepare":
                    request = {
                        "unit_id": args.unit,
                        "description": args.description,
                        "idempotency_key": args.key or uuid.uuid4().hex,
                        "service_code": "maintenance",
                    }
                    result = app.prepare_work_order(token, request)
                    if result.get("response_type") == "FIXED_REPLY":
                        # Do not leave a stale draft usable after a blocked prepare command.
                        preview_path.unlink(missing_ok=True)
                    else:
                        private_json(
                            preview_path, {"request": request, "confirmation_ref": result["confirmation_ref"]}
                        )
                        result = {k: v for k, v in result.items() if k != "confirmation_ref"}
                elif args.command == "handoff":
                    # Compatibility shortcut: this command only returns the fixed phone reply.
                    result = app.query_knowledge(
                        token, args.question if classify_message(args.question) else "转人工"
                    )
                elif args.command in {"confirm", "create"}:
                    if not preview_path.exists():
                        raise POCError("CONFIRMATION_REQUIRED", "请先运行prepare核对报修内容", 409)
                    saved = json.loads(preview_path.read_text())
                    request_type = saved.get("request_type", "REPAIR")
                    if request_type == "HUMAN_HANDOFF":
                        result = app.query_knowledge(token, "转人工")
                    elif request_type != "REPAIR":
                        raise POCError("INVALID_INPUT", "请求类型无效，请重新核对", 400)
                    elif args.command == "confirm":
                        print(json.dumps(saved["request"], ensure_ascii=False, indent=2))
                        result = app.confirm_work_order(
                            token,
                            saved["confirmation_ref"],
                            accept=input("确认上述内容？输入“确认”提交确认，其余输入取消：").strip()
                            == "确认",
                        )
                    else:
                        result = app.create_work_order(
                            token, {**saved["request"], "confirmation_ref": saved["confirmation_ref"]}
                        )
                elif args.command == "status":
                    result = app.get_work_order_status(token, args.number)
                elif args.command == "update":
                    result = app.update_work_order_status(
                        token,
                        {
                            "number": args.number,
                            "target_status": args.target,
                            "expected_version": args.expected_version,
                            "reason": args.reason,
                            "idempotency_key": args.key,
                        },
                    )
                else:
                    result = app.answer_question(
                        token,
                        args.question,
                        service_code=args.service_code,
                        unit_id=args.unit,
                        knowledge_mode=args.mode,
                        renovation_activity=args.renovation_activity,
                    )
        if result.get("response_type") == "FIXED_REPLY" and (
            args.command != "consult" or args.format == "text"
        ):
            print(result["message"])
        elif args.command == "consult" and args.format == "text":
            print(knowledge_text(result))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except POCError as exc:
        print(
            json.dumps({"error": exc.code, "message": exc.message, "status": exc.status}, ensure_ascii=False)
        )
        return 1
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": "CONFIGURATION_ERROR", "message": str(exc)}, ensure_ascii=False))
        return 1
    finally:
        if app:
            app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
