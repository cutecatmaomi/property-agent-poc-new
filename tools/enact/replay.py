"""Run the shipped Golden Agent with isolated identity/storage and capture node outputs."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "tests/unit")]

from langchain_core.callbacks import BaseCallbackHandler
from golden_support import BY_CASE, ROWS, NoModel, PacketBusiness
from property_ontology.agent import AgentSession
from property_ontology.agent.dialogue import decision_context
from property_ontology.agent.planner import LocalDemoPlanner
from property_ontology.domain import CreateRequest, RepairDraft
from property_ontology.response_policy import classify_message


class CaptureNodes(BaseCallbackHandler):
    """Collect the real node results without changing graph routing."""

    def __init__(self, graph_name):
        self.graph_name = graph_name
        self.starts = {}
        self.outputs = []

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, name=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        outer_nodes = {"understand", "execute", "compose", "template_direct", "remember"}
        is_outer = node in outer_nodes
        if name and name == node and is_outer == (self.graph_name == "conversation"):
            self.starts[run_id] = node

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        node = self.starts.pop(run_id, None)
        if node is not None and isinstance(outputs, dict):
            self.outputs.append({"graph": self.graph_name, "node": node, "output": copy.deepcopy(outputs)})


class ObservedGraph:
    def __init__(self, graph, recorder):
        self.graph = graph
        self.recorder = recorder

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def invoke(self, input, config=None, **kwargs):
        return self.graph.invoke(input, {**(config or {}), "callbacks": [self.recorder]}, **kwargs)


MASKED = {
    "request_key", "session_id", "thread_id", "ticket", "nonce", "token",
    "confirmation_ref", "idempotency_key", "session_hash", "audit_id",
}


def mask(value):
    if isinstance(value, dict):
        return {key: "[临时测试标识已隐藏]" if key in MASKED and item is not None else mask(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask(item) for item in value]
    return value


def write_json(path, data):
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run_chat(directory, estate, question, *, trace=False, planner=None):
    directory.mkdir()
    business = PacketBusiness(directory, root=ROOT)
    session = AgentSession(business, business.token(estate), planner=planner or NoModel())
    try:
        outer, inner = CaptureNodes("conversation"), CaptureNodes("turn")
        if trace:
            session._conversation_graph = ObservedGraph(session._conversation_graph, outer)
            session._graph = ObservedGraph(session._graph, inner)
        before = decision_context(session) if trace else None
        reply = session.chat(question)
        return mask({
            "estate_id": estate,
            "user_message": question,
            "deterministic_classification": classify_message(question),
            "planning_context_before_turn": before,
            "conversation_graph_node_outputs": outer.outputs,
            "turn_graph_node_outputs": inner.outputs,
            "reply": reply,
            "work_orders_created": len(business.orders),
        })
    finally:
        session.close()


def verdict(case, run):
    reply = run["reply"]
    data = reply.get("data", {})
    checks = {
        "knowledge_set": data.get("knowledge_set") == "GOLDEN_QUESTIONS_100",
        "estate": data.get("estate_id") == case["estate_id"],
        "answer_status": data.get("status") == "ANSWERED",
        "template_source": data.get("answer_source") == "TEMPLATE",
        "golden_question_id": data.get("matched_template_id") == case["case_id"],
        "policy_id": data.get("matched_policy_ids") == [case["expected_policy_id"]],
        "policy_version": data.get("template_version_hash") == case["expected_version"],
        "authorized_answer_exact": reply["text"] == case["answer"],
        "zero_generation_calls": data.get("generation_llm_call_count") == 0,
        "exclusive_route": data.get("debug_info", {}).get("legacy_retrieval_called") is False,
        "no_work_order": run["work_orders_created"] == 0,
    }
    return {
        "question_id": case["case_id"],
        "estate_id": case["estate_id"],
        "checks": checks,
        "passed": all(checks.values()),
        "workbook_reference_exact": reply["text"] == case["reference_answer"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="Run all 100 standard questions as well")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_metadata = {
        "runtime_knowledge_set": "GOLDEN_QUESTIONS_100",
        "provenance": "实际调用本包 AgentSession.chat 与 LangGraph 节点。使用临时模拟账号和业务存储。",
        "provider_requests": 0,
        "real_residents_or_production_orders": False,
    }
    with TemporaryDirectory(prefix="enact-golden-regression-") as path:
        tmp = Path(path)
        case = BY_CASE["E001-GQ-010"]
        normal = run_chat(tmp / "normal", case["estate_id"], case["simulated_question"], trace=True)
        normal["verdict"] = verdict(case, normal)
        assert normal["verdict"]["passed"], normal["verdict"]
        assert [step["node"] for step in normal["conversation_graph_node_outputs"]] == [
            "understand", "execute", "template_direct", "remember"]
        assert [step["node"] for step in normal["turn_graph_node_outputs"]] == [
            "guard", "plan", "dialogue", "render"]
        write_json(args.output_dir / "current_success.json", {**run_metadata, **normal})

        refused = run_chat(tmp / "refused", "E001", "请帮我转人工客服。", trace=True)
        assert refused["reply"]["kind"] == "FIXED_REPLY"
        assert refused["reply"]["data"]["reason_code"] == "USER_REQUEST"
        assert refused["reply"]["data"]["created"] is False
        assert refused["reply"]["trace"] == ["guard", "render"]
        assert refused["work_orders_created"] == 0
        write_json(args.output_dir / "current_human_request.json", {**run_metadata, **refused})

        preview = run_chat(tmp / "preview", "E001", "我要报修，水龙头滴水", planner=LocalDemoPlanner())
        assert preview["reply"]["kind"] == "REPAIR_PREVIEW"
        assert preview["reply"]["data"]["expires_in_seconds"] == 600
        assert preview["work_orders_created"] == 0
        write_json(args.output_dir / "repair_preview.json", {**run_metadata, **preview})
        write_json(args.output_dir / "repair_actions_schema.json", {
            "prepare_work_order_request": RepairDraft.model_json_schema(),
            "create_work_order_request": CreateRequest.model_json_schema(),
        })

        if args.all:
            results = [
                verdict(case, run_chat(tmp / case["case_id"], case["estate_id"], case["simulated_question"]))
                for case in ROWS
            ]
            summary = {
                **run_metadata,
                "total": len(results),
                "passed": sum(item["passed"] for item in results),
                "by_estate": dict(Counter(item["estate_id"] for item in results if item["passed"])),
                "workbook_reference_exact": sum(item["workbook_reference_exact"] for item in results),
                "workbook_reference_differences": [item["question_id"] for item in results
                                                   if not item["workbook_reference_exact"]],
                "results": results,
            }
            write_json(args.output_dir / "golden_regression.json", summary)
            assert summary["total"] == summary["passed"] == 100
            assert summary["by_estate"] == {"E001": 50, "E002": 50}
            assert summary["workbook_reference_differences"] == ["E001-GQ-009"]
            print(f"Golden regression {summary['passed']}/{summary['total']}")


if __name__ == "__main__":
    main()
