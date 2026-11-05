"""The complete model-facing tool allowlist. Confirmation and writes are absent."""

from ..domain import POCError
from .planner import PLAN


class ControlledTools:
    def __init__(self, actions, token):
        self._actions, self._token = actions, token

    def execute(self, raw_plan, *, message, unit_id, knowledge_mode, renovation_activity, request_key):
        call = PLAN.validate_python(raw_plan)
        args = call.arguments
        if call.name == "consult":
            if args.question != message:
                raise POCError("INVALID_TOOL_INPUT", "查询问题不能被替换，请重新提问", 400)
            return self._actions.answer_question(self._token, message, service_code=args.service_code,
                                                 unit_id=unit_id, knowledge_mode=knowledge_mode,
                                                 renovation_activity=renovation_activity)
        if call.name == "prepare_repair":
            if args.description != message:
                raise POCError("INVALID_TOOL_INPUT", "报修描述不能由模型改写，请重新说明故障", 400)
            if not unit_id:
                raise POCError("UNIT_REQUIRED", "请先在侧栏选择已关联的房屋", 400)
            return self._actions.prepare_work_order(self._token, {
                "description": message, "unit_id": unit_id, "service_code": "maintenance",
                "idempotency_key": request_key,
            })
        if call.name == "track_order":
            return self._actions.get_work_order_status(self._token, args.number)
        raise POCError("INVALID_TOOL", "该操作不是允许的业务工具", 400)
