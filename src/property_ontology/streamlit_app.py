"""Property chat with an explicitly configured online planner or clearly labelled local demo."""

import streamlit as st

from property_ontology.agent.session import ESTATES
from property_ontology.agent.ui_backend import open_session, timed_reply
from property_ontology.domain import POCError


def show_reply(reply):
    # Plain text keeps document contents from becoming executable HTML, scripts or hidden links.
    if reply.get('data', {}).get('quote_notice'):
        st.caption(reply['data']['quote_notice'])
    st.text(reply.get("data", {}).get("answer_text", reply["text"]))
    if (seconds := reply.get("response_time_seconds")) is not None:
        st.caption(f"本次回答耗时 {seconds:.2f} 秒")


def main():
    st.set_page_config(page_title="华润置地物业客服 POC", page_icon="🏡", layout="wide")
    # Only the framework's running animation is hidden; connection/errors remain visible.
    st.html('<style>[data-testid="stStatusWidget"]:has([data-testid="stStatusWidgetRunningIcon"]) {display: none !important;}</style>')
    st.title("华润置地物业管家")
    st.caption("提供物业知识咨询、模拟报修和本人工单进度查询。")
    if st.session_state.get("answer_contract_version") != "steward-v3":
        st.session_state["messages"] = []
        st.session_state["answer_contract_version"] = "steward-v3"
    st.session_state.setdefault("messages", [])
    agent = st.session_state.get("agent_session")
    if agent is None:
        st.info("请先用模拟住户账号登录。仅使用Mock身份与房屋数据。")
        with st.form("login", clear_on_submit=True):
            estate_id = st.selectbox("选择楼盘", list(ESTATES), format_func=lambda e: e + " · " + ESTATES[e], key="login_estate")
            username = st.selectbox("模拟住户", ["owner1", "owner2"], key="login_username")
            password = st.text_input("模拟账号密码", type="password", key="login_password")
            submit = st.form_submit_button("登录", type="primary")
        st.caption("请选择对应楼盘的模拟住户账号登录。")
        if submit:
            try:
                st.session_state["agent_session"] = open_session(estate_id, username, password)
                st.session_state["messages"] = []
                st.rerun()
            except (POCError, ValueError) as exc:
                st.error(exc.message if isinstance(exc, POCError) else "数据库未配置，请先按启动说明启动Neo4j。")
        return
    try:
        agent.verify()
    except POCError as exc:
        st.error(exc.message)
        if st.button("重新登录", key="expired_logout"):
            agent.close()
            st.session_state.clear()
            st.rerun()
        return
    if st.session_state.get('knowledge_epoch') != getattr(agent, 'knowledge_epoch', None):
        st.session_state['messages'] = []
        st.session_state['knowledge_epoch'] = getattr(agent, 'knowledge_epoch', None)
    with st.sidebar:
        st.subheader(ESTATES[agent.estate_id])
        st.caption("仅处理当前楼盘及本人关联房屋的业务。")
        if getattr(agent, 'knowledge_epoch', None):
            from property_ontology.qa.day1_poc import configuration
            active = configuration(agent.actions.settings.root)['active_set']
            st.caption("当前使用 Golden Questions：E001、E002各50条。仅限POC演示，历史资料不代表现行政策。")
        st.caption(getattr(agent.planner, "label", "本地演示（未连接在线模型）"))
        if st.button("退出 / 切换账号或楼盘", key="logout"):
            agent.close()
            st.session_state.clear()
            st.rerun()
        units = {u["unit_id"]: u for u in agent.context["units"]}
        options = list(units) or [None]
        unit_id = st.selectbox("本人的房屋", options, index=options.index(agent.unit_id) if agent.unit_id in options else 0,
                               format_func=lambda u: (units[u]["building_name"] + " · " + units[u]["room_no"]
                                                      + ("（可报修）" if units[u]["can_create_work_order"] else "（仅咨询）")) if u else "暂无有效关联房屋",
                               key="unit_choice")
        mode = st.selectbox("知识用途", ["demo", "current"], format_func=lambda m: "POC 演示资料" if m == "demo" else "现行政策（需正式审核与有效期）", key="knowledge_mode")
        activity = st.selectbox("装修活动（咨询装修收费时选择）", [None, "BARE", "MINOR", "DEMOLITION", "FLOOR_HEATING"],
                                format_func=lambda a: {None: "未指定", "BARE": "毛坯装修", "MINOR": "简单装修", "DEMOLITION": "拆改", "FLOOR_HEATING": "地暖施工"}[a], key="renovation_choice")
        try:
            if agent.set_context(unit_id, mode, activity):
                st.session_state["messages"] = []
                st.session_state.pop("last_created_ticket", None)
                st.info("已切换查询上下文，旧聊天及待确认草稿已清理。")
        except POCError as exc:
            st.error(exc.message)
            st.stop()
    if not st.session_state["messages"]:
        st.info("可以直接说说遇到的情况，也可以咨询物业知识、提交模拟报修或查询本人工单。")
    for entry in st.session_state["messages"]:
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.text(entry["text"])
            else:
                show_reply(entry["reply"])
    pending = agent.pending
    if pending:
        with st.container(border=True):
            st.subheader("请确认本次报修")
            selected = units[pending["data"]["unit_id"]]
            st.text("房屋：" + selected["building_name"] + " · " + selected["room_no"])
            st.text("故障：" + pending["data"]["description"])
            details = pending["data"].get("dialogue_details", {})
            if details.get("available_time"):
                st.text("方便上门的时间意向：" + details["available_time"])
            st.caption("核对内容后再确认；当前尚未收到成功编号时，请保留同一张卡片重试。")
            if pending.get("blocked_reason"):
                st.warning(pending["blocked_reason"])
            left, right = st.columns(2)
            accept = left.button("确认提交报修", type="primary", key="confirm_repair", disabled=bool(pending.get("blocked_reason")))
            cancel = right.button("取消本次报修", key="cancel_repair")
            if accept or cancel:
                reply = timed_reply(agent.confirm, pending["ticket"], accept=accept)
                st.session_state["messages"].append({"role": "assistant", "text": reply["text"], "reply": reply})
                if reply["kind"] == "ORDER_CREATED":
                    st.session_state["last_created_ticket"] = pending["ticket"]
                st.rerun()
    elif st.session_state.get("last_created_ticket"):
        if st.button("重复确认测试（应返回原工单）", key="repeat_repair"):
            reply = timed_reply(agent.confirm, st.session_state["last_created_ticket"], accept=True)
            st.session_state["messages"].append({"role": "assistant", "text": reply["text"], "reply": reply})
            st.rerun()
    if prompt := st.chat_input("请输入问题或具体报修故障，最多300字", max_chars=300, key="chat"):
        with st.chat_message("user"):
            st.text(prompt)
        reply = timed_reply(agent.chat, prompt)
        st.session_state["messages"].extend([{"role": "user", "text": prompt},
                                             {"role": "assistant", "text": reply["text"], "reply": reply}])
        st.rerun()


if __name__ == "__main__":
    main()
