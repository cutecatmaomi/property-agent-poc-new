"""Construct one uncached controller per verified browser login; no shared chat or model state."""

from time import perf_counter

from ..actions import PropertyActions
from ..runtime import RuntimeSettings
from .llm import build_planner
from .session import AgentSession


def timed_reply(callback, *args, **kwargs):
    """Keep server response duration with the UI message, outside its answer and model history."""
    started = perf_counter()
    reply = dict(callback(*args, **kwargs))
    reply["response_time_seconds"] = max(0.0, perf_counter() - started)
    return reply


def open_session(estate_id, username, password):
    actions = PropertyActions(RuntimeSettings.load())
    token = None
    try:
        token = actions.login(estate_id, username, password)["token"]
        return AgentSession(actions, token, planner=build_planner(), owns_actions=True)
    except BaseException:
        if token:
            actions.logout(token)
        actions.close()
        raise
