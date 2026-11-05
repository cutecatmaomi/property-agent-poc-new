"""Interactive local model configuration. Never prints or places the API key in shell history."""

import getpass
import json
import os

from property_ontology.agent.llm import LLMSettings
from property_ontology.config import ROOT


def main():
    print("配置兼容OpenAI的模型接口。直接回车使用已核实的DeepSeek接口与模型；不会自动充值或调用API。")
    base_url = input("接口地址 [https://api.deepseek.com]：").strip() or "https://api.deepseek.com"
    model = input("模型名 [deepseek-flash]：").strip() or "deepseek-flash"
    key = getpass.getpass("粘贴API Key后回车（不会显示字符）：").strip()
    settings = LLMSettings(mode="online", base_url=base_url, model=model, api_key=key)
    path = ROOT / "config/llm.local.json"
    if path.exists() and input("已有模型配置。输入“覆盖”保存新配置，其余输入退出：").strip() != "覆盖":
        print("未修改已有配置。")
        return
    data = settings.model_dump(mode="json", exclude={"api_key"})
    data["api_key"] = settings.api_key.get_secret_value()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)
    print("已保存本机配置。下一步运行：.venv/bin/python scripts/check_llm.py")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, EOFError, KeyboardInterrupt):
        raise SystemExit("配置未完成；请确认输入了API Key，不要把密钥发到聊天中。") from None
