"""Launch the local chat demo without telemetry or model credentials."""

import os
import subprocess
import sys

from property_ontology.config import ROOT


def main():
    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS="false", LANGSMITH_TRACING="false",
               LANGCHAIN_TRACING_V2="false")
    return subprocess.call([sys.executable, "-m", "streamlit", "run",
                            str(ROOT / "src/property_ontology/streamlit_app.py"),
                            "--server.address=127.0.0.1", "--server.port=8501",
                            "--server.headless=true", "--browser.gatherUsageStats=false"], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
