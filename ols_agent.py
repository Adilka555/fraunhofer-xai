import os
import time
import requests
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent


def load_env():
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# The tool only searches. The agent chooses the short medical phrase.
@tool
def search_ols(phrase: str) -> str:
    """Search OLS for a short medical phrase. Returns up to 5 concepts, each with a label and a code."""
    response = None
    for attempt in range(4):
        try:
            response = requests.get(
                "https://www.ebi.ac.uk/ols4/api/search",
                params={"q": phrase, "rows": 5, "lang": "en"},
                headers={"Accept": "application/json", "User-Agent": "fraunhofer-xai"},
                timeout=40,
            )
            if response.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    time.sleep(0.3)
    docs = response.json().get("response", {}).get("docs", [])
    if not docs:
        return "no hit"
    lines = []
    for doc in docs:
        label = doc.get("label") or ""
        code = doc.get("obo_id") or doc.get("short_form") or ""
        lines.append(f"{label} | {code}")
    return "\n".join(lines)


class Concept(BaseModel):
    column: str = Field(description="Column name from the input")
    term: str = Field(description="OLS label, or empty if there is no hit")
    code: str = Field(description="OLS code, or empty if there is no hit")


class OLSAnswer(BaseModel):
    concepts: list[Concept]


load_env()
if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit("Set GOOGLE_API_KEY")

description = (ROOT / "dataset_description.txt").read_text()
llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)
agent = create_agent(
    model=llm,
    tools=[search_ols],
    response_format=OLSAnswer,
    system_prompt=(
        "You map clinical column names to OLS concepts. "
        "You only know the column name and the general dataset description. "
        "For each column, call search_ols with a short medical phrase, "
        "not the raw column name. "
        "Fill one concept per column. "
        "Use the concept whose name matches the phrase. "
        "If nothing matches, leave term and code empty."
    ),
)


def lookup(features):
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                f"Dataset description:\n{description}\n\n"
                "Top columns:\n"
                + "\n".join(features)
            ),
        }]
    })
    return result["structured_response"]


if __name__ == "__main__":
    print(lookup(["age", "d1_heartrate_max", "d1_sysbp_min"]))
