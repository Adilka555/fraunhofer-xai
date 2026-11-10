import os
import time
import requests
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
QUERY_PATH = ROOT / "pubmed_query.txt"


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


def query_template():
    if QUERY_PATH.exists():
        return QUERY_PATH.read_text().strip()

    load_env()
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY")

    description = (ROOT / "dataset_description.txt").read_text()
    llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Write one PubMed search query. "
            "Put {{term}} where the medical term goes. "
            "Reply with the query only, on one line.",
        ),
        (
            "human",
            "Dataset description:\n{description}\n\n"
            "Find papers whose title contains the term, "
            "death or mortality, and intensive care or ICU.",
        ),
    ])
    answer = (prompt | llm).invoke({"description": description}).content
    if isinstance(answer, list):
        answer = answer[0]["text"]
    template = answer.strip()
    if "{term}" not in template:
        raise SystemExit("Query template has no {term}")
    QUERY_PATH.write_text(template + "\n")
    return template


def clean_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text
    return ""


def pubmed_get(url, params):
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=40)
            if response.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            time.sleep(0.4)
            return response.json()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise SystemExit("PubMed did not answer")


def search_ids(query):
    data = pubmed_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 2,
            "tool": "fraunhofer-xai",
        },
    )
    return data.get("esearchresult", {}).get("idlist", [])


def pubmed_titles(ids):
    clean = []
    for value in ids:
        text = clean_id(value)
        if text and text not in clean:
            clean.append(text)
    if not clean:
        return {}

    result = pubmed_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        {
            "db": "pubmed",
            "retmode": "json",
            "id": ",".join(clean),
            "tool": "fraunhofer-xai",
        },
    ).get("result", {})
    titles = {}
    for pmid in result.get("uids", []):
        titles[pmid] = (result.get(pmid) or {}).get("title") or ""
    return titles


# The tool is a normal function the agent is allowed to call.
# It receives one term, fills the saved query, and comes back with two papers.
@tool
def find_papers(term: str) -> str:
    """Find two PubMed papers for one medical term about death in intensive care."""
    query = query_template().replace("{term}", term)
    ids = search_ids(query)
    titles = pubmed_titles(ids)
    lines = [query]
    if not ids:
        lines.append("no papers")
    for pmid in ids:
        lines.append(pmid + " " + titles.get(pmid, ""))
    return "\n".join(lines)


class Paper(BaseModel):
    pmid: str = Field(description="PubMed id")
    title: str = Field(description="Article title")


class PubMedAnswer(BaseModel):
    term: str = Field(description="The clinical term that was searched")
    query: str = Field(description="The PubMed query the tool used")
    papers: list[Paper] = Field(description="Up to two papers")


load_env()
if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit("Set GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)
agent = create_agent(
    model=llm,
    tools=[find_papers],
    response_format=PubMedAnswer,
    system_prompt=(
        "You look up literature for one clinical term. "
        "Call find_papers with that term. "
        "Fill the answer from the tool result. "
        "If there are no papers, leave the list empty."
    ),
)


def lookup(term):
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Find papers for the term: {term}",
        }]
    })
    return result["structured_response"]


if __name__ == "__main__":
    print(lookup("Heart Rate"))
