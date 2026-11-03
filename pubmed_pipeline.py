import os
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit("Set GOOGLE_API_KEY")

description = (Path(__file__).parent / "dataset_description.txt").read_text()

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
        "Find papers whose title contains the clinical measurement, "
        "death or mortality, and intensive care or ICU.",
    ),
])

chain = prompt | llm
answer = chain.invoke({"description": description}).content
if isinstance(answer, list):
    answer = answer[0]["text"]
print(answer)
