import random

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from ols_agent import lookup as ols_lookup
from pubmed_agent import lookup as pubmed_lookup

DROP_COLUMNS = [
    "encounter_id",
    "patient_id",
    "hospital_id",
    "icu_id",
    "hospital_death",
    "apache_4a_hospital_death_prob",
    "apache_4a_icu_death_prob",
]
CAT_COLUMNS = [
    "ethnicity",
    "gender",
    "hospital_admit_source",
    "icu_admit_source",
    "icu_stay_type",
    "icu_type",
    "apache_3j_bodysystem",
    "apache_2_bodysystem",
    "apache_2_diagnosis",
    "apache_3j_diagnosis",
]

model = CatBoostClassifier()
model.load_model("models/icu_death.cbm")

df = pd.read_csv("data/training_v2.csv")
y = df["hospital_death"].astype(int)
X = df.drop(columns=DROP_COLUMNS)
for col in CAT_COLUMNS:
    X[col] = X[col].fillna("missing").astype(str)

X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)

row = random.randrange(len(X_test))
patient = X_test.iloc[[row]]
risk = float(model.predict_proba(patient)[:, 1][0])
died = int(y_test.iloc[row])

shap_row = model.get_feature_importance(
    Pool(patient, cat_features=CAT_COLUMNS),
    type="ShapValues",
)[0]
pushes = shap_row[:-1]
order = np.argsort(np.abs(pushes))[::-1][:6]
top = [(patient.columns[i], float(pushes[i])) for i in order]

print("test row", row)
print("died", died)
print("risk", round(risk, 3))
print()

concepts = ols_lookup([name for name, push in top]).concepts
by_column = {item.column: item for item in concepts}

for name, push in top:
    concept = by_column.get(name)
    term = concept.term if concept else ""
    code = concept.code if concept else ""
    print(name, round(push, 3), "|", term, "|", code)
    if not term:
        print("no papers")
        print()
        continue
    papers = pubmed_lookup(term)
    if not papers.papers:
        print("no papers")
    for paper in papers.papers:
        print(paper.pmid, paper.title)
    print()
