# milestone1.py

import pandas as pd
import ast
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ======================
# Load & Prepare Data
# ======================

# Load dataset (must include 'query' and 'intent' columns)
data = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\python bank\bankbot_final_expanded1.csv")

print("Dataset loaded successfully!")
print(data.head())
# Convert entity strings (like "{'account_type': 'savings'}") into Python dicts
data["entities"] = data["entities"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})

# Split dataset into features (queries) and labels (intents)
X = data["query"]
y = data["intent"]

# Train-test split (80–20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# Build Intent Classifier
# ======================

clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("logreg", LogisticRegression(max_iter=1000))
])

# Train classifier
clf.fit(X_train, y_train)

# Evaluate performance
print("=== Intent Classification Report ===")
print(classification_report(y_test, clf.predict(X_test)))

# ======================
# Entity Extraction
# ======================

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities using spaCy and simple custom rules.
    """
    doc = nlp(text)
    spacy_entities = {ent.label_.lower(): ent.text for ent in doc.ents}

    # Add custom rule: detect account number (pure digits)
    if any(token.like_num for token in doc):
        for token in doc:
            if token.like_num and len(token.text.strip()) >= 4:
                spacy_entities["account_number"] = token.text

    # Add custom rule: detect account type keywords
    account_keywords = ["savings", "current", "salary", "business"]
    for word in account_keywords:
        if word in text.lower():
            spacy_entities["account_type"] = word

    return spacy_entities

# ======================
# Slot Filling Logic
# ======================

def fill_slots(intent, entities):
    """
    Slot filling based on intent type.
    Defines which pieces of information are needed for each intent.
    """
    slots_required = {
        "check_balance": ["account_type", "account_number"],
        "transfer_money": ["account_number", "amount", "receiver_name"],
        "card_block": ["card_type", "card_number"],
        "open_account": ["account_type"],
    }

    filled_slots = {}
    missing_slots = []

    if intent in slots_required:
        for slot in slots_required[intent]:
            if slot in entities:
                filled_slots[slot] = entities[slot]
            else:
                missing_slots.append(slot)

    return filled_slots, missing_slots

import joblib
joblib.dump(clf, "models/intent_pipeline.joblib")
print("Model saved successfully!")

# ======================
# Test Sample
# ======================

if __name__ == "__main__":
    sample = "Show me balance of my savings account 12345"
    intent = clf.predict([sample])[0]
    entities = extract_entities(sample)
    filled_slots, missing_slots = fill_slots(intent, entities)

    print("\n=== TEST QUERY ===")
    print("Query:", sample)
    print("Predicted Intent:", intent)
    print("Extracted Entities:", entities)
    print("Filled Slots:", filled_slots)
    print("Missing Slots:", missing_slots)
