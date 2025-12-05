# milestone2.py
"""
Smart Bank Assistant - milestone2.py
Final improved version (full file).
- Uses ML intent prediction if available.
- Hybrid rules + dataset-driven responses.
- Session-based context/slot tracking.
- Strict DB account validation (account_no column).
- Session login: asks account number once and remembers it.
- Transfer flow with detailed receipt.
- Card services flow improved: after user selects a card menu option (1-4),
  the bot asks "Credit Card or Debit Card?" and uses the answer together with
  the selected option to produce the result.
- Loan services, greeting recognition, exit commands, change account, escalation.
- Uses dataset CSV for responses and fallback (bankbot_final_expanded1.csv).
- Responds with trailing tag: [ intent: xxx ]
- Keep personalization OFF by default.
"""

import os
import re
import sys
import random
import sqlite3
import joblib
import pandas as pd
import difflib
from datetime import datetime

# --------------------------
# CONFIGURATION
# --------------------------
DB_PATH = "bank.db"
MODEL_PATH = os.path.join("models", "intent_pipeline.joblib")
DATA_FILE = "bankbot_final_expanded1.csv"
CONFIDENCE_THRESHOLD = 0.55  # confidence below triggers fallback flow
EXIT_COMMANDS = {"exit", "quit", "bye", "stop"}
CHANGE_ACCOUNT_COMMANDS = {"change account", "switch account", "use different account"}
ESCALATE_KEYWORDS = {"human", "agent", "representative", "support", "talk to", "help me", "someone"}
GREETINGS_REGEX = re.compile(r'\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b', re.I)
PERSONALIZATION_ENABLED = False  # toggled by user prompt at runtime (off by default)

# Keywords list for spell-correction and rescue
COMMON_KEYWORDS = ["loan", "card", "transfer", "balance", "hello", "hi", "pay", "payment"]
LOAN_KEYWORDS = {"loan", "loans", "home loan", "personal loan", "car loan", "education loan"}
CARD_KEYWORDS = {"card", "cards", "credit card", "debit card", "block card", "card limit", "card apply"}

# Policy limits / thresholds used in eligibility checks
MIN_AGE_CARD = 18
MIN_AGE_LOAN = 21
MIN_MONTHLY_SALARY_FOR_LOAN = 10000.0   # example threshold
MIN_MONTHLY_SALARY_FOR_CARD = 10000.0   # example threshold

import sqlite3
from datetime import datetime

DB_PATH = "bank.db"

# -------------------------- DB / UTILITY LAYER --------------------------

def ensure_db_schema(db_path="bank.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table only if it does not exist; do NOT drop existing data
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_number TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        phone TEXT,
        balance REAL DEFAULT 0
    );
    """)

    conn.commit()
    conn.close()



def account_exists(account_number, db_path=DB_PATH):
    """Check if a user exists in 'users' table by account_number."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE account_number = ?", (account_number,))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def get_account(account_number, db_path=DB_PATH):
    """Return account info from 'users' table."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT account_number, name, balance, email, phone FROM users WHERE account_number = ?", (account_number,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "account_number": row[0],
            "name": row[1],
            "balance": row[2],
            "email": row[3],
            "phone": row[4]
        }
    return None


def update_balance(account_number, new_balance, db_path=DB_PATH):
    """Update balance in 'users' table."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE users SET balance = ? WHERE account_number = ?", (new_balance, account_number))
    conn.commit()
    conn.close()


def record_transaction(from_acc, to_acc, amount, remark="", db_path=DB_PATH):
    """Record a transaction row in 'transactions' table."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO transactions (from_account, to_account, amount, timestamp, remark) VALUES (?, ?, ?, ?, ?)",
        (from_acc, to_acc, amount, ts, remark)
    )
    txn_id = c.lastrowid
    conn.commit()
    conn.close()
    return txn_id, ts

# ----------------------- LOGIN FUNCTION -----------------------

def verify_user_login(email, password, db_path=DB_PATH):
    """
    Verify login by email and password using 'users' table.
    Returns dict if successful, None otherwise.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT account_number, name, email, phone, balance
        FROM users
        WHERE email = ? AND password = ?
    """, (email, password))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "account_number": row[0],
            "name": row[1],
            "email": row[2],
            "phone": row[3],
            "balance": row[4]
        }
    return None


# --------------------------
# NLU / Dataset Loading
# --------------------------


def load_intent_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"WARNING: intent model not found at '{model_path}'. Running with rule-based fallbacks.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None


def load_dataset_responses(data_file=DATA_FILE):
    """Load CSV dataset and build intent->responses mapping for reuse and fallback responses."""
    if not os.path.exists(data_file):
        print(f"WARNING: dataset CSV not found at '{data_file}'. Some fallback behaviors may not work.")
        return {}
    try:
        df = pd.read_csv(data_file, encoding='latin1')
        # Expect columns at least: 'intent' and 'response'
        if 'intent' not in df.columns or 'response' not in df.columns:
            print("WARNING: dataset missing 'intent' or 'response' columns.")
            return {}
        mapping = {}
        for _, row in df.dropna(subset=['intent', 'response']).iterrows():
            intent = str(row['intent']).strip()
            resp = str(row['response']).strip()
            mapping.setdefault(intent, []).append(resp)
        return mapping
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return {}


# --------------------------
# SPELL CORRECTION / NORMALIZATION
# --------------------------


def correct_common_mistakes(text):
    """
    Simple spell-correction for a few common keywords using difflib.
    Returns corrected text (only adjusts closest keyword matches).
    """
    if not text:
        return text
    tokens = re.findall(r'\w+|\S', text)  # keep punctuation separate
    vocab = COMMON_KEYWORDS
    new_text = text
    for token in list(set(tokens)):
        if not token.isalpha():
            continue
        low = token.lower()
        match = difflib.get_close_matches(low, vocab, n=1, cutoff=0.78)
        if match and match[0] != low:
            pattern = re.compile(r'\b' + re.escape(token) + r'\b', flags=re.I)
            new_text = pattern.sub(match[0], new_text, count=1)
            tokens = re.findall(r'\w+|\S', new_text)
    return new_text


# --------------------------
# PARSING HELPERS
# --------------------------


def try_parse_number(text):
    """
    Try to extract a sensible float/int from text.
    Accepts strings like "22,45000", "223000", "₹223,000", "22,300.50".
    Returns float or None.
    """
    if not text:
        return None
    cleaned = re.sub(r'[^\d.]', '', text)
    if cleaned == "":
        return None
    try:
        if cleaned.count('.') > 1:
            parts = cleaned.split('.')
            cleaned = parts[0] + '.' + ''.join(parts[1:])
        return float(cleaned)
    except:
        cleaned2 = re.sub(r'[,\s]', '', text)
        try:
            return float(cleaned2)
        except:
            return None


def extract_age_and_income(text):
    """
    Extract age and income fields from a free-form string such as:
      "age:22 income:400000" or "age 22 income 400000"
    Returns (age:int or None, income:float or None)
    """
    if not text:
        return None, None
    text = text.lower()
    age = None
    income = None

    m_age = re.search(r'age[:\s]*([0-9]{1,3})', text)
    if m_age:
        try:
            age = int(m_age.group(1))
        except:
            age = None

    m_inc = re.search(r'(income|salary|monthly income|annual income)[:\s]*([0-9,.\s]+)', text)
    if m_inc:
        num = try_parse_number(m_inc.group(2))
        if num is not None:
            income = float(num)

    if age is None:
        nums = re.findall(r'\d{1,7}', text)
        if len(nums) >= 1:
            first = int(nums[0])
            if 10 <= first <= 120:
                age = first
    if income is None:
        nums = re.findall(r'\d{3,15}', text)
        if nums:
            try:
                income_candidates = [float(n) for n in nums]
                income = max(income_candidates)
            except:
                income = None
    return age, income


# --------------------------
# NLU PREDICTION & HELPERS
# --------------------------


def predict_intent(model, text):
    """
    Returns (intent_label, confidence).
    If model is None, fallback to simple rules.
    """
    text = text.strip()
    if model:
        try:
            probs = model.predict_proba([text])[0]
            labels = model.classes_
            best_idx = int(probs.argmax())
            return labels[best_idx], float(probs[best_idx])
        except Exception:
            pass

    if GREETINGS_REGEX.search(text):
        return "greeting", 0.9
    lowered = text.lower()
    if any(word in lowered for word in ["balance", "bal", "balance inquiry"]):
        return "balance_inquiry", 0.8
    if "transfer" in lowered or "send money" in lowered:
        return "fund_transfer", 0.8
    if "card" in lowered:
        return "card_services", 0.7
    if "loan" in lowered:
        return "loan_services", 0.7
    if any(x in lowered for x in EXIT_COMMANDS):
        return "exit", 0.99
    return "fallback", 0.4


def sample_dataset_response(intent, dataset_map):
    """Pick a random response for an intent from the dataset mapping."""
    if not dataset_map:
        return None
    vals = dataset_map.get(intent)
    if vals:
        return random.choice(vals)
    for fallback_key in ("fallback", "unknown", "nlu_fallback", "default_fallback"):
        vals = dataset_map.get(fallback_key)
        if vals:
            return random.choice(vals)
    return None


# --------------------------
# DIALOGUE / FLOWS
# --------------------------


class Session:
    def __init__(self):
        self.account_number = None
        self.user_name = None
        self.slots = {}  # generic slot store for flows (e.g. awaiting loan selection / numeric)
        self.logged_in = False

    def login(self, account_number):
        self.account_number = account_number
        self.logged_in = True
        acc = get_account(account_number)
        if acc:
            self.user_name = acc.get("name")
        else:
            self.user_name = None

    def logout(self):
        self.account_number = None
        self.user_name = None
        self.slots = {}
        self.logged_in = False


def ask_for_account(session, dataset_map):
    """
    Ask and validate account number. Repeat until valid input or user exits.
    Returns True if logged in successfully, False if user chose to exit.
    """
    while True:
        sys.stdout.write("Bot: Please enter your account number: ")
        sys.stdout.flush()
        user_in = input().strip()
        if not user_in:
            continue
        if user_in.lower() in EXIT_COMMANDS:
            print(f"Bot: Goodbye. [ intent: exit ]")
            return False
        if account_exists(user_in):
            session.login(user_in)
            name = session.user_name or ""
            if PERSONALIZATION_ENABLED and name:
                print(f"Bot: Welcome back, {name.split()[0]}! How can I help? [ intent: session_login ]")
            else:
                print(f"Bot: Account {user_in} recognized. You are logged in. [ intent: session_login ]")
            return True
        else:
            resp = sample_dataset_response("account_not_found", dataset_map) or \
                   "I couldn't find that account. Please re-enter a valid account number."
            print(f"Bot: {resp} [ intent: account_not_found ]")


def handle_greeting(intent, confidence, dataset_map, session):
    resp = sample_dataset_response("greeting", dataset_map) or "Hello! How can I help you today?"
    if session.logged_in and PERSONALIZATION_ENABLED and session.user_name:
        resp = f"Welcome back, {session.user_name.split()[0]}! How can I help?"
    return resp


def handle_balance_inquiry(intent, confidence, dataset_map, session):
    # This intent requires account
    if not session.logged_in:
        got = ask_for_account(session, dataset_map)
        if not got:
            return None, "exit"
    acc = get_account(session.account_number)
    if not acc:
        resp = sample_dataset_response("account_not_found", dataset_map) or "Account not found. Please try again."
        return resp, "account_not_found"
    resp = sample_dataset_response("balance_inquiry", dataset_map) or f"Your current balance is {acc['balance']:.2f}."
    resp = f"{resp} (Account: {acc['account_number']}, Balance: ₹{acc['balance']:.2f})"
    return resp, "balance_inquiry"


def handle_fund_transfer(intent, confidence, dataset_map, session, text):
    """
    Fund transfer flow.
    """
    if not session.logged_in:
        got = ask_for_account(session, dataset_map)
        if not got:
            return None, "exit"

    sender_acc = get_account(session.account_number)
    if not sender_acc:
        resp = sample_dataset_response("account_not_found", dataset_map) or "Sender account not found."
        return resp, "account_not_found"

    amount = None
    receiver = None
    amount_search = re.search(r'₹?\s*([0-9]+(?:\.[0-9]{1,2})?)', text.replace(',', ''))
    if amount_search:
        try:
            amount = float(amount_search.group(1))
        except:
            amount = None

    acc_search = re.search(r'\b(\d{6,16})\b', text)
    if acc_search:
        receiver = acc_search.group(1)

    while not receiver:
        sys.stdout.write("Bot: Please provide receiver account number (or type 'cancel' to abort): ")
        sys.stdout.flush()
        u = input().strip()
        if not u:
            continue
        if u.lower() in EXIT_COMMANDS or u.lower() == "cancel":
            return "Transfer cancelled.", "transfer_cancelled"
        if account_exists(u):
            receiver = u
            break
        else:
            sys.stdout.write("Bot: I couldn't find that receiver in our bank. Do you want to proceed to an external recipient? (yes/no): ")
            sys.stdout.flush()
            ack = input().strip().lower()
            if ack in ("yes", "y"):
                receiver = u
                break
            else:
                continue

    while amount is None:
        sys.stdout.write("Bot: Enter amount to transfer (numbers only): ")
        sys.stdout.flush()
        u = input().strip()
        if not u:
            continue
        if u.lower() in EXIT_COMMANDS or u.lower() == "cancel":
            return "Transfer cancelled.", "transfer_cancelled"
        try:
            amount = float(u.replace(',', '').replace('₹', '').strip())
            if amount <= 0:
                print("Bot: Amount must be greater than 0.")
                amount = None
                continue
        except:
            print("Bot: Couldn't parse the amount. Try again.")
            amount = None
            continue

    if sender_acc["balance"] < amount:
        resp = sample_dataset_response("insufficient_funds", dataset_map) or "Insufficient balance for this transfer."
        return resp, "insufficient_funds"

    receiver_name = None
    if account_exists(receiver):
        rec_acc = get_account(receiver)
        receiver_name = rec_acc.get("name")
    confirm_msg = f"You're about to transfer ₹{amount:.2f} from {sender_acc['account_number']} to {receiver}"
    if receiver_name:
        confirm_msg += f" ({receiver_name})"
    confirm_msg += ". Confirm? (yes/no): "
    sys.stdout.write("Bot: " + confirm_msg)
    sys.stdout.flush()
    ack = input().strip().lower()
    if ack not in ("yes", "y"):
        return "Transfer cancelled.", "transfer_cancelled"

    new_sender_balance = sender_acc["balance"] - amount
    update_balance(sender_acc["account_number"], new_sender_balance)

    if account_exists(receiver):
        rec_acc = get_account(receiver)
        new_receiver_balance = rec_acc["balance"] + amount
        update_balance(receiver, new_receiver_balance)
    txn_id, ts = record_transaction(sender_acc["account_number"], receiver, amount, remark="fund_transfer")

    receipt_lines = [
        "Transfer Successful!",
        f"Transaction ID: {txn_id}",
        f"Timestamp (UTC): {ts}",
        f"From: {sender_acc['account_number']} ({sender_acc.get('name') or 'N/A'})",
        f"To: {receiver} {f'({receiver_name})' if receiver_name else ''}",
        f"Amount: ₹{amount:.2f}",
        f"Remaining Balance: ₹{new_sender_balance:.2f}",
        "If you need a copy of this receipt, say 'email receipt' or 'save receipt'."
    ]
    receipt = "\n".join(receipt_lines)
    return receipt, "fund_transfer"


# --------------------------
# Loan & Card Menus + Handlers
# --------------------------


def build_loan_menu(dataset_map):
    menu = sample_dataset_response("loan_services_menu", dataset_map)
    if menu:
        return menu
    return ("Loan Services Menu:\n"
            "1. Personal Loan - Eligibility & Documents\n"
            "2. Home Loan - Eligibility & Documents\n"
            "3. Car Loan - Eligibility & Documents\n"
            "4. Education Loan - Eligibility & Documents\n"
            "5. Loan EMI & Repayment\n"
            "Type option number or ask e.g. 'personal loan documents'.")


def handle_loan_selection(selection, dataset_map, session):
    sel = selection.strip().lower()
    if sel in ("1", "personal", "personal loan", "personal loan eligibility", "personal loan documents"):
        resp = sample_dataset_response("loan_personal", dataset_map) or sample_dataset_response("loan_documents", dataset_map)
        if resp:
            prompt = resp + "\nPlease enter your monthly salary (numbers only) to check eligibility."
            return prompt, "loan_numeric"
        return ("Personal Loan - Eligibility: Income > X, Documents: ID proof, Address proof, Salary slip.\n"
                "Please enter your monthly salary (numbers only) to check eligibility."), "loan_numeric"

    if sel in ("2", "home", "home loan", "home loan documents"):
        resp = sample_dataset_response("loan_home", dataset_map)
        if resp:
            prompt = resp + "\nIf you'd like eligibility estimate, enter your monthly salary."
            return prompt, "loan_numeric"
        return ("Home Loan - Eligibility & Documents: Property papers, ID proof, Income proof.\n"
                "Enter your monthly salary to get an eligibility estimate."), "loan_numeric"

    if sel in ("3", "car", "car loan", "car loan documents"):
        resp = sample_dataset_response("loan_car", dataset_map)
        if resp:
            prompt = resp + "\nEnter your monthly salary to know eligibility."
            return prompt, "loan_numeric"
        return ("Car Loan - Eligibility & Documents: Quotation, ID, Income.\n"
                "Enter your monthly salary to know eligibility."), "loan_numeric"

    if sel in ("4", "education", "education loan", "education loan documents"):
        resp = sample_dataset_response("loan_education", dataset_map)
        if resp:
            prompt = resp + "\nPlease enter the annual fees or monthly income for eligibility estimation."
            return prompt, "loan_numeric"
        return ("Education Loan - Eligibility & Documents: Admission proof, fee structure, ID.\n"
                "Please enter your monthly income (or expected annual fees) to continue."), "loan_numeric"

    if sel in ("5", "emi", "loan emi", "repayment"):
        resp = sample_dataset_response("loan_emi", dataset_map)
        if resp:
            prompt = resp + "\nIf you'd like an EMI estimate, enter the loan amount."
            return prompt, "loan_numeric"
        return ("Loan EMI: Provide loan amount and tenure to calculate EMI. Enter loan amount to proceed."), "loan_numeric"

    for k in ("personal", "home", "car", "education", "emi"):
        if k in sel:
            return handle_loan_selection(k, dataset_map, session)

    resp = sample_dataset_response("loan_services_fallback", dataset_map) or "Please choose a valid option from the loan menu."
    return resp, None


def build_card_menu(dataset_map):
    menu = sample_dataset_response("card_services_menu", dataset_map)
    if menu:
        return menu
    return ("Card Services Menu:\n"
            "1. Check Card Eligibility\n"
            "2. Card Limit Enquiry\n"
            "3. Block Lost/Stolen Card\n"
            "4. Apply for New Card\n"
            "Type the option number or ask, e.g. 'block my card'.")


# -- Separate card helper functions (exist in your original code structure) --

def card_eligibility_short(card_type, dataset_map):
    # Short Option A style response
    ct = "Credit Card" if "credit" in card_type.lower() else "Debit Card"
    resp = sample_dataset_response("card_eligibility", dataset_map) or ""
    short = f"{ct} Eligibility:\n✔ Minimum Age: 21\n✔ Income Proof Required\n✔ Status: Eligible"
    if resp:
        return f"{resp}\n{short}"
    return short


def card_limit_enquiry(card_type, dataset_map):
    ct = "Credit Card" if "credit" in card_type.lower() else "Debit Card"
    resp = sample_dataset_response("card_limit", dataset_map) or ""
    short = f"{ct} Limit enquiry: Limits depend on salary & credit score. Provide monthly income to get an estimate."
    if resp:
        return f"{resp}\n{short}"
    return short


def card_block_action(card_type, dataset_map):
    ct = "Credit Card" if "credit" in card_type.lower() else "Debit Card"
    resp = sample_dataset_response("card_block", dataset_map) or ""
    short = f"{ct} blocked successfully if reported lost/stolen. To confirm, provide last 4 digits."
    if resp:
        return f"{resp}\n{short}"
    return short


def card_apply_flow(card_type, dataset_map):
    ct = "Credit Card" if "credit" in card_type.lower() else "Debit Card"
    resp = sample_dataset_response("card_apply", dataset_map) or ""
    short = f"To apply for {ct}, submit ID, PAN and income proof. You can apply online or at your branch."
    if resp:
        return f"{resp}\n{short}"
    return short


def handle_card_selection(selection, dataset_map, session):
    """
    Modified: This function records the selected option in session.slots
    and returns a prompt asking for card type (credit/debit).
    """
    sel = selection.strip().lower()
    # Map common synonyms to the numbered option
    if sel in ("1", "eligibility", "check card eligibility", "card eligibility"):
        session.slots['card_pending_option'] = "1"
        # Ask for type next
        return "Which card type: Credit Card or Debit Card? (reply 'credit' or 'debit')", "card_type"
    if sel in ("2", "limit", "card limit", "card limit enquiry"):
        session.slots['card_pending_option'] = "2"
        return "Which card type: Credit Card or Debit Card? (reply 'credit' or 'debit')", "card_type"
    if sel in ("3", "block", "block card", "lost", "stolen"):
        session.slots['card_pending_option'] = "3"
        return "Which card type: Credit Card or Debit Card? (reply 'credit' or 'debit')", "card_type"
    if sel in ("4", "apply", "apply card", "card apply"):
        session.slots['card_pending_option'] = "4"
        return "Which card type: Credit Card or Debit Card? (reply 'credit' or 'debit')", "card_type"

    # If user wrote "credit card" or "debit card" directly while in selection
    if "credit" in sel or "debit" in sel:
        return "Please choose an option number from the card menu (1-4) and then specify credit or debit when prompted.", "card_selection"

    resp = sample_dataset_response("card_services_fallback", dataset_map) or "Please choose a valid option from the card menu."
    return resp, None


def perform_card_action_based_on_type(card_type, dataset_map, session):
    """
    After user replies credit/debit, this function reads session.slots['card_pending_option']
    and returns the short professional message (Option A style) tailored by card_type.
    It also clears the card_pending_option slot.
    """
    opt = session.slots.get('card_pending_option')
    session.slots.pop('card_pending_option', None)
    ct = "Credit Card" if 'credit' in card_type.lower() else "Debit Card"

    if opt == "1":
        # Check Card Eligibility
        return card_eligibility_short(card_type, dataset_map)

    if opt == "2":
        # Card Limit Enquiry
        return card_limit_enquiry(card_type, dataset_map)

    if opt == "3":
        # Block Lost/Stolen Card
        return card_block_action(card_type, dataset_map)

    if opt == "4":
        # Apply for New Card
        return card_apply_flow(card_type, dataset_map)

    # fallback
    return "Okay. How can I help further with cards?"


# --------------------------
# OTHER HANDLERS (unchanged)
# --------------------------


def handle_change_account(session, dataset_map):
    session.logout()
    print("Bot: Okay, let's change account.")
    success = ask_for_account(session, dataset_map)
    if not success:
        return None, "exit"
    return f"Switched to account {session.account_number}.", "change_account"


def handle_escalation(intent, confidence, dataset_map, session, text):
    resp = sample_dataset_response("escalate_to_human", dataset_map) or "I will connect you to a human agent. Please hold. (Simulated)"
    return resp, "escalation"


def handle_fallback(intent, confidence, dataset_map, session, text):
    resp = sample_dataset_response("fallback", dataset_map) or sample_dataset_response("default_fallback", dataset_map)
    if not resp:
        resp = "Sorry, I didn't quite get that. Could you rephrase?"
    return resp, "fallback"


# --------------------------
# MAIN DIALOGUE MANAGER
# --------------------------


class BankAssistant:
    def __init__(self, model=None, dataset_map=None):
        self.model = model
        self.dataset_map = dataset_map or {}
        self.session = Session()

    def process(self, raw_text):
        # normalization + spell-correction for common keywords
        corrected_text = correct_common_mistakes(raw_text)
        text = corrected_text.strip()
        lowered = text.lower()

        # Exit commands
        if any(cmd == lowered or cmd in lowered for cmd in EXIT_COMMANDS):
            return f"Goodbye! If you need anything else, just say hi. [ intent: exit ]", True

        # Change account commands
        if any(phrase in lowered for phrase in CHANGE_ACCOUNT_COMMANDS):
            resp, intent = handle_change_account(self.session, self.dataset_map)
            tag = f"[ intent: {intent} ]"
            return f"{resp} {tag}" if resp else f"Okay. {tag}", False

        # Escalation
        if any(k in lowered for k in ESCALATE_KEYWORDS):
            resp, intent = handle_escalation(None, None, self.dataset_map, self.session, raw_text)
            return f"{resp} [ intent: {intent} ]", False

        # If user is in loan menu awaiting selection
        if self.session.slots.get("awaiting") == "loan_selection":
            sel = text
            reply, next_state = handle_loan_selection(sel, self.dataset_map, self.session)
            self.session.slots.pop("awaiting", None)
            if next_state:
                self.session.slots["awaiting"] = next_state
            return f"{reply} [ intent: loan_services ]", False

        # If user is in card menu awaiting selection (first step: option number)
        if self.session.slots.get("awaiting") == "card_selection":
            sel = text
            reply, next_state = handle_card_selection(sel, self.dataset_map, self.session)
            # handle_card_selection now sets session.slots['card_pending_option'] and returns next_state 'card_type'
            self.session.slots.pop("awaiting", None)
            if next_state:
                # next_state will be 'card_type' when asking card type
                self.session.slots["awaiting"] = next_state
            return f"{reply} [ intent: card_services ]", False

        # If awaiting card type (credit/debit) after selecting option
        if self.session.slots.get("awaiting") == "card_type":
            # interpret reply as card type
            card_type_text = text.lower()
            if not any(k in card_type_text for k in ("credit", "debit")):
                return f"Please reply with 'credit' or 'debit' to proceed. [ intent: card_services ]", False
            # perform action using pending option
            reply = perform_card_action_based_on_type(card_type_text, self.dataset_map, self.session)
            # clear the awaiting state
            self.session.slots.pop("awaiting", None)
            return f"{reply} [ intent: card_services ]", False

        # If awaiting numeric input for loan
        if self.session.slots.get("awaiting") == "loan_numeric":
            salary = try_parse_number(text)
            if salary is None:
                age, income = extract_age_and_income(text)
                if income:
                    salary = income
                elif age and salary is None:
                    return f"Provide salary details for eligibility assessment. [ intent: loan_eligibility_check ]", False
            if salary is None:
                return f"I couldn't understand that number. Please enter your monthly salary as digits (e.g. 40000). [ intent: loan_eligibility ]", False
            monthly_salary = float(salary)
            if monthly_salary < MIN_MONTHLY_SALARY_FOR_LOAN:
                self.session.slots.pop("awaiting", None)
                reply = sample_dataset_response("loan_ineligible", self.dataset_map) or \
                        f"Based on a monthly salary of ₹{monthly_salary:.2f}, you do not meet the minimum salary requirement for most personal loans."
                return f"{reply} [ intent: loan_eligibility ]", False
            estimated_max_loan = monthly_salary * 10
            self.session.slots.pop("awaiting", None)
            reply_lines = [
                "✔ Loan Eligibility Result:",
                f"Monthly Salary: ₹{monthly_salary:,.2f}",
                f"Estimated Maximum Loan (subject to credit checks): ₹{estimated_max_loan:,.2f}",
                "Next steps: If you'd like, provide desired loan amount and tenure to get an EMI estimate."
            ]
            reply = "\n".join(reply_lines)
            return f"{reply} [ intent: loan_eligibility ]", False

        # If awaiting numeric input for card (this is separate from card_type flow)
        if self.session.slots.get("awaiting") == "card_numeric":
            digits = re.sub(r'\D', '', text)
            age, income = extract_age_and_income(text)
            if age or income:
                if age is None:
                    return f"Provide age and income for eligibility assessment. [ intent: card_eligibility_check ]", False
                if income is None:
                    return f"Provide salary details for eligibility assessment. [ intent: card_eligibility_check ]", False
                monthly_income = float(income)
                eligible = (age >= MIN_AGE_CARD) and (monthly_income >= MIN_MONTHLY_SALARY_FOR_CARD)
                status = "Eligible" if eligible else "Not Eligible"
                suggestion = ""
                if eligible:
                    suggested_cards = sample_dataset_response("card_suggest", self.dataset_map) or "Silver / Gold"
                    suggestion = f"\nYou can apply for: {suggested_cards}"
                else:
                    suggestion = "\nYou do not meet the basic eligibility. Contact support for alternatives."
                reply_lines = [
                    "✔ Eligibility Result:",
                    f"Age: {age}",
                    f"Monthly Income: ₹{monthly_income:,.2f}",
                    f"Card Status: {status}",
                    suggestion
                ]
                reply = "\n".join([line for line in reply_lines if line])
                self.session.slots.pop("awaiting", None)
                return f"{reply} [ intent: card_eligibility ]", False
            if len(digits) == 4:
                self.session.slots.pop("awaiting", None)
                resp = sample_dataset_response("card_block_confirm", self.dataset_map) or "Thanks. The card will be blocked shortly."
                return f"{resp} [ intent: card_services ]", False
            inc = try_parse_number(text)
            if inc is not None:
                monthly_income = float(inc)
                estimated_limit = monthly_income * 4
                self.session.slots.pop("awaiting", None)
                reply_lines = [
                    "✔ Card Limit Estimate:",
                    f"Monthly Income: ₹{monthly_income:,.2f}",
                    f"Estimated Card Limit: ₹{estimated_limit:,.2f}",
                    "This is an estimate subject to credit score and verification."
                ]
                reply = "\n".join(reply_lines)
                return f"{reply} [ intent: card_limit_estimate ]", False
            return f"Couldn't parse input. Provide last 4 digits of card or monthly income (e.g. age:22 income:400000). [ intent: card_services ]", False

        # Greeting quick path (always rescue)
        if GREETINGS_REGEX.search(text):
            resp = handle_greeting("greeting", 0.99, self.dataset_map, self.session)
            return f"{resp} [ intent: greeting ]", False

        # Predict intent using model (on corrected text)
        intent, confidence = predict_intent(self.model, text)
        
        # Prevent misclassification of loan -> loan_payment when just 'loan'
        payment_keywords = {"paid", "payment", "pay", "paid-off", "installment", "emi", "paid"}
        if intent and "loan" in intent and "payment" in intent:
            if "loan" in lowered and not any(pk in lowered for pk in payment_keywords):
                intent = "loan_services"
                confidence = 0.9
        
        # Low confidence rescue logic
        if confidence < CONFIDENCE_THRESHOLD:
            if GREETINGS_REGEX.search(text):
                resp = handle_greeting("greeting", 0.99, self.dataset_map, self.session)
                return f"{resp} [ intent: greeting ]", False

            if any(k in lowered for k in LOAN_KEYWORDS) or "loan" in lowered:
                menu = build_loan_menu(self.dataset_map)
                self.session.slots["awaiting"] = "loan_selection"
                return f"{menu} [ intent: loan_services ]", False

            if any(k in lowered for k in CARD_KEYWORDS) or "card" in lowered:
                menu = build_card_menu(self.dataset_map)
                self.session.slots["awaiting"] = "card_selection"
                return f"{menu} [ intent: card_services ]", False

            if "balance" in lowered or "bal" in lowered:
                resp, lbl = handle_balance_inquiry("balance_inquiry", confidence, self.dataset_map, self.session)
                if resp is None:
                    return f"Goodbye. [ intent: exit ]", True
                return f"{resp} [ intent: {lbl} ]", False

            if "transfer" in lowered or "send money" in lowered:
                resp, lbl = handle_fund_transfer("fund_transfer", confidence, self.dataset_map, self.session, text)
                return f"{resp} [ intent: {lbl} ]", False

            resp, intent_label = handle_fallback(intent, confidence, self.dataset_map, self.session, text)
            return f"{resp} [ intent: {intent_label} ]", False

        # High confidence routing
        if intent in ("balance_inquiry", "balance_check"):
            resp, label = handle_balance_inquiry(intent, confidence, self.dataset_map, self.session)
            return f"{resp} [ intent: {label} ]", False

        if intent in ("fund_transfer", "transfer", "send_money"):
            resp, label = handle_fund_transfer(intent, confidence, self.dataset_map, self.session, text)
            return f"{resp} [ intent: {label} ]", False

        if intent in ("loan_services", "loan", "loan_info", "loan_details"):
            menu = build_loan_menu(self.dataset_map)
            self.session.slots["awaiting"] = "loan_selection"
            return f"{menu} [ intent: loan_services ]", False

        if intent in ("card_services", "card", "card_info", "credit_card", "debit_card"):
            menu = build_card_menu(self.dataset_map)
            self.session.slots["awaiting"] = "card_selection"
            return f"{menu} [ intent: card_services ]", False

        if intent in ("session_login", "login"):
            if not self.session.logged_in:
                got = ask_for_account(self.session, self.dataset_map)
                if not got:
                    return f"Goodbye. [ intent: exit ]", True
                return f"Logged in to account {self.session.account_number}. [ intent: session_login ]", False
            else:
                return f"You're already logged in as {self.session.account_number}. [ intent: session_login ]", False

        if intent in ("change_account",):
            resp, label = handle_change_account(self.session, self.dataset_map)
            return f"{resp} [ intent: {label} ]", False

        if intent in ("escalate", "help", "human_agent"):
            resp, label = handle_escalation(intent, confidence, self.dataset_map, self.session, text)
            return f"{resp} [ intent: {label} ]", False

        ds_resp = sample_dataset_response(intent, self.dataset_map)
        if ds_resp:
            return f"{ds_resp} [ intent: {intent} ]", False

        resp, label = handle_fallback(intent, confidence, self.dataset_map, self.session, text)
        return f"{resp} [ intent: {label} ]", False


# --------------------------
# START / CHAT LOOP
# --------------------------


def start_chat():
    print("Starting Smart Bank Assistant (milestone2)...")
    ensure_db_schema()

    model = load_intent_model()
    dataset_map = load_dataset_responses()

    assistant = BankAssistant(model=model, dataset_map=dataset_map)

    print("\n🤖 Smart Bank Assistant is ready!\n")
    print("Bot: Hello! I'm your Smart Bank Assistant. Type 'exit' to end the chat.")
    print("Bot: (Personalization is disabled.)")

    while True:
        try:
            sys.stdout.write("You: ")
            sys.stdout.flush()
            text = input()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye! [ intent: exit ]")
            break

        if not text:
            continue

        if text.strip().lower() in EXIT_COMMANDS:
            print(f"Bot: Goodbye! [ intent: exit ]")
            break

        reply, should_exit = assistant.process(text)
        if reply is None:
            print("Bot: Goodbye. [ intent: exit ]")
            break

        print("Bot:", reply)

        if should_exit:
            break


if __name__ == "__main__":
    # Launch the interactive chat loop
    start_chat()
