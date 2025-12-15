import sqlite3

DB_PATH = "bank.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- CREATE DB TABLES ----------------
def create_db():
    conn = get_db()
    c = conn.cursor()

    # 1. USERS TABLE
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_number TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            balance INTEGER DEFAULT 0
        )
    """
    )

    # 2. CHAT LOGS TABLE (Updated for Analytics)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account TEXT,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            predicted_intent TEXT,
            confidence REAL,
            is_fallback BOOLEAN DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # 3. TRANSACTIONS TABLE
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_account TEXT,
            receiver_account TEXT,
            receiver_name TEXT,
            amount INTEGER,
            mode TEXT,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # 4. KNOWLEDGE BASE / FAQS TABLE
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS faqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category TEXT DEFAULT 'General'
        )
    """
    )

    conn.commit()
    conn.close()
    print("✅ Bank of Trust Database Ready (Final Schema).")


# ---------------- LOGIN CHECK ----------------
def verify_user_login(email, password):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
    row = c.fetchone()
    conn.close()
    return row


# ---------------- GET USER DETAILS ----------------
def get_user_by_account(account):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE account_number=?", (account,))
    row = c.fetchone()
    conn.close()
    return row


# ---------------- BALANCE HANDLING ----------------
def get_balance(account):
    user = get_user_by_account(account)
    if user:
        return user["balance"]
    return None


def update_balance(account, new_balance):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "UPDATE users SET balance=? WHERE account_number=?", (new_balance, account)
    )
    conn.commit()
    conn.close()


# ---------------- TRANSFER FUNDS ----------------
def transfer_funds(sender_account, receiver_account, amount):
    sender_balance = get_balance(sender_account)
    if sender_balance is None or sender_balance < amount:
        return False, "Insufficient Balance"

    receiver = get_user_by_account(receiver_account)
    if not receiver:
        return False, "Receiver account does not exist"

    update_balance(sender_account, sender_balance - amount)
    update_balance(receiver_account, receiver["balance"] + amount)

    return True, "Transfer Successful"


# ---------------- RECORD TRANSACTION ----------------
def record_transaction(sender, receiver, receiver_name, amount, mode, status):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO transactions (sender_account, receiver_account, receiver_name, amount, mode, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (sender, receiver, receiver_name, amount, mode, status),
    )
    conn.commit()
    conn.close()


# ---------------- SAVE CHAT ----------------
def save_chat(
    account, user_message, bot_response, intent=None, confidence=0.0, is_fallback=0
):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO chat_logs (account, user_message, bot_response, predicted_intent, confidence, is_fallback) 
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (account, user_message, bot_response, intent, confidence, is_fallback),
    )
    conn.commit()
    conn.close()


# ---------------- DASHBOARD: TRANSACTIONS ----------------
def get_transactions(account):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        SELECT 
            timestamp,
            sender_account,
            receiver_account,
            receiver_name,
            amount,
            mode,
            status
        FROM transactions
        WHERE sender_account=? OR receiver_account=?
        ORDER BY id DESC
    """,
        (account, account),
    )

    rows = c.fetchall()
    conn.close()

    formatted = []
    for t in rows:
        if t["sender_account"] == account:
            txn_type = f"Sent to {t['receiver_name']}"
        else:
            sender = get_user_by_account(t["sender_account"])
            sender_name = sender["name"] if sender else "Unknown"
            txn_type = f"Received from {sender_name}"

        formatted.append(
            {
                "date": t["timestamp"],
                "type": txn_type,
                "amount": t["amount"],
                "mode": t["mode"],
                "status": t["status"],
            }
        )

    return formatted


# ==========================================
#  MISSING ADMIN HELPER FUNCTIONS (Restored)
# ==========================================


def get_total_queries():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM chat_logs")
    total = c.fetchone()[0]
    conn.close()
    return total


def get_total_intents():
    conn = get_db()
    c = conn.cursor()
    # Count unique non-null intents
    c.execute(
        "SELECT COUNT(DISTINCT predicted_intent) FROM chat_logs WHERE predicted_intent IS NOT NULL"
    )
    total = c.fetchone()[0]
    conn.close()
    return total


def get_recent_chats(limit=5):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        SELECT account, user_message, bot_response, timestamp 
        FROM chat_logs 
        ORDER BY id DESC LIMIT ?
    """,
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ==========================================
#  NEW MILESTONE 4 ANALYTICS
# ==========================================


def get_analytics_stats():
    """Returns a dictionary of stats for the Admin Dashboard."""
    conn = get_db()
    c = conn.cursor()

    # 1. Total Queries
    c.execute("SELECT COUNT(*) FROM chat_logs")
    total_queries = c.fetchone()[0]

    # 2. Success Rate (Confidence > 0.65 and not fallback)
    c.execute(
        "SELECT COUNT(*) FROM chat_logs WHERE confidence > 0.65 AND is_fallback=0"
    )
    successful = c.fetchone()[0]
    success_rate = (
        round((successful / total_queries * 100), 1) if total_queries > 0 else 0
    )

    # 3. Top Intents
    c.execute(
        """
        SELECT predicted_intent, COUNT(*) as count 
        FROM chat_logs 
        WHERE predicted_intent IS NOT NULL AND predicted_intent != 'fallback'
        GROUP BY predicted_intent 
        ORDER BY count DESC LIMIT 5
    """
    )
    top_intents = c.fetchall()

    conn.close()
    return {
        "total": total_queries,
        "success_rate": success_rate,
        "top_intents": top_intents,
    }


# --- FAQ MANAGEMENT ---
def add_faq(question, answer):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO faqs (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()


def get_all_faqs():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM faqs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    create_db()
