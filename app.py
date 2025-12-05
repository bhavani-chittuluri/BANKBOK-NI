from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session, flash, send_file
)
import os
import csv
from datetime import datetime, timedelta

# ---------------- BOT (milestone2.py) ----------------
from milestone2 import BankAssistant, verify_user_login, get_account

# ---------------- DB ----------------
from db import (
    get_db,
    get_user_by_account,
    save_chat,
    get_transactions,
    get_total_queries,
    get_total_intents,
    get_recent_chats
)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "super-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- INIT BOT ----------------
bot = BankAssistant()
bot.memory = {}
bot.session.logout()

# ---------------- RESET BOT ----------------
def reset_bot():
    bot.session.logout()
    bot.session.slots = {}
    bot.memory = {}

# ---------------- LOGIN CHECK ----------------
def logged_in():
    return "account" in session


# -----------------------------------------------------
# PAGE ROUTES
# -----------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login-page")
def login_page():
    return render_template("login.html")


@app.route("/chat-page")
def chat_page():
    if not logged_in():
        return redirect("/login-page")
    return render_template("chat.html")


@app.route("/customer-dashboard")
def customer_dashboard():
    if not logged_in():
        return redirect("/login-page")
    return render_template("customer_dashboard.html")


@app.route("/admin-dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect("/login-page")
    return render_template("admin_dashboard.html")


# -----------------------------------------------------
# LOGIN API (used by login.html JS)
# -----------------------------------------------------

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    user = verify_user_login(email, password)

    if user:
        session["account"] = user["account_number"]
        session["email"] = user["email"]
        session["name"] = user["name"]
        reset_bot()

        return jsonify({"success": True, "message": "Login OK"})

    return jsonify({"success": False, "message": "Invalid email or password"})


# -----------------------------------------------------
# CHAT ENDPOINT (used by chat.html and index.html)
# -----------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat_api():
    if not logged_in():
        return jsonify({"reply": "Please login first."}), 401

    msg = request.json.get("message", "").strip()
    if msg == "":
        return jsonify({"reply": "Please type something."})

    reply, _ = bot.process(msg)

    save_chat(session["account"], msg, reply)

    return jsonify({"reply": reply})


# -----------------------------------------------------
# RESET CHAT SESSION
# -----------------------------------------------------

@app.route("/reset", methods=["POST"])
def reset_session():
    reset_bot()
    return jsonify({"status": "reset"})


# -----------------------------------------------------
# CUSTOMER PROFILE API
# -----------------------------------------------------

@app.route("/customer/profile")
def customer_profile():
    if not logged_in():
        return jsonify({"error": "Not logged in"}), 401

    user = get_user_by_account(session["account"])
    return jsonify(user)


# -----------------------------------------------------
# ADMIN API — LOGS
# -----------------------------------------------------

@app.route("/admin/logs")
def admin_logs():
    logs = get_recent_chats(limit=200)
    out = []
    for l in logs:
        out.append({
            "time": l["timestamp"],
            "account_number": l["account"],
            "user": l["user_message"],
            "message": l["user_message"],
            "reply": l["bot_response"]
        })

    return jsonify({
        "total_all": len(out),
        "logs": out
    })


# -----------------------------------------------------
# ADMIN API — USERS
# -----------------------------------------------------

@app.route("/admin/users")
def admin_users():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT account_number,name,email,phone,balance,'customer' as role FROM users")
    rows = c.fetchall()
    conn.close()

    return jsonify({"users": rows})


# -----------------------------------------------------
# LOGOUT
# -----------------------------------------------------

@app.route("/logout")
def logout():
    reset_bot()
    session.clear()
    return redirect("/login-page")


# -----------------------------------------------------
# START APP
# -----------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=True)

