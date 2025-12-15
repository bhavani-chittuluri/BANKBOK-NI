from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    flash,
    Response,
)
import csv
import pandas as pd
import io
from datetime import datetime, timedelta
import os
from flask import redirect, url_for, send_file

# ---------------- BOT LOGIC ----------------
from milestone2 import generate_bot_response

TRAINING_FILE = "bankbot_final_expanded1.csv"

# ---------------- DATABASE ----------------
from bank_db import (
    get_db,
    get_user_by_account,
    verify_user_login,
    save_chat,
    get_transactions,
    get_recent_chats,
    get_analytics_stats,
    add_faq,
    get_all_faqs,
)

# ---------------- FLASK CONFIG ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "bot_secure_key_2025"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- ROUTE FIXES ----------------
@app.route("/login-page")
def login_page_alias():
    return redirect(url_for("login"))

@app.route('/favicon.ico')
def favicon():
    favicon_path = os.path.join(app.root_path, 'static', 'favicon.ico')
    if os.path.exists(favicon_path):
        return send_file(favicon_path)
    return "", 204

# ---------------- AUTH HELPERS ----------------
def require_login():
    return bool(session.get("account"))

# ---------------- ROUTE: HOME ----------------
@app.route("/")
def admin_home():
    return render_template("admin_home.html", current_year=datetime.now().year)

# ---------------- ROUTE: USER LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        user = verify_user_login(email, password)

        if user:
            session["account"] = user["account_number"]
            session["email"] = user["email"]
            session["name"] = user["name"]
            session["balance"] = user["balance"]
            session["phone"] = user["phone"]

            return redirect(url_for("dashboard"))

        flash("❌ Invalid Username or Password", "error")
        return render_template("Login.html")

    return render_template("Login.html")

# ---------------- ROUTE: DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if not require_login():
        return redirect(url_for("login"))

    user = get_user_by_account(session["account"])
    transactions = get_transactions(session["account"])

    return render_template(
        "dashboard.html",
        name=user["name"],
        account=user["account_number"],
        balance=f"{user['balance']:,}",
        email=user["email"],
        phone=user["phone"],
        transactions=transactions,
    )

# ---------------- ROUTE: CHAT UI ----------------
@app.route("/chat")
def chat():
    if not require_login():
        return redirect(url_for("login"))
    return render_template("chat.html")

# ---------------- API: GET BOT RESPONSE ----------------
@app.route("/get_response", methods=["POST"])
def get_response():
    if not require_login():
        return jsonify({"response": "Session expired. Please login again."}), 401

    user_msg = request.json.get("message", "").strip()

    try:
        intent, entities, reply, confidence = generate_bot_response(user_msg)
    except Exception as e:
        print("BOT ERROR:", e)
        reply = "⚠️ System Error"
        intent, confidence = "error", 0.0

    save_chat(
        session["account"],
        user_msg,
        reply,
        intent,
        confidence,
        1 if intent == "fallback" else 0,
    )

    return jsonify({"response": reply, "intent": intent, "confidence": confidence})

# ---------------- ROUTE: CHAT HISTORY ----------------
@app.route("/chat_logs")
def chat_logs():
    if not require_login():
        return redirect(url_for("login"))

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT user_message, bot_response, timestamp FROM chat_logs WHERE account=? ORDER BY id DESC",
        (session["account"],),
    )
    rows = c.fetchall()
    conn.close()

    formatted_logs = []
    for r in rows:
        ts = r["timestamp"]
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") + timedelta(hours=5, minutes=30)
            ts = dt.strftime("%d %b %I:%M %p")
        except:
            pass
        formatted_logs.append((r["user_message"], r["bot_response"], ts))

    return render_template("chat_logs.html", logs=formatted_logs)

# ---------------- EXPORT USER LOGS ----------------
@app.route("/export_excel")
def export_excel():
    if not require_login():
        return redirect(url_for("login"))

    filename = f"Statement_{session['account']}.csv"
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["User Message", "Bot Response", "Timestamp"])

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT user_message, bot_response, timestamp FROM chat_logs WHERE account=? ORDER BY id",
        (session["account"],),
    )
    rows = c.fetchall()
    conn.close()

    for r in rows:
        writer.writerow([r["user_message"], r["bot_response"], r["timestamp"]])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

# ---------------- ADMIN LOGIN ----------------
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("username") == "admin_bot" and request.form.get("password") == "trust@2025":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        flash("❌ Invalid Admin Credentials", "error")
    return render_template("admin_login.html")

# ---------------- ADMIN DASHBOARD ----------------
@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    stats = get_analytics_stats()

    return render_template(
        "admin_dashboard.html",
        stats=stats,
        total_queries=stats["total"],
        accuracy=f"{stats['success_rate']}%",
        recent_queries=get_recent_chats(limit=8),
    )

# =================================================
# NEW MILESTONE 4 ADMIN ROUTES (CLEANED)
# =================================================

@app.route('/admin/training_data', methods=['GET', 'POST'])
def manage_training_data():
    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 403

    file_path = os.path.join(BASE_DIR, TRAINING_FILE)

    if request.method == 'POST':
        new_text = request.form.get('text', '').strip()
        new_intent = request.form.get('intent', '').strip()
        new_response = request.form.get('response', '').strip()
        
        try:
            file_exists = os.path.exists(file_path)
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['text', 'intent', 'response', 'entities'])
                writer.writerow([new_text, new_intent, new_response, '{}'])
            return jsonify({"status": "success", "msg": "Training example added."})
        except Exception as e:
            return jsonify({"status": "error", "msg": str(e)})

    try:
        if not os.path.exists(file_path):
            return jsonify([])

        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', engine='python')

        df = df.fillna("")
        for col in ['text', 'intent', 'response']:
            if col not in df.columns:
                df[col] = ""

        data = df.tail(50).to_dict(orient='records')
        return jsonify(data)

    except Exception as e:
        print(f"CSV Load Error: {e}")
        return jsonify([])

@app.route("/admin/faqs", methods=["GET", "POST"])
def manage_faqs():
    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 403

    if request.method == "POST":
        question = request.form.get("question")
        answer = request.form.get("answer")
        add_faq(question, answer)
        return jsonify({"status": "success", "msg": "FAQ added."})

    faqs = get_all_faqs()
    faq_list = [{"id": f[0], "question": f[1], "answer": f[2]} for f in faqs]
    return jsonify(faq_list)

@app.route("/admin/export_logs")
def admin_export_logs():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM chat_logs")
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "ID",
            "Account",
            "User Message",
            "Bot Response",
            "Intent",
            "Confidence",
            "Is Fallback",
            "Timestamp",
        ]
    )
    writer.writerows(rows)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=admin_system_logs.csv"},
    )

@app.route('/admin/logs_json')
def admin_logs_json():
    if not session.get("admin"):
        return jsonify([])

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM chat_logs ORDER BY id DESC LIMIT 100")
    rows = c.fetchall()
    conn.close()

    logs_data = [dict(row) for row in rows]
    return jsonify(logs_data)

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("admin_home"))

# ---------------- RUNNER ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
