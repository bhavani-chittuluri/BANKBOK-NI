import sqlite3

DB_PATH = "bank.db"

# Connect to DB
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# 🔥 Delete old table (important!)
c.execute("DROP TABLE IF EXISTS users")

# 🔥 Create correct users table that matches Flask requirements
c.execute("""
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_number TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT,
    balance REAL DEFAULT 0
)
""")

# 🔥 Insert sample users (10)
users = [
    ("8123623741", "Muruga@123", "Muruga S", "muruga.ca@gmail.com", "6513429873", 250000),
    ("8912672463", "Tharunika@123", "Tharunika K", "tharunika3@gmail.com", "9812327638", 420000),
    ("23647126543", "Krishna@123", "Krishna P", "krishna4@gmail.com", "9856437865", 300000),
    ("4523678123", "Anjali@456", "Anjali R", "anjali12@gmail.com", "9123456780", 150000),
    ("9812365478", "Rohit@789", "Rohit S", "rohit99@gmail.com", "9988776655", 500000),
    ("8745612390", "Priya@321", "Priya M", "priya34@gmail.com", "9876543210", 275000),
    ("6234789123", "Suresh@654", "Suresh K", "suresh@gmail.com", "9123478560", 325000),
    ("7123456789", "Neha@987", "Neha P", "neha@gmail.com", "9876123450", 400000),
    ("5987123465", "Vikram@111", "Vikram R", "vikram12@gmail.com", "9765432109", 200000),
    ("4871236598", "Sneha@222", "Sneha L", "sneha@gmail.com", "9654321789", 180000)
]

c.executemany("""
INSERT INTO users
(account_number, password, name, email, phone, balance)
VALUES (?, ?, ?, ?, ?, ?)
""", users)

conn.commit()
conn.close()

print("✅ Users table recreated to match bank.db structure and 10 sample users added!")
