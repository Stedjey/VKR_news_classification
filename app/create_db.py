import sqlite3

def create_db():
    conn = sqlite3.connect("bot_messages.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            text TEXT,
            predicted_category TEXT,
            confidence REAL,
            keywords TEXT,
            emotion TEXT,
            sentiment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Вызови функцию для создания таблицы при запуске
create_db()