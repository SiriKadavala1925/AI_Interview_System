import sqlite3
conn = sqlite3.connect("interview_system.db")
cur = conn.cursor()
cur.execute("SELECT DISTINCT subject FROM question_bank")
for row in cur.fetchall():
    print(row[0])
conn.close()
