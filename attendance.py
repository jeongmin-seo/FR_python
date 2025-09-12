"""
출석 체크 시간 기록용 DB
uvicorn attendance:app --reload --host 0.0.0.0 --port 8000 로 실행
인식 되고 나면 requests.post("http://localhost:8000/attendance", json= {...})
방식으로 REST API 호출하면 기록
"""
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

DB_PATH = "attendance.db"

# DB 초기화
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        student_name TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        cam_id TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()


# 입력용 모델
class AttendanceIn(BaseModel):
    student_id: str
    student_name: str
    cam_id: str | None = None


@app.post("/attendance")
def record_attendance(data: AttendanceIn):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO attendance (student_id, student_name, cam_id)
        VALUES (?, ?, ?)
    """, (data.student_id, data.student_name, data.cam_id))
    conn.commit()
    conn.close()
    return {"status": "success", "student_id": data.student_id, "time": datetime.now()}


@app.get("/attendance/{date}")
def get_attendance_by_date(date: str):
    """
    날짜별 출석 기록 조회
    예: /attendance/2025-09-12
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT student_id, student_name, timestamp, cam_id
        FROM attendance
        WHERE date(timestamp) = ?
    """, (date,))
    rows = cursor.fetchall()
    conn.close()
    return {"date": date, "records": rows}


@app.get("/attendance/student/{student_id}")
def get_attendance_by_student(student_id: str):
    """
    특정 학생의 전체 출석 기록 조회
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT student_id, student_name, timestamp, cam_id
        FROM attendance
        WHERE student_id = ?
        ORDER BY timestamp DESC
    """, (student_id,))
    rows = cursor.fetchall()
    conn.close()
    return {"student_id": student_id, "records": rows}
