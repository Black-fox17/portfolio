from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, date, timedelta
import sqlite3
from typing import List, Optional
import hashlib

app = FastAPI()

# Enable CORS for your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://salam-portfolio-three.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('portfolio_analytics.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_hash TEXT,
            visit_date DATE,
            page_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_visitor_hash(request: Request) -> str:
    # Create a unique visitor identifier using IP and user agent
    ip = request.client.host
    user_agent = request.headers.get('user-agent', '')
    visitor_id = f"{ip}-{user_agent}"
    return hashlib.md5(visitor_id.encode()).hexdigest()

@app.post("/api/track-visit")
async def track_visit(request: Request):
    visitor_hash = get_visitor_hash(request)
    page_path = request.headers.get('referer', '/')
    
    conn = sqlite3.connect('portfolio_analytics.db')
    c = conn.cursor()
    
    # Record the visit
    c.execute('''
        INSERT INTO visits (visitor_hash, visit_date, page_path)
        VALUES (?, ?, ?)
    ''', (visitor_hash, date.today(), page_path))
    
    conn.commit()
    conn.close()
    
    return {"status": "success"}

@app.get("/api/analytics/daily")
async def get_daily_analytics():
    conn = sqlite3.connect('portfolio_analytics.db')
    c = conn.cursor()
    
    # Get unique visitors for today
    c.execute('''
        SELECT COUNT(DISTINCT visitor_hash)
        FROM visits
        WHERE visit_date = ?
    ''', (date.today(),))
    
    visitors_count = c.fetchone()[0]
    conn.close()
    
    return {
        "date": date.today().isoformat(),
        "unique_visitors": visitors_count
    }

@app.get("/api/analytics/history")
async def get_analytics_history(days: Optional[int] = 7):
    conn = sqlite3.connect('portfolio_analytics.db')
    c = conn.cursor()
    
    history = []
    for i in range(days):
        target_date = date.today() - timedelta(days=i)
        c.execute('''
            SELECT COUNT(DISTINCT visitor_hash)
            FROM visits
            WHERE visit_date = ?
        ''', (target_date,))
        
        count = c.fetchone()[0]
        history.append({
            "date": target_date.isoformat(),
            "unique_visitors": count
        })
    
    conn.close()
    return {"history": history}
