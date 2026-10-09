import os
import subprocess
import threading
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="AI Trading Bot Gateway")

def run_streamlit():
    port = os.environ.get("STREAMLIT_PORT", "8501")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run([
        "streamlit", "run", "src/dashboard.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.headless=true"
    ], env=env)

def run_scanner_task():
    try:
        result = subprocess.run(
            ["python", "cron_scanner.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Scanner cron executed successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Scanner cron failed: {e.stderr}")

@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=run_streamlit, daemon=True)
    t.start()

@app.get("/")
async def root():
    return {"status": "online", "service": "AI Trading Bot Gateway", "endpoints": ["/scan", "/health"]}

@app.post("/scan")
async def trigger_scan(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scanner_task)
    return JSONResponse(status_code=202, content={"status": "accepted", "message": "Volume profile scan triggered"})

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
