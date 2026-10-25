from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import DATA_DIR
from app.database import init_db
from app.routers import analyze, observations, mosaic

app = FastAPI(
    title="LunarVision API",
    description="Multi-modal, Sun-angle and scale invariant image correspondence "
                "for Chandrayaan-2 OHRC/TMC-2/IIRS imagery (SIH26166 prototype).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# Serve generated/processed/synthetic images so the frontend can <img src=...> them directly
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

app.include_router(analyze.router)
app.include_router(observations.router)
app.include_router(mosaic.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "LunarVision API"}
