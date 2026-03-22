from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.session import engine, Base

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Error initializing database: {e}")

app = FastAPI(
    title="Hybrid HR Analytics System",
    description="Backend for HR Analytics with ML, NLP and AI Agents",
    version="1.0.0"
)

# CORS Middleware
origins = [
    "http://localhost:5173",  # React Frontend
    "http://localhost:5174",  # React Frontend (alternate port)
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Hybrid HR Analytics API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

from app.api.v1.api import api_router
from app.core.config import settings

app.include_router(api_router, prefix=settings.API_V1_STR)
