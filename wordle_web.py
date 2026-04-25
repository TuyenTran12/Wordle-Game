"""
Wordle Web — AI-powered Wordle solver.
Run: python wordle_web.py
"""

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import requests
import os

app = FastAPI(title="Wordle AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

API_BASE = "https://wordle.votee.dev:8000"

# ── API Client ─────────────────────────────────────────────────────────────────

class WordleAPI:
    def __init__(self, base_url: str = API_BASE, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict) -> list:
        try:
            r = requests.get(self.base_url + path, params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=504, detail="API timed out")
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Cannot reach API server")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 422:
                d = e.response.json().get("detail", [])
                msgs = [x.get("msg", str(x)) for x in d] if isinstance(d, list) else [str(d)]
                raise HTTPException(status_code=422, detail=", ".join(msgs))
            raise

    def daily(self, guess: str) -> list:
        return self._get("/daily", {"guess": guess.lower().strip(), "size": 5})

    def random(self, guess: str, size: int = 5, seed: int = None) -> list:
        p = {"guess": guess.lower().strip(), "size": size}
        if seed:
            p["seed"] = seed
        return self._get("/random", p)

    def specific(self, word: str, guess: str) -> list:
        return self._get(f"/word/{word.strip().lower()}", {"guess": guess.lower().strip()})


api = WordleAPI()


# ── API JSON Endpoints (for browser JS) ─────────────────────────────────────────

@app.get("/daily")
async def daily_api(guess: str = Query(...), size: int = Query(5)):
    return api.daily(guess)


@app.get("/random")
async def random_api(guess: str = Query(...), size: int = Query(5), seed: int = Query(None)):
    size = max(3, min(8, size))
    return api.random(guess, size=size, seed=seed)


@app.get("/word/{word}")
async def word_api(word: str, guess: str = Query(...)):
    return api.specific(word, guess)


# ── Page Routes (HTML) ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/daily-page", response_class=HTMLResponse)
async def daily_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/random-page", response_class=HTMLResponse)
async def random_page(request: Request, size: int = 5, seed: int = None):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/word-page/{word}", response_class=HTMLResponse)
async def word_page(request: Request, word: str):
    return templates.TemplateResponse("index.html", {"request": request})
