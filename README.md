# Parakeet ASR Server — local run, no Docker

## Setup

```powershell
cd parakeet-server-local
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server up at http://localhost:8000. First run downloads model, take while.

## Endpoints

- `GET /health` — status check
- `POST /v1/audio/transcriptions` — multipart form, field `file` (16kHz mono wav), optional `model_name`

## Test

```powershell
curl.exe -F "file=@sample.wav" http://localhost:8000/v1/audio/transcriptions
```

## Config

Set model via env var (see `.env.example`):

```powershell
$env:PARAKEET_MODEL="nvidia/parakeet-unified-en-0.6b"
```
