import io
import os
import tempfile

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

import nemo.collections.asr as nemo_asr

MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-unified-en-0.6b")

app = FastAPI(title="Parakeet ASR Server", version="1.0.0")

model = None


@app.on_event("startup")
def load_model():
    global model
    print(f"Loading Parakeet model: {MODEL_NAME} ...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    model.eval()

    # pretrained checkpoint ships without validation_ds config; nemo's
    # transcribe() dataloader setup reads it unconditionally, so stub it in.
    from omegaconf import OmegaConf

    if model.cfg.get("validation_ds") is None:
        model.cfg.validation_ds = OmegaConf.create({})
    print("Parakeet model loaded and ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model_name: str = Form(default="parakeet")):
    audio_bytes = await file.read()
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # if sample_rate != 16000:
    #     return JSONResponse(
    #         status_code=400,
    #         content={"error": "Audio must be 16kHz mono. Convert with ffmpeg first."},
    #     )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio_array, sample_rate)
        result = model.transcribe([tmp_path])
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
    finally:
        os.remove(tmp_path)

    return {"text": text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
