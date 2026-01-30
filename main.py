# -*- coding: utf-8 -*-

import base64
from typing import BinaryIO

from anyio import to_thread
from fastapi import FastAPI, Form, Header, UploadFile
from fastapi.responses import JSONResponse
from httpx import AsyncClient
from loguru import logger
from magika import Magika
from pydantic import BaseModel

app = FastAPI()
magika = Magika()


class AudioTranscriptionReq(BaseModel):
    file: UploadFile
    model: str = "qwen3-asr-flash"
    language: str | None = None
    prompt: str | None = None
    stream: bool = False


async def get_content_type(file: UploadFile):
    def magika_identify(bio: BinaryIO):
        res = magika.identify_stream(bio)
        return res.output.mime_type

    if file.content_type:
        return file.content_type
    elif not file.filename:
        return await to_thread.run_sync(magika_identify, file.file)
    elif file.filename.endswith(".wav"):
        return "audio/wav"
    elif file.filename.endswith(".mp3"):
        return "audio/mpeg"
    else:
        raise ValueError("Unsupported file type")


@app.post("/v1/audio/transcriptions")
async def v1_audio_transcriptions(
    req: AudioTranscriptionReq = Form(...),
    authorization: str = Header(...),
):
    file_content = await req.file.read()
    file_b64 = base64.b64encode(file_content).decode("utf-8")
    await req.file.seek(0)
    content_type = await get_content_type(req.file)

    logger.debug(f"input file: {content_type}, size: {len(file_content)}")

    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:{content_type};base64,{file_b64}",
                    },
                },
            ],
        },
    ]
    if req.prompt:
        messages.insert(0, {"role": "system", "content": req.prompt})

        logger.debug(f"prompt: {req.prompt}")

    model_name = req.model
    asr_options = {}
    if req.language:
        asr_options["language"] = req.language
    if req.model.endswith(":itn"):
        asr_options["enable_itn"] = True
        model_name = req.model.rstrip(":itn")

    logger.debug(f"model: {model_name}, asr_options: {asr_options}")

    async with AsyncClient(
        headers={"Authorization": authorization},
        timeout=300,
    ) as client:
        resp = await client.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            json={
                "model": model_name,
                "messages": messages,
                # "stream": req.stream,  # TODO: stream
                "asr_options": asr_options,
            },
        )
    assert resp.is_success, f"HTTP {resp.status_code}: {resp.text!r}"

    resp_json = resp.json()
    choices = resp_json["choices"]
    assert len(choices) == 1
    text = choices[0]["message"]["content"]
    usage = resp_json["usage"]

    logger.debug(f"transcription: {text!r}")

    return JSONResponse(
        {
            "text": text,
            "usage": {
                "type": "tokens",
                "input_tokens": usage["prompt_tokens"],
                "input_token_details": {
                    "text_tokens": usage["prompt_tokens_details"]["text_tokens"],
                    "audio_tokens": usage["prompt_tokens_details"]["audio_tokens"],
                },
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
