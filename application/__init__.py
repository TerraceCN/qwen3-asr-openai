# -*- coding: utf-8 -*-

from typing import Callable, Awaitable

from fastapi import (
    FastAPI,
    Request,
    Response,
    Form,
    UploadFile,
    HTTPException,
    status,
)
from pydantic import BaseModel

from application.asr import asr_openai, asr_dashscope_async
from application.utils.audio import get_input_audio
from application.vars import auth_token

BASE64_MAX_FILE_SIZE = (1024 * 1024 * 10) / 1.334
SUPPORTED_ASR_MODELS = {
    "qwen3-asr-flash": "openai",
    "qwen3-asr-flash-filetrans": "dashscope_async",
}

app = FastAPI()


@app.middleware("http")
async def set_auth_token(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
):
    authorization = request.headers.get("Authorization")
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization required",
        )
    token = auth_token.set(authorization)
    try:
        return await call_next(request)
    finally:
        auth_token.reset(token)


class AudioTranscriptionReq(BaseModel):
    file: UploadFile
    model: str = "qwen3-asr-flash"
    prompt: str | None = None
    language: str | None = None
    stream: bool = False


@app.post("/v1/audio/transcriptions")
async def v1_audio_transcriptions(req: AudioTranscriptionReq = Form(...)):
    # 解析模型名和itn参数
    model_name = req.model
    enable_itn = False
    if req.model.endswith(":itn"):  # 逆文本标准化
        enable_itn = True
        model_name = req.model.rstrip(":itn")

    if model_name not in SUPPORTED_ASR_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model_name}",
        )

    if SUPPORTED_ASR_MODELS[model_name] == "openai":
        input_audio = await get_input_audio(req.file, model_name)
        return await asr_openai(
            model_name,
            input_audio,
            stream=req.stream,
            prompt=req.prompt,
            language=req.language,
            enable_itn=enable_itn,
        )
    elif SUPPORTED_ASR_MODELS[model_name] == "dashscope_async":
        if req.stream:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Model "{model_name}" does not support streaming',
            )
        input_audio = await get_input_audio(req.file, model_name, force_oss=True)
        return await asr_dashscope_async(
            model_name,
            input_audio,
            language=req.language,
            enable_itn=enable_itn,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Model {model_name} does not support "/v1/audio/transcriptions" endpoint',
        )
