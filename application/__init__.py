# -*- coding: utf-8 -*-

from fastapi import FastAPI, Form, Header, UploadFile, HTTPException, status
from loguru import logger
from pydantic import BaseModel

from application.asr.openai import asr_openai
from application.utils.audio import get_input_audio

BASE64_MAX_FILE_SIZE = (1024 * 1024 * 10) / 1.334
SUPPORTED_ASR_MODELS = {
    "qwen3-asr-flash": "openai",
    "qwen3-asr-flash-filetrans": "dashscope",
    "qwen3-asr-flash-realtime": "websocket",
}

app = FastAPI()


class AudioTranscriptionReq(BaseModel):
    file: UploadFile
    model: str = "qwen3-asr-flash"
    prompt: str | None = None
    language: str | None = None
    stream: bool = False


@app.post("/v1/audio/transcriptions")
async def v1_audio_transcriptions(
    req: AudioTranscriptionReq = Form(...),
    authorization: str = Header(...),
):
    # 解析模型名和ASR参数
    model_name = req.model
    asr_options = {}
    if req.language:  # 语言
        asr_options["language"] = req.language
    if req.model.endswith(":itn"):  # 逆文本标准化
        asr_options["enable_itn"] = True
        model_name = req.model.rstrip(":itn")

    logger.debug(f"model: {model_name}, asr_options: {asr_options}")
    if model_name not in SUPPORTED_ASR_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model_name}",
        )

    # 获取音频文件数据
    input_audio = await get_input_audio(req.file, authorization, model_name)

    if SUPPORTED_ASR_MODELS[model_name] == "openai":
        return await asr_openai(
            model_name,
            input_audio,
            authorization,
            req.prompt,
            asr_options,
            req.stream,
        )
    elif SUPPORTED_ASR_MODELS[model_name] == "dashscope":
        raise NotImplementedError("DashScope ASR is not supported yet")
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Model {model_name} does not support "/v1/audio/transcriptions" endpoint',
        )
