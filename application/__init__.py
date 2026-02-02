# -*- coding: utf-8 -*-

from fastapi import FastAPI, Form, Header, UploadFile
from loguru import logger
from pydantic import BaseModel

from application.asr.openai import asr_openai
from application.utils.audio import get_input_audio

BASE64_MAX_FILE_SIZE = (1024 * 1024 * 10) / 1.334

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

    # 获取音频文件数据
    input_audio = await get_input_audio(req.file, authorization, model_name)
    use_oss = input_audio.startswith("oss://")

    # 构造消息
    messages: list[dict] = []
    if req.prompt:
        messages.append({"role": "system", "content": req.prompt})
        logger.debug(f"prompt: {req.prompt}")
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": input_audio,
                    },
                },
            ],
        },
    )

    return await asr_openai(
        model_name,
        messages,
        authorization,
        asr_options,
        req.stream,
        use_oss,
    )
