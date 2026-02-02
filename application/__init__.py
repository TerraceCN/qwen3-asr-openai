# -*- coding: utf-8 -*-

from fastapi import FastAPI, Form, Header, UploadFile
from fastapi.responses import JSONResponse
from httpx import AsyncClient
from loguru import logger
from pydantic import BaseModel

from application.utils import convert_file_to_base64, upload_file

BASE64_MAX_FILE_SIZE = (1024 * 1024 * 10) / 1.334

app = FastAPI()


class AudioTranscriptionReq(BaseModel):
    file: UploadFile
    model: str = "qwen3-asr-flash"
    language: str | None = None
    prompt: str | None = None
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

    # 获取文件内容
    file_size = req.file.size
    if not file_size:
        raise ValueError("File is empty")

    if file_size < BASE64_MAX_FILE_SIZE:  # 小于10M的文件转base64
        logger.debug(f"file size: {file_size}, using base64")
        input_audio = await convert_file_to_base64(req.file)
    else:  # 大于10M的文件上传到临时OSS
        logger.debug(f"file size: {file_size}, using oss")
        input_audio = await upload_file(authorization, model_name, req.file)
        logger.debug(f"input_audio: {input_audio}")

    # 构造消息
    messages: list[dict] = [
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
    ]
    if req.prompt:
        messages.insert(0, {"role": "system", "content": req.prompt})

        logger.debug(f"prompt: {req.prompt}")

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
            headers={
                "X-DashScope-OssResourceResolve": "enable"
                if input_audio.startswith("oss://")
                else "disable"
            },
        )
    assert resp.is_success, (
        f"Failed to call ASR, HTTP {resp.status_code}: {resp.text!r}"
    )

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
