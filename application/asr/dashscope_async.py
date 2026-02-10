# -*- coding: utf-8 -*-

import asyncio
import time

from fastapi.responses import JSONResponse
import httpx
from loguru import logger

from application.vars import auth_token

ASYNC_TASK_CHECK_INTERVAL = 1


async def parse_result(resp_json: dict):
    transcription_url = resp_json["output"]["result"]["transcription_url"]

    scheduled_time = time.strptime(
        resp_json["output"]["scheduled_time"], "%Y-%m-%d %H:%M:%S.%f"
    )
    end_time = time.strptime(resp_json["output"]["end_time"], "%Y-%m-%d %H:%M:%S.%f")
    time_usage = time.mktime(end_time) - time.mktime(scheduled_time)
    seconds = resp_json["usage"]["seconds"]
    rtf = time_usage / seconds
    if time_usage >= 1:
        time_usage_str = f"{time_usage:.2f}s"
    else:
        time_usage_str = f"{time_usage * 1000:.2f}ms"

    authorization = auth_token.get()
    async with httpx.AsyncClient(
        headers={"Authorization": authorization},
        timeout=300,
    ) as client:
        resp = await client.get(transcription_url)
    assert resp.is_success, (
        f"Failed to parse ASR result, HTTP {resp.status_code}: {resp.text!r}"
    )

    resp_json = resp.json()
    transcript = resp_json["transcripts"][0]
    text = transcript["text"]

    logger.debug(f"time: {time_usage_str}, RTF: {rtf:.2f}, transcription: {text!r}")

    return {
        "text": text,
        "usage": {
            "type": "tokens",
            "input_tokens": 0,
            "input_token_details": {
                "text_tokens": 0,
                "audio_tokens": 0,
            },
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


async def get_task_result(task_id: str):
    authorization = auth_token.get()
    async with httpx.AsyncClient(
        headers={"Authorization": authorization},
        timeout=300,
    ) as client:
        resp = await client.get(
            f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}",
            headers={"X-DashScope-Async": "enable"},
        )
    assert resp.is_success, (
        f"Failed to get ASR task result, HTTP {resp.status_code}: {resp.text!r}"
    )

    resp_json = resp.json()
    output = resp_json["output"]
    task_status = output["task_status"]
    logger.debug(f"ASR task {task_id}: {task_status}")
    if task_status == "FAILED":
        raise Exception(
            f"ASR task failed, code: {output['code']}, message: {output['message']}"
        )
    elif task_status == "SUCCEEDED":
        return await parse_result(resp_json)

    return None


async def asr_dashscope_async(
    model: str,
    input_audio: str,
    *,
    language: str | None = None,
    enable_itn: bool = False,
    enable_words: bool = False,
    channel_id: list[int] | None = None,
):
    # 构造请求参数
    parameters = {
        "enable_itn": enable_itn,
        "enable_words": enable_words,
        "channel_id": channel_id or [0],
    }
    if language:
        parameters["language"] = language
    req_json = {
        "model": model,
        "input": {"file_url": input_audio},
        "parameters": parameters,
    }

    # 提交任务
    authorization = auth_token.get()
    async with httpx.AsyncClient(
        headers={"Authorization": authorization},
        timeout=300,
    ) as client:
        resp = await client.post(
            "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
            json=req_json,
            headers={
                "X-DashScope-OssResourceResolve": "enable"
                if input_audio.startswith("oss://")
                else "disable",
                "X-DashScope-Async": "enable",
            },
        )
    assert resp.is_success, (
        f"Failed to create ASR task, HTTP {resp.status_code}: {resp.text!r}"
    )

    resp_json = resp.json()
    task_id = resp_json["output"]["task_id"]
    logger.debug(f"ASR task created, task_id: {task_id}")

    while True:
        result = await get_task_result(task_id)

        if result is None:
            await asyncio.sleep(ASYNC_TASK_CHECK_INTERVAL)
            continue

        return JSONResponse(content=result)
