# -*- coding: utf-8 -*-

import asyncio

from fastapi.responses import JSONResponse
import httpx

ASYNC_TASK_CHECK_INTERVAL = 3


async def parse_result(transcription_url: str, authorization: str):
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

    return {
        "text": transcript["text"],
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


async def get_task_result(task_id: str, authorization: str):
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
    if output["task_status"] == "RUNNING":
        return None
    elif output["task_status"] == "FAILED":
        raise Exception(
            f"ASR task failed, code: {output['code']}, message: {output['message']}"
        )
    elif output["task_status"] == "SUCCEEDED":
        return await parse_result(output["result"]["transcription_url"], authorization)
    else:
        raise ValueError(f"Unknown ASR task status: {output['task_status']}")

async def asr_dashscope_async(
    model: str,
    input_audio: str,
    authorization: str,
    asr_options: dict | None = None,
):
    # 构造请求参数
    req_json = {"model": model, "input": {"file_url": input_audio}}
    if asr_options is not None:
        req_json["parameters"] = asr_options
    
    # 提交任务
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

    while True:
        result = await get_task_result(task_id, authorization)

        if result is None:
            await asyncio.sleep(ASYNC_TASK_CHECK_INTERVAL)
            continue

        return JSONResponse(content=result)
