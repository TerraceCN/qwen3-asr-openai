# -*- coding: utf-8 -*-

import json
from typing import AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from loguru import logger

from application.utils.timer import Timer

BASE64_MAX_FILE_SIZE = (1024 * 1024 * 10) / 1.334


async def _handle_non_stream(resp: httpx.Response, timer: Timer):
    timer.stop()

    resp_json = resp.json()
    choices = resp_json["choices"]
    assert len(choices) == 1
    text = choices[0]["message"]["content"]
    usage = resp_json["usage"]
    rtf = round(timer.get_time() / usage["seconds"], 2)

    logger.debug(f"time: {timer}, RTF: {rtf}, transcription: {text!r}")

    return {
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


async def _handle_stream(resp: httpx.Response, timer: Timer) -> AsyncIterator[str]:
    seconds: float | None = None
    text: str = ""

    async for chunk in resp.aiter_lines():
        chunk = chunk.strip()
        if not chunk.startswith("data:"):
            continue
        data_str = chunk[5:].strip()
        if data_str == "[DONE]":
            break
        data: dict = json.loads(data_str)

        usage: dict | None = data.get("usage")
        if usage:
            seconds = usage["seconds"]

        choices: list | None = data.get("choices")
        if not choices:
            continue
        delta: dict = choices[0]["delta"]
        content: str | None = delta.get("content")
        if content is None:
            continue
        text += content

        resp_data = json.dumps(
            {"type": "transcript.text.delta", "delta": content},
            ensure_ascii=False,
        )
        yield f"data: {resp_data}\n\n"

    timer.stop()

    resp_data = json.dumps(
        {
            "type": "transcript.text.done",
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
        },
        ensure_ascii=False,
    )
    yield f"data: {resp_data}\n\n"
    
    if seconds:
        rtf = round(timer.get_time() / seconds, 2)
    else:
        rtf = None
    logger.debug(f"time: {timer}, RTF: {rtf}, transcription: {text!r}")


async def asr_openai(
    model: str,
    messages: list[dict],
    authorization: str,
    asr_options: dict | None = None,
    stream: bool = False,
    use_oss: bool = False,
):
    # 构造请求参数
    req_json = {"model": model, "messages": messages}
    if asr_options is not None:
        req_json["asr_options"] = asr_options
    if stream:
        req_json["stream"] = True
        req_json["stream_options"] = {"include_usage": True}

    # 构造请求
    client = httpx.AsyncClient(
        headers={"Authorization": authorization},
        timeout=300,
    )
    req = client.build_request(
        method="POST",
        url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        json=req_json,
        headers={"X-DashScope-OssResourceResolve": "enable" if use_oss else "disable"},
    )

    # 发送请求
    timer = Timer()
    timer.start()
    resp = await client.send(req, stream=stream)
    if resp.is_error:
        await resp.aread()
        assert resp.is_success, (
            f"Failed to call dashscope, HTTP {resp.status_code}: {resp.text!r}"
        )

    if not stream:
        return JSONResponse(
            content=await _handle_non_stream(resp, timer), background=client.aclose
        )
    else:
        return StreamingResponse(
            _handle_stream(resp, timer),
            media_type="text/event-stream",
            background=client.aclose,
        )
