# -*- coding: utf-8 -*-

import base64
import os
from typing import BinaryIO

from anyio import to_thread
from fastapi import UploadFile
import httpx
from magika import Magika

magika = Magika()


async def get_file_content_type(file: UploadFile):
    def magika_identify(bio: BinaryIO):
        res = magika.identify_stream(bio)
        content_type = res.output.mime_type
        extensions = res.output.extensions
        if extensions:
            ext = extensions[0]
        else:
            ext = ""
        return content_type, ext

    if file.content_type:
        if "/" in file.content_type:
            _, ext = file.content_type.split("/")
        else:
            ext = ""
        content_type = file.content_type
    elif not file.filename:
        return await to_thread.run_sync(magika_identify, file.file)
    elif file.filename.endswith(".wav"):
        content_type, ext = "audio/wav", ".wav"
    elif file.filename.endswith(".mp3"):
        content_type, ext = "audio/mpeg", ".mp3"
    else:
        raise ValueError("Unsupported file type")

    return content_type, ext


async def convert_file_to_base64(file: UploadFile):
    content_type, _ = await get_file_content_type(file)
    await file.seek(0)
    file_content = await file.read()
    file_b64 = base64.b64encode(file_content).decode("utf-8")
    return f"data:{content_type};base64,{file_b64}"


async def get_upload_policy(api_key: str, model: str) -> dict:
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
    ) as client:
        res = await client.get(
            "https://dashscope.aliyuncs.com/api/v1/uploads",
            params={"action": "getPolicy", "model": model},
        )
    assert res.is_success, (
        f"Failed to get upload policy, HTTP {res.status_code}: {res.text}"
    )

    json_data = res.json()
    return json_data["data"]


async def upload_file_to_oss(policy: dict, file: UploadFile):
    root, _ = os.path.splitext(file.filename)
    _, ext = await get_file_content_type(file)
    filename = f"{root}{ext}"

    key = f"{policy['upload_dir']}/{filename}"
    await file.seek(0)
    async with httpx.AsyncClient() as client:
        res = await client.post(
            policy["upload_host"],
            data={
                "OSSAccessKeyId": policy["oss_access_key_id"],
                "Signature": policy["signature"],
                "policy": policy["policy"],
                "x-oss-object-acl": policy["x_oss_object_acl"],
                "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
                "key": key,
                "success_action_status": "200",
            },
            files={"file": (filename, file.file)},
        )
    assert res.is_success, f"Failed to upload file, HTTP {res.status_code}: {res.text}"

    return f"oss://{key}"


async def upload_file(api_key: str, model: str, file: UploadFile):
    policy = await get_upload_policy(api_key, model)
    return await upload_file_to_oss(policy, file)
