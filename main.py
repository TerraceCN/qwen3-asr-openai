# -*- coding: utf-8 -*-

import sys

import click
from loguru import logger
import uvicorn

from application import app


@click.command()
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default=8000)
@click.option("--log-level", type=str, default="INFO")
@click.option("--access-log", "--no-access-log", is_flag=True, default=False)
def main(
    host: str,
    port: int,
    log_level: str,
    access_log: bool,
):
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    uvicorn.run(app, host=host, port=port, access_log=access_log)


if __name__ == "__main__":
    main()
