# -*- coding: utf-8 -*-

from contextvars import ContextVar


auth_token: ContextVar[str] = ContextVar("auth_token")
