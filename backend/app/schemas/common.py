"""The response envelope kept simple and consistent (Section 7):

    { "success": true, "data": { ... }, "message": "" }
"""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ResponseEnvelope(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    message: str = ""
