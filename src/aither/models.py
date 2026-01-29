"""Pydantic models for Aither API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class APIKey(BaseModel):
    """API key information (without the secret)."""

    id: str
    name: Optional[str] = None
    key_prefix: str
    scopes: list[str]
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    created_at: datetime


class APIKeyWithSecret(BaseModel):
    """API key with the secret (only returned on creation)."""

    id: str
    name: Optional[str] = None
    key: str  # The full key, only shown once!
    key_prefix: str
    scopes: list[str]
    expires_at: Optional[datetime] = None
    created_at: datetime


class Organization(BaseModel):
    """Organization information."""

    id: str
    name: str
    plan: str
    created_at: datetime


class UsageStats(BaseModel):
    """Usage statistics for the current billing period."""

    api_calls: int
    predictions_logged: int
    data_bytes: int
    period_start: datetime
    period_end: datetime


class User(BaseModel):
    """User information."""

    id: str
    email: str
    name: Optional[str] = None
    created_at: datetime
