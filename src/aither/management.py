"""Management API for Aither SDK - API keys, org, user operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import httpx

from aither.models import APIKey, APIKeyWithSecret, Organization, UsageStats, User

if TYPE_CHECKING:
    from aither.client import AitherClient


def _update_usage_from_response(
    client: "AitherClient", response: httpx.Response
) -> None:
    """Parse rate limit headers and update client usage info."""
    # Reuse the client's existing header parsing logic
    client._parse_rate_limit_headers(response)


class APIKeysNamespace:
    """Namespace for API key management operations.

    Requires admin scope on the API key.

    Example:
        keys = aither.api_keys.list()
        new_key = aither.api_keys.create(name="Production")
        aither.api_keys.revoke(key_id="...")
    """

    def __init__(self, get_client: callable) -> None:
        self._get_client = get_client

    def _client(self) -> "AitherClient":
        return self._get_client()

    def list(self) -> list[APIKey]:
        """List all API keys for the organization (masked).

        Returns:
            List of APIKey objects (without secrets).

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        client = self._client()
        response = httpx.get(
            f"{client.endpoint}/api-keys",
            headers=client._build_headers("application/json"),
            timeout=client.timeout,
        )
        response.raise_for_status()
        _update_usage_from_response(client, response)
        return [APIKey(**key) for key in response.json()]

    def create(
        self,
        name: Optional[str] = None,
        scopes: Optional[list[Literal["read", "write", "admin"]]] = None,
        expires_in_days: Optional[int] = None,
    ) -> APIKeyWithSecret:
        """Create a new API key.

        Args:
            name: Optional name for the key.
            scopes: Scopes for the key (default: ["read"]).
            expires_in_days: Days until expiration (1-3650, or None for no expiry).

        Returns:
            APIKeyWithSecret with the full key (only shown once!).

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        client = self._client()
        payload = {}
        if name is not None:
            payload["name"] = name
        if scopes is not None:
            payload["scopes"] = scopes
        if expires_in_days is not None:
            payload["expires_in_days"] = expires_in_days

        response = httpx.post(
            f"{client.endpoint}/api-keys",
            json=payload,
            headers=client._build_headers("application/json"),
            timeout=client.timeout,
        )
        response.raise_for_status()
        _update_usage_from_response(client, response)
        return APIKeyWithSecret(**response.json())

    def revoke(self, key_id: str) -> None:
        """Revoke an API key.

        This operation is idempotent - revoking an already-revoked key succeeds.

        Args:
            key_id: The ID of the key to revoke.

        Raises:
            httpx.HTTPStatusError: If the key doesn't exist (404).
        """
        client = self._client()
        response = httpx.delete(
            f"{client.endpoint}/api-keys/{key_id}",
            headers=client._build_headers("application/json"),
            timeout=client.timeout,
        )
        response.raise_for_status()
        _update_usage_from_response(client, response)


class OrgNamespace:
    """Namespace for organization operations.

    Example:
        org = aither.org.get()
        usage = aither.org.usage()
    """

    def __init__(self, get_client: callable) -> None:
        self._get_client = get_client

    def _client(self) -> "AitherClient":
        return self._get_client()

    def get(self) -> Organization:
        """Get current organization info.

        Returns:
            Organization object with id, name, plan, created_at.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        client = self._client()
        response = httpx.get(
            f"{client.endpoint}/org",
            headers=client._build_headers("application/json"),
            timeout=client.timeout,
        )
        response.raise_for_status()
        _update_usage_from_response(client, response)
        return Organization(**response.json())

    def usage(self) -> UsageStats:
        """Get usage stats for the current billing period.

        Returns:
            UsageStats with api_calls, predictions_logged, etc.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        client = self._client()
        response = httpx.get(
            f"{client.endpoint}/org/usage",
            headers=client._build_headers("application/json"),
            timeout=client.timeout,
        )
        response.raise_for_status()
        _update_usage_from_response(client, response)
        return UsageStats(**response.json())


class UserNamespace:
    """Namespace for user operations.

    Example:
        me = aither.user.me()
    """

    def __init__(self, get_client: callable) -> None:
        self._get_client = get_client

    def _client(self) -> "AitherClient":
        return self._get_client()

    def me(self) -> User:
        """Get current user info.

        Returns:
            User object with id, email, name, created_at.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        client = self._client()
        response = httpx.get(
            f"{client.endpoint}/user/me",
            headers=client._build_headers("application/json"),
            timeout=client.timeout,
        )
        response.raise_for_status()
        _update_usage_from_response(client, response)
        return User(**response.json())
