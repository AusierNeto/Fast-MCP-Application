"""
BotCity Maestro API – Python Client

- Base URL: https://developers.botcity.dev
- Single entry point: MaestroClient
- Token caching + auto-refresh on 401
- Consistent response wrapper: MaestroResponse
- Sub-APIs grouped by resource (Tasks, Logs, Automations, Bots, etc.)
- All comments and docstrings in English.

Author: Oráculo's Assistant
"""

from __future__ import annotations

import time
import threading
import requests

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Iterable


# -------------------------
# Exceptions & Data Models
# -------------------------

class MaestroClientError(Exception):
    """Raised for client-side errors or non-OK responses from Maestro API."""


@dataclass
class MaestroResponse:
    """
    Standardized response wrapper for Maestro API calls.

    Attributes:
        ok: True if HTTP status is in 2xx.
        status_code: HTTP status code.
        url: Final URL that was called.
        headers: Response headers.
        data: Parsed JSON (dict/list) when possible; else raw bytes/text.
        raw: The original requests.Response object (optional for deep inspection).
    """
    ok: bool
    status_code: int
    url: str
    headers: Dict[str, str]
    data: Any
    raw: Optional[requests.Response] = None


# -------------
# Core Client
# -------------

class MaestroClient:
    """
    BotCity Maestro API client with token caching and organized resource helpers.

    Typical usage:
        client = MaestroClient(
            login="your_login",
            key="your_key",
            base_url="https://developers.botcity.dev"
        )

        # Authenticate (auto-called on first request if needed)
        client.authenticate()

        # Use resource helpers
        tasks = client.tasks.list(page=1, size=50)

        # Or make arbitrary calls
        resp = client.request_raw("GET", "/maestro/api/tasks", params={"page": 1, "size": 50})

    Notes:
        - Token & organization are retrieved by POST /maestro/api/login
        - All subsequent requests use Authorization: Bearer <token>
          and the X-Organization (or Organization) header when required by backend.
    """

    def __init__(
        self,
        login: str,
        key: str,
        base_url: str = "https://developers.botcity.dev",
        *,
        request_timeout: float = 30.0,
        token_skew: int = 10,
        default_org_header: str = "X-Organization"
    ):
        """
        Args:
            login: Maestro login credential.
            key: Maestro key credential.
            base_url: Maestro API base URL (no trailing slash required).
            request_timeout: Requests timeout in seconds.
            token_skew: Seconds to subtract from token expiry to avoid race conditions.
            default_org_header: Header name to send the organization value.
        """
        self.base_url = base_url.rstrip("/")
        self.login_value = login
        self.key_value = key
        self.timeout = request_timeout
        self.token_skew = token_skew
        self.org_header_name = default_org_header

        self._token: Optional[str] = None
        self._organization: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._lock = threading.Lock()

        # Resource helpers
        self.tasks = _TasksAPI(self)
        self.logs = _LogsAPI(self)
        self.automations = _AutomationsAPI(self)
        self.bots = _BotsAPI(self)
        self.runners = _RunnersAPI(self)
        self.credentials = _CredentialsAPI(self)
        self.datapools = _DatapoolsAPI(self)
        self.result_files = _ResultFilesAPI(self)
        self.errors = _ErrorsAPI(self)
        self.schedules = _SchedulesAPI(self)
        self.workspaces = _WorkspacesAPI(self)

    # --------
    # Auth
    # --------

    def _is_token_valid(self) -> bool:
        """Return True if we have a token and it hasn't expired considering skew."""
        return bool(self._token) and (time.time() < (self._token_expires_at - self.token_skew))

    def authenticate(self) -> MaestroResponse:
        """
        Perform authentication against Maestro login route and cache token & organization.

        Returns:
            MaestroResponse with token and organization fields.

        Raises:
            MaestroClientError on non-OK responses or malformed payloads.
        """
        url = f"{self.base_url}/api/v2/workspace/login"
        payload = {"login": self.login_value, "key": self.key_value}
        resp = requests.post(url, json=payload, timeout=self.timeout)

        if not resp.ok:
            raise MaestroClientError(f"Authentication failed: {resp.status_code} {resp.text}")

        data = _safe_json(resp)
        token = data.get("accessToken")
        organization = data.get("organizationLabel")

        if not token or not organization:
            raise MaestroClientError(f"Invalid login response. Expected token & organization. Got: {data}")

        with self._lock:
            self._token = token
            self._organization = organization
            # If the API returns an expiry value, use it here.
            # The public examples do not show it; we assume 1 hour validity by default.
            self._token_expires_at = time.time() + 3600.0

        return MaestroResponse(
            ok=True,
            status_code=resp.status_code,
            url=url,
            headers=dict(resp.headers),
            data=data,
            raw=resp
        )

    def _auth_headers(self) -> Dict[str, str]:
        """
        Build Authorization headers. Re-auth if token is absent/expired.
        """
        with self._lock:
            if not self._is_token_valid():
                self.authenticate()

            headers = {
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            }
            # Organization header (required by the API in many calls)
            if self._organization:
                headers[self.org_header_name] = self._organization
            return headers

    # ---------------
    # Core requestor
    # ---------------

    def request_raw(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_on_401: bool = True,
        stream: bool = False,
    ) -> MaestroResponse:
        """
        Low-level request method. Use this to call any route not yet wrapped.

        Args:
            method: HTTP verb (GET/POST/PUT/DELETE/PATCH).
            path: Absolute path or relative to base (e.g., "/maestro/api/tasks").
            params: Querystring parameters.
            json: JSON body.
            files: Files dict for multipart/form-data.
            headers: Additional headers (merged with auth headers).
            retry_on_401: If True, on 401 the client will re-auth and retry once.
            stream: If True, return the Response with streaming content.

        Returns:
            MaestroResponse
        """
        url = self._normalize_url(path)
        base_headers = self._auth_headers()
        if headers:
            base_headers.update(headers)

        resp = requests.request(
            method=method.upper(),
            url=url,
            headers=base_headers,
            params=params,
            json=json if files is None else None,
            files=files,
            timeout=self.timeout,
            stream=stream
        )

        # If unauthorized, try to refresh token once and retry.
        if resp.status_code == 401 and retry_on_401:
            with self._lock:
                # Force a new token
                self.authenticate()
            base_headers = self._auth_headers()
            if headers:
                base_headers.update(headers)
            resp = requests.request(
                method=method.upper(),
                url=url,
                headers=base_headers,
                params=params,
                json=json if files is None else None,
                files=files,
                timeout=self.timeout,
                stream=stream
            )

        return _wrap_response(resp)

    def _normalize_url(self, path: str) -> str:
        """
        Ensure a valid absolute URL:
        - If 'path' already starts with http, return as-is.
        - Else, resolve relative to <base_url>.
          If it doesn't start with '/maestro/api', we prepend it.
        """
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        if not path.startswith("/api/v2"):
            path = "/api/v2" + path
        return f"{self.base_url}{path}"

    # -------------
    # Utilities
    # -------------

    @property
    def token(self) -> Optional[str]:
        """Return the cached token (if any)."""
        return self._token

    @property
    def organization(self) -> Optional[str]:
        """Return the cached organization (if any)."""
        return self._organization

    def set_organization(self, organization: str) -> None:
        """
        Manually override the cached organization header value.

        Useful in multi-workspace contexts if your login returns multiple orgs
        or you need to switch org context without re-authenticating.
        """
        with self._lock:
            self._organization = organization


# ------------------------
# Resource Helper Classes
# ------------------------

class _TasksAPI:
    """
    Tasks-related routes.
    Common patterns observed in Maestro CLI & docs:
      - GET /tasks
      - POST /tasks
      - GET /tasks/{taskId}
      - POST /tasks/{taskId}/cancel
      - POST /tasks/{taskId}/finish
      - POST /tasks/{taskId}/restart
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, **kwargs) -> MaestroResponse:
        """GET /tasks"""
        print(f"kwargs: {kwargs}")
        query_params = "&".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        return self._c.request_raw("GET", f"/api/v2/task?{query_params}")

    def create(self, automation_label: str, data: Dict[str, Any]) -> MaestroResponse:
        """POST /tasks"""
        payload = {"automationLabel": automation_label, "data": data}
        return self._c.request_raw("POST", "/api/v2/task", json=payload)

    def get(self, task_id: Union[str, int]) -> MaestroResponse:
        """GET /tasks/{taskId}"""
        return self._c.request_raw("GET", f"/api/v2/task/{task_id}")

    def cancel(self, task_id: Union[str, int]) -> MaestroResponse:
        """POST /tasks/{taskId}/cancel"""
        return self._c.request_raw("POST", f"/api/v2/task/{task_id}/cancel")

    def finish(self, task_id: Union[str, int], result: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """POST /tasks/{taskId}/finish"""
        return self._c.request_raw("POST", f"/api/v2/task/{task_id}/finish", json=result or {})

    def restart(self, task_id: Union[str, int]) -> MaestroResponse:
        """POST /tasks/{taskId}/restart"""
        return self._c.request_raw("POST", f"/api/v2/task/{task_id}/restart")


class _LogsAPI:
    """
    Logs-related routes.
      - POST /logs
      - GET /logs?...
      - GET /logs/{logId}
      - DELETE /logs/{logId}
      - GET /logs/{logId}/download
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def create(self, label: str, message: str, level: str = "INFO", **kwargs) -> MaestroResponse:
        """POST /logs"""
        payload = {"label": label, "message": message, "level": level}
        payload.update(kwargs)
        return self._c.request_raw("POST", "/maestro/api/logs", json=payload)

    def list(self, *, label: Optional[str] = None, page: int = 1, size: int = 50,
             extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /logs"""
        params = {"page": page, "size": size}
        if label:
            params["label"] = label
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/maestro/api/logs", params=params)

    def get(self, log_id: Union[str, int]) -> MaestroResponse:
        """GET /logs/{logId}"""
        return self._c.request_raw("GET", f"/maestro/api/logs/{log_id}")

    def delete(self, log_id: Union[str, int]) -> MaestroResponse:
        """DELETE /logs/{logId}"""
        return self._c.request_raw("DELETE", f"/maestro/api/logs/{log_id}")

    def download(self, log_id: Union[str, int]) -> MaestroResponse:
        """GET /logs/{logId}/download"""
        # stream not necessary unless huge; keeping simple.
        return self._c.request_raw("GET", f"/maestro/api/logs/{log_id}/download")


class _AutomationsAPI:
    """
    Automations-related routes.
      - GET /automations
      - GET /automations/{id}
      - (Sometimes create/update via bots upload or CI/CD – kept generic)
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, *, label: Optional[str] = None, page: int = 1, size: int = 50,
             extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /automations"""
        params = {"page": page, "size": size}
        if label:
            params["label"] = label
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/api/v2/activity", params=params)

    def get(self, automation_label:str) -> MaestroResponse:
        """GET /automations/{id}"""
        return self._c.request_raw("GET", f"/api/v2/activity/{automation_label}")


class _BotsAPI:
    """
    Bots-related routes.
      - GET /bots
      - GET /bots/{botId}
      - POST /bots
      - PUT /bots/{botId}
      - Optional routes: /bots/{botId}/release etc., depending on workspace features
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, page: int = 1, size: int = 50, extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /bots"""
        params = {"page": page, "size": size}
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/api/v2/bot", params=params)

    def get(self, bot_id:str, bot_version:str) -> MaestroResponse:
        """GET /bots/{botId}"""
        return self._c.request_raw("GET", f"/api/v2/bot/{bot_id}/version/{bot_version}")

    def create(self, *, label: str, repository: Optional[str] = None, **kwargs) -> MaestroResponse:
        """POST /bots"""
        payload = {"label": label}
        if repository:
            payload["repository"] = repository
        payload.update(kwargs)
        return self._c.request_raw("POST", "/api/v2/bot", json=payload)

    def update(self, bot_id: Union[str, int], **fields) -> MaestroResponse:
        """PUT /bots/{botId}"""
        return self._c.request_raw("PUT", f"/api/v2/bot/{bot_id}", json=fields)

    def release(self, bot_id: Union[str, int], **fields) -> MaestroResponse:
        """POST /bots/{botId}/release (if supported)"""
        return self._c.request_raw("POST", f"/api/v2/bot/{bot_id}/release", json=fields)


class _RunnersAPI:
    """
    Runners-related routes.
      - GET /runners
      - GET /runners/{runnerId}
      - (actions may exist such as attach/release via Session Manager; keep generic endpoints)
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, page: int = 1, size: int = 50, extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /runners"""
        params = {"page": page, "size": size}
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/maestro/api/runners", params=params)

    def get(self, runner_id: Union[str, int]) -> MaestroResponse:
        """GET /runners/{runnerId}"""
        return self._c.request_raw("GET", f"/maestro/api/runners/{runner_id}")


class _CredentialsAPI:
    """
    Credentials-related routes.
      - GET /credentials
      - GET /credentials/{id}
      - POST /credentials
      - PUT /credentials/{id}
      - DELETE /credentials/{id}
      - GET /credentials/{label}/{key}  (common pattern for key retrieval)
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, page: int = 1, size: int = 50, extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /credentials"""
        params = {"page": page, "size": size}
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/maestro/api/credentials", params=params)

    def get(self, credential_id: Union[str, int]) -> MaestroResponse:
        """GET /credentials/{id}"""
        return self._c.request_raw("GET", f"/maestro/api/credentials/{credential_id}")

    def create(self, label: str, values: Dict[str, Any], **kwargs) -> MaestroResponse:
        """POST /credentials"""
        payload = {"label": label, "values": values}
        payload.update(kwargs)
        return self._c.request_raw("POST", "/maestro/api/credentials", json=payload)

    def update(self, credential_id: Union[str, int], **fields) -> MaestroResponse:
        """PUT /credentials/{id}"""
        return self._c.request_raw("PUT", f"/maestro/api/credentials/{credential_id}", json=fields)

    def delete(self, credential_id: Union[str, int]) -> MaestroResponse:
        """DELETE /credentials/{id}"""
        return self._c.request_raw("DELETE", f"/maestro/api/credentials/{credential_id}")

    def get_key(self, label: str, key: str) -> MaestroResponse:
        """GET /credentials/{label}/{key}"""
        return self._c.request_raw("GET", f"/maestro/api/credentials/{label}/{key}")


class _DatapoolsAPI:
    """
    Datapool-related routes.
      - GET /datapools
      - GET /datapools/{label}/items
      - POST /datapools/{label}/items
      - PUT /datapools/{label}/items/{itemId}
      - DELETE /datapools/{label}/items/{itemId}
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, page: int = 1, size: int = 50, extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /datapools"""
        params = {"page": page, "size": size}
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/maestro/api/datapools", params=params)

    def items(self, label: str, page: int = 1, size: int = 100,
              extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /datapools/{label}/items"""
        params = {"page": page, "size": size}
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", f"/maestro/api/datapools/{label}/items", params=params)

    def add_item(self, label: str, item: Dict[str, Any]) -> MaestroResponse:
        """POST /datapools/{label}/items"""
        return self._c.request_raw("POST", f"/maestro/api/datapools/{label}/items", json=item)

    def update_item(self, label: str, item_id: Union[str, int], fields: Dict[str, Any]) -> MaestroResponse:
        """PUT /datapools/{label}/items/{itemId}"""
        return self._c.request_raw("PUT", f"/maestro/api/datapools/{label}/items/{item_id}", json=fields)

    def delete_item(self, label: str, item_id: Union[str, int]) -> MaestroResponse:
        """DELETE /datapools/{label}/items/{itemId}"""
        return self._c.request_raw("DELETE", f"/maestro/api/datapools/{label}/items/{item_id}")


class _ResultFilesAPI:
    """
    Result Files-related routes.
      - GET /artifacts
      - GET /artifacts/{artifactId}
      - GET /artifacts/{artifactId}/download
      - POST /artifacts (multipart/form-data)
      - DELETE /artifacts/{artifactId}
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, page: int = 1, size: int = 50, extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /artifacts"""
        params = {"page": page, "size": size}
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/maestro/api/artifacts", params=params)

    def get(self, artifact_id: Union[str, int]) -> MaestroResponse:
        """GET /artifacts/{artifactId}"""
        return self._c.request_raw("GET", f"/maestro/api/artifacts/{artifact_id}")

    def download(self, artifact_id: Union[str, int]) -> MaestroResponse:
        """GET /artifacts/{artifactId}/download"""
        return self._c.request_raw("GET", f"/maestro/api/artifacts/{artifact_id}/download")

    def upload(self, *, file_field: str = "file", file_tuple: Iterable = None, **meta) -> MaestroResponse:
        """
        POST /artifacts (multipart). 
        Args:
            file_field: Field name expected by API (commonly "file").
            file_tuple: A (filename, fileobj, content_type) tuple as expected by requests.
            meta: Extra form fields to be sent as JSON part or text fields.
        """
        files = {file_field: file_tuple} if file_tuple else None
        # Separate JSONable fields in a small trick:
        headers = {"Content-Type": None}  # Let requests set multipart boundary.
        return self._c.request_raw("POST", "/maestro/api/artifacts", files=files, headers=headers, json=meta)

    def delete(self, artifact_id: Union[str, int]) -> MaestroResponse:
        """DELETE /artifacts/{artifactId}"""
        return self._c.request_raw("DELETE", f"/maestro/api/artifacts/{artifact_id}")


class _ErrorsAPI:
    """
    Errors-related routes.
      - GET /errors
      - GET /errors/{errorId}
      - DELETE /errors/{errorId}
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, page: int = 1, size: int = 50, extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /errors"""
        params = {"page": page, "size": size}
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/maestro/api/errors", params=params)

    def get(self, error_id: Union[str, int]) -> MaestroResponse:
        """GET /errors/{errorId}"""
        return self._c.request_raw("GET", f"/maestro/api/errors/{error_id}")

    def delete(self, error_id: Union[str, int]) -> MaestroResponse:
        """DELETE /errors/{errorId}"""
        return self._c.request_raw("DELETE", f"/maestro/api/errors/{error_id}")


class _SchedulesAPI:
    """
    Schedules-related routes.
      - GET /schedules
      - GET /schedules/{id}
      - POST /schedules
      - PUT /schedules/{id}
      - DELETE /schedules/{id}
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, page: int = 1, size: int = 50, extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /schedules"""
        params = {"page": page, "size": size}
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/maestro/api/schedules", params=params)

    def get(self, schedule_id: Union[str, int]) -> MaestroResponse:
        """GET /schedules/{id}"""
        return self._c.request_raw("GET", f"/maestro/api/schedules/{schedule_id}")

    def create(self, **fields) -> MaestroResponse:
        """POST /schedules"""
        return self._c.request_raw("POST", "/maestro/api/schedules", json=fields)

    def update(self, schedule_id: Union[str, int], **fields) -> MaestroResponse:
        """PUT /schedules/{id}"""
        return self._c.request_raw("PUT", f"/maestro/api/schedules/{schedule_id}", json=fields)

    def delete(self, schedule_id: Union[str, int]) -> MaestroResponse:
        """DELETE /schedules/{id}"""
        return self._c.request_raw("DELETE", f"/maestro/api/schedules/{schedule_id}")


class _WorkspacesAPI:
    """
    Workspaces/Organization routes (read-only in most public cases).
      - GET /workspaces
      - GET /workspaces/{id}
    """

    def __init__(self, client: MaestroClient):
        self._c = client

    def list(self, page: int = 1, size: int = 50, extra: Optional[Dict[str, Any]] = None) -> MaestroResponse:
        """GET /workspaces"""
        params = {"page": page, "size": size}
        if extra:
            params.update(extra)
        return self._c.request_raw("GET", "/maestro/api/workspaces", params=params)

    def get(self, workspace_id: Union[str, int]) -> MaestroResponse:
        """GET /workspaces/{id}"""
        return self._c.request_raw("GET", f"/maestro/api/workspaces/{workspace_id}")


# ----------------
# Helper functions
# ----------------

def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    """Try parsing JSON; return empty dict on failure."""
    try:
        return resp.json()
    except Exception:
        return {}

def _wrap_response(resp: requests.Response) -> MaestroResponse:
    """Build a MaestroResponse from a requests.Response."""
    data: Any
    ctype = resp.headers.get("Content-Type", "")
    if "application/json" in ctype:
        data = _safe_json(resp)
    else:
        # If content is text, get .text; else raw bytes
        try:
            if "text/" in ctype:
                data = resp.text
            else:
                data = resp.content
        except Exception:
            data = resp.content

    return MaestroResponse(
        ok=resp.ok,
        status_code=resp.status_code,
        url=str(resp.url),
        headers=dict(resp.headers),
        data=data,
        raw=resp
    )
