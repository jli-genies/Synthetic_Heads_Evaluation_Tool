"""HTTP client for GenieSAM local Docker (`use_local: true`)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def host_to_container_path(host_path: Path, host_root: Path, container_root: str) -> str:
    """Map a host path under host_root to the equivalent container path."""
    host_path = host_path.resolve()
    host_root = host_root.resolve()
    try:
        relative = host_path.relative_to(host_root)
    except ValueError as error:
        raise ValueError(
            f"Path {host_path} is outside mounted host root {host_root}"
        ) from error
    # Always use forward slashes for Linux container paths.
    return f"{container_root.rstrip('/')}/{relative.as_posix()}"


def build_local_payload(
    *,
    image_host: Path,
    output_dir_host: Path,
    host_renders_root: Path,
    container_renders_root: str,
    categories: list[str] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a GenieSAM /invocations body for local Docker mounts."""
    image_container = host_to_container_path(
        image_host, host_renders_root, container_renders_root
    )
    output_container = host_to_container_path(
        output_dir_host, host_renders_root, container_renders_root
    )
    if not output_container.endswith("/"):
        output_container += "/"

    payload: dict[str, Any] = {
        "use_local": True,
        "image_path": image_container,
        "output_prefix": output_container,
        "request_id": request_id or f"eval-{uuid.uuid4().hex[:12]}",
    }
    if categories:
        payload["categories"] = list(categories)
    return payload


def invoke_geniesam(
    endpoint_url: str,
    payload: dict[str, Any],
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """POST JSON to GenieSAM and return the parsed response body."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"HTTP {error.code} from GenieSAM: {detail[-2000:]}"
            ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach GenieSAM at {endpoint_url}: {error.reason}. "
            "Is the Docker container running and port 8080 published?"
        ) from error


def expected_local_output(output_dir_host: Path) -> Path:
    """GenieSAM local mode always writes segmentation.json under output_prefix."""
    return output_dir_host / "segmentation.json"
