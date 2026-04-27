"""SSH utility module for See-Shell OCT visualization app.

Stdlib-only SSH operations: parse config, list remote dirs, find files, download.
"""

import os
import shlex
import subprocess
from pathlib import Path


def parse_ssh_config() -> list[dict]:
    """Parse ``~/.ssh/config``, return list of host entries.

    Each entry: ``host``, ``hostname``, ``user``, ``port``,
    ``identity_file``, ``proxy_jump``.
    Wildcard hosts (``*`` / ``?``) are skipped.
    """
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.is_file():
        return []

    hosts: list[dict] = []
    current: dict | None = None

    for raw in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        kw, val = parts[0].lower(), parts[1].strip()

        if kw == "host":
            if current is not None:
                hosts.append(current)
            if "*" in val or "?" in val:
                current = None
                continue
            current = dict(host=val, hostname=None, user=None, port=None,
                           identity_file=None, proxy_jump=None)
            continue

        if current is None:
            continue
        if kw == "hostname":
            current["hostname"] = val
        elif kw == "user":
            current["user"] = val
        elif kw == "port":
            try:
                current["port"] = int(val)
            except ValueError:
                pass
        elif kw == "identityfile":
            current["identity_file"] = os.path.expanduser(val)
        elif kw == "proxyjump":
            current["proxy_jump"] = val

    if current is not None:
        hosts.append(current)
    return hosts


def list_remote_dir(host: str, remote_path: str, timeout: int = 30) -> list[dict]:
    """List remote directory via ``ssh {host} ls -la {path}``.

    Returns list of dicts: ``name``, ``size``, ``is_dir``, ``modified``.
    """
    try:
        r = subprocess.run(
            ["ssh", host, "ls", "-la", shlex.quote(remote_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SSH list timed out after {timeout}s: {remote_path}") from exc

    if r.returncode != 0:
        raise RuntimeError(f"SSH list failed (exit {r.returncode}): {r.stderr.strip()}")

    entries: list[dict] = []
    for line in r.stdout.splitlines():
        parts = line.strip().split(None, 8)
        if len(parts) < 9:
            continue
        perm = parts[0]
        name = parts[8]
        if name in (".", "..") or not perm or perm[0] not in "-dlbcps":
            continue
        try:
            size = int(parts[4])
        except ValueError:
            size = 0
        entries.append(dict(
            name=name, size=size,
            is_dir=perm.startswith("d"),
            modified=f"{parts[5]} {parts[6]} {parts[7]}",
        ))
    return entries


def find_remote_files(host: str, remote_dir: str, ext: str = ".ply",
                      timeout: int = 60) -> list[str]:
    """Recursively find files matching *ext* under *remote_dir*.

    Returns absolute remote paths.
    """
    try:
        r = subprocess.run(
            ["ssh", host, "find", shlex.quote(remote_dir),
             "-type", "f", "-name", f"*{ext}"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Remote find timed out after {timeout}s") from exc

    if r.returncode != 0:
        raise RuntimeError(f"Remote find failed (exit {r.returncode}): {r.stderr.strip()}")

    return [p for p in r.stdout.strip().splitlines() if p]


def download_file(host: str, remote_path: str, local_path: str,
                  timeout: int = 600) -> str:
    """Download single file via ``scp -q -C``. Returns *local_path*."""
    try:
        r = subprocess.run(
            ["scp", "-q", "-C", f"{host}:{remote_path}", local_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SCP timed out after {timeout}s: {remote_path}") from exc

    if r.returncode != 0:
        raise RuntimeError(f"SCP failed (exit {r.returncode}): {r.stderr.strip()}")
    return local_path


def get_remote_file_size(host: str, remote_path: str) -> int:
    """Remote file size in bytes, or -1 on failure."""
    try:
        r = subprocess.run(
            ["ssh", host, "stat", "-c", "%s", shlex.quote(remote_path)],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            return int(r.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return -1


if __name__ == "__main__":
    hosts = parse_ssh_config()
    print(f"{len(hosts)} host(s): {[h['host'] for h in hosts]}")
