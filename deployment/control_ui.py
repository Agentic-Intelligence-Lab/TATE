#!/usr/bin/env python3
"""Loopback-only TATE stack-cube control panel for the ARX host."""

from __future__ import annotations

import asyncio
import math
import os
import pty
import secrets
import signal
import subprocess
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlopen

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(APP_ROOT))

from deployment.constants import CAMERA_SERIALS, DEFAULT_CHECKPOINT_DIR, PREVIEW_DIR, checkpoint_status  # noqa: E402


TOKEN = secrets.token_urlsafe(24)
WRAPPER = APP_ROOT / "deployment" / "run_policy.sh"
RESET_WRAPPER = Path("/home/qijun/lyt/fold/src/deployment/run_reset_arx_home.sh")
POLICY_PYTHON = Path(
    os.environ.get("TATE_POLICY_PYTHON", "/home/qijun/models/TATE/openpi/.venv/bin/python")
)
POLICY_READY_MARKER = POLICY_PYTHON.parent.parent / ".tate_inference_ready"
IDLE_PREVIEW_DIR = Path("/tmp/tate_arx_camera_idle")


def selected_checkpoint(raw_path: str | None) -> Path:
    value = str(DEFAULT_CHECKPOINT_DIR) if raw_path is None else raw_path.strip()
    if not value or len(value) > 4096 or "\0" in value:
        raise HTTPException(400, "请输入有效的 checkpoint 路径")
    path = Path(value)
    if not path.is_absolute():
        raise HTTPException(400, "checkpoint 路径必须是 ARX 主机上的绝对路径")
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(400, f"checkpoint 路径不可用: {exc}") from exc


def model_fingerprint(checkpoint_dir: Path) -> tuple[str, int, int] | None:
    model = checkpoint_dir / "model.safetensors"
    try:
        stat = model.stat()
    except OSError:
        return None
    return str(checkpoint_dir), stat.st_size, stat.st_mtime_ns


def fold_task_active() -> bool:
    markers = (b"fold_box_policy_robot.py", b"run_fold_box_policy.sh", b"run_reset_arx_home.sh")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if any(marker in command for marker in markers):
            return True
    return False


class StartRequest(BaseModel):
    mode: str
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_DIR)
    fps: float = Field(default=5.0, ge=1, le=10)
    n_action_steps: int = Field(default=1, ge=1, le=50)
    start_pose: bool = False
    tcp_offset_m: tuple[float, float, float] = (0.15, 0.0, 0.0)


class ProcessManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.process: subprocess.Popen[bytes] | None = None
        self.master_fd: int | None = None
        self.mode = "idle"
        self.last_result = "尚未运行"
        self.started_at: float | None = None
        self.logs: deque[str] = deque(maxlen=300)
        self.smoke_fingerprint: tuple[str, int, int] | None = None
        self.active_checkpoint: Path | None = None

    def _append(self, value: str) -> None:
        with self.lock:
            for line in value.replace("\r", "").splitlines():
                if not line.strip():
                    continue
                clean = line[-500:]
                self.logs.append(clean)
                if "READY_FOR_RUN" in clean and self.mode == "preparing":
                    self.mode = "awaiting_confirmation"
                    self.last_result = "首个目标已通过检查，等待 RUN 确认"
                elif "READY_FOR_START_POSE" in clean and self.mode == "preparing":
                    self.mode = "awaiting_start_pose"
                    self.last_result = "已检查真实数据第 60 帧目标，等待 MOVE 确认"
                elif clean.startswith("MOVE accepted"):
                    self.mode = "moving_start_pose"
                    self.last_result = "正缓慢移动到真实数据第 60 帧"
                elif clean.startswith("START_POSE_REACHED"):
                    self.mode = "preparing"
                    self.last_result = "起点已到达，正在采集画面并推理首个目标"
                elif clean.startswith("RUN accepted"):
                    self.mode = "testing"
                    self.last_result = "真机测试运行中"
                elif clean == "PAUSED":
                    self.mode = "paused"
                    self.last_result = "右臂保持当前位置"
                elif clean == "RESUMED":
                    self.mode = "testing"
                    self.last_result = "已继续推理"

    def _reader(self, process: subprocess.Popen[bytes], fd: int, started_mode: str, checkpoint_dir: Path | None) -> None:
        try:
            while True:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                self._append(data.decode("utf-8", errors="replace"))
        finally:
            code = process.wait()
            try:
                os.close(fd)
            except OSError:
                pass
            with self.lock:
                if self.process is process:
                    if code == 0 and started_mode == "smoking" and checkpoint_dir is not None and any("SMOKE_INFERENCE_OK" in x for x in self.logs):
                        self.smoke_fingerprint = model_fingerprint(checkpoint_dir)
                    if code == 0:
                        self.last_result = "检查或测试已完成"
                    elif self.mode == "stopping":
                        self.last_result = "任务已停止，保护模式与原服务正在恢复"
                    else:
                        self.last_result = f"进程退出，状态码 {code}"
                    self.mode = "idle"
                    self.process = None
                    self.master_fd = None
                    self.active_checkpoint = None

    def launch(self, command: list[str], mode: str, checkpoint_dir: Path | None = None, confirmation: str | None = None) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise HTTPException(409, "已有 TATE 任务在运行")
            master_fd, slave_fd = pty.openpty()
            child_env = os.environ.copy()
            if checkpoint_dir is not None:
                child_env["TATE_CHECKPOINT_DIR"] = str(checkpoint_dir)
            try:
                process = subprocess.Popen(
                    command,
                    env=child_env,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                os.close(slave_fd)
            self.process = process
            self.master_fd = master_fd
            self.active_checkpoint = checkpoint_dir
            self.mode = mode
            self.started_at = time.time()
            self.last_result = "启动中"
            self.logs.clear()
            self.logs.append("$ " + (f"TATE_CHECKPOINT_DIR={checkpoint_dir} " if checkpoint_dir else "") + " ".join(command))
            threading.Thread(target=self._reader, args=(process, master_fd, mode, checkpoint_dir), daemon=True).start()
            if confirmation is not None:
                os.write(master_fd, (confirmation + "\n").encode("utf-8"))

    def confirm(self) -> None:
        with self.lock:
            if self.mode != "awaiting_confirmation" or self.master_fd is None:
                raise HTTPException(409, "当前没有等待 RUN 确认的目标")
            os.write(self.master_fd, b"RUN\n")
            self.mode = "preparing"
            self.last_result = "已发送 RUN，等待新一轮观测和检查"

    def confirm_start_pose(self) -> None:
        with self.lock:
            if self.mode != "awaiting_start_pose" or self.master_fd is None:
                raise HTTPException(409, "当前没有等待 MOVE 确认的起点")
            os.write(self.master_fd, b"MOVE\n")
            self.mode = "moving_start_pose"
            self.last_result = "已发送 MOVE，等待右臂缓慢到达起点"

    @staticmethod
    def _robot_pid(root_pid: int) -> int | None:
        pending = [root_pid]
        while pending:
            pid = pending.pop()
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
                children = Path(f"/proc/{pid}/task/{pid}/children").read_text()
            except OSError:
                continue
            if b"deployment/robot_runner.py" in cmdline:
                return pid
            pending.extend(int(value) for value in children.split())
        return None

    def pause(self) -> None:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None or self.mode not in {"testing", "paused"}:
                raise HTTPException(409, "当前没有可暂停的真机测试")
            robot_pid = self._robot_pid(process.pid)
            if robot_pid is None:
                raise HTTPException(409, "机器人进程尚未就绪")
            os.kill(robot_pid, signal.SIGUSR1)
            self.last_result = "已请求切换暂停状态"

    def stop(self) -> None:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                self.last_result = "当前没有运行中的任务"
                return
            self.mode = "stopping"
            self.last_result = "正在停止并恢复原控制服务"
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def snapshot(self, checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> dict:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            return {
                "mode": self.mode,
                "running": running,
                "pid": self.process.pid if running else None,
                "started_at": self.started_at,
                "active_checkpoint": str(self.active_checkpoint) if self.active_checkpoint else None,
                "result": self.last_result,
                "smoke_ready": self.smoke_fingerprint is not None and self.smoke_fingerprint == model_fingerprint(checkpoint_dir),
                "logs": list(self.logs),
            }


class IdleCameraRelay:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        IDLE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        for role in CAMERA_SERIALS:
            thread = threading.Thread(target=self._relay, args=(role,), daemon=True)
            self.threads.append(thread)
            thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=1.5)

    def _relay(self, role: str) -> None:
        url = f"http://127.0.0.1:8090/stream/{role}"
        while not self.stop_event.is_set():
            try:
                with urlopen(url, timeout=3) as response:
                    buffer = b""
                    while not self.stop_event.is_set():
                        block = response.read(8192)
                        if not block:
                            break
                        buffer += block
                        while True:
                            start = buffer.find(b"\xff\xd8")
                            end = buffer.find(b"\xff\xd9", start + 2)
                            if start < 0 or end < 0:
                                buffer = buffer[-1_000_000:]
                                break
                            jpg = buffer[start : end + 2]
                            buffer = buffer[end + 2 :]
                            temporary = IDLE_PREVIEW_DIR / f".{role}.tmp"
                            temporary.write_bytes(jpg)
                            os.replace(temporary, IDLE_PREVIEW_DIR / f"{role}.jpg")
            except Exception:
                self.stop_event.wait(0.8)


def latest_camera_frame(role: str) -> tuple[bytes, int] | None:
    if role not in CAMERA_SERIALS:
        return None
    candidates = [PREVIEW_DIR / f"{role}.jpg", IDLE_PREVIEW_DIR / f"{role}.jpg"]
    existing = []
    for path in candidates:
        try:
            if time.time_ns() - path.stat().st_mtime_ns < 5_000_000_000:
                existing.append(path)
        except OSError:
            pass
    if not existing:
        return None
    path = max(existing, key=lambda item: item.stat().st_mtime_ns)
    try:
        return path.read_bytes(), path.stat().st_mtime_ns
    except OSError:
        return None


def camera_stream(role: str):
    last_stamp = 0
    while True:
        frame = latest_camera_frame(role)
        if frame is not None and frame[1] != last_stamp:
            jpg, last_stamp = frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.08)


manager = ProcessManager()
camera_relay = IdleCameraRelay()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    camera_relay.start()
    try:
        yield
    finally:
        manager.stop()
        camera_relay.stop()
        for _ in range(50):
            if not manager.snapshot()["running"]:
                break
            await asyncio.sleep(0.1)


app = FastAPI(title="TATE ARX EEF 控制台", docs_url=None, redoc_url=None, lifespan=lifespan)


def authorize(token: str | None) -> None:
    if not token or not secrets.compare_digest(token, TOKEN):
        raise HTTPException(403, "控制令牌无效")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_ROOT / "deployment" / "control_panel.html").read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)


@app.get("/api/checkpoint")
def checkpoint(checkpoint_path: str | None = None) -> dict:
    selected = selected_checkpoint(checkpoint_path)
    partial = selected / "model.safetensors.partial"
    return {
        **checkpoint_status(selected),
        "policy_python_present": POLICY_PYTHON.is_file() and POLICY_READY_MARKER.is_file(),
        "partial_bytes": partial.stat().st_size if partial.is_file() else None,
    }


@app.get("/api/status")
def status(checkpoint_path: str | None = None) -> dict:
    return manager.snapshot(selected_checkpoint(checkpoint_path))


@app.get("/camera/{role}")
def camera(role: str) -> StreamingResponse:
    if role not in CAMERA_SERIALS:
        raise HTTPException(404, "未知摄像头")
    return StreamingResponse(camera_stream(role), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/start")
def start(req: StartRequest, x_control_token: str | None = Header(default=None)) -> dict:
    authorize(x_control_token)
    selected = selected_checkpoint(req.checkpoint_path)
    if req.mode not in {"check", "smoke", "execute"}:
        raise HTTPException(400, "未知运行模式")
    if not checkpoint_status(selected)["ready"]:
        raise HTTPException(409, "模型权重、归一化文件或 tokenizer 尚未齐全")
    if not POLICY_PYTHON.is_file() or not POLICY_READY_MARKER.is_file():
        raise HTTPException(409, "OpenPI 推理环境尚未安装")
    command = [str(WRAPPER), {"check": "--check", "smoke": "--smoke", "execute": "--execute"}[req.mode]]
    mode = {"check": "checking", "smoke": "smoking", "execute": "preparing"}[req.mode]
    if req.mode == "execute":
        command += ["--fps", str(req.fps), "--n-action-steps", str(req.n_action_steps)]
        tcp_offset = tuple(float(value) for value in req.tcp_offset_m)
        if not all(math.isfinite(value) for value in tcp_offset) or math.sqrt(sum(value * value for value in tcp_offset)) > 0.30:
            raise HTTPException(400, "TCP offset 必须是有限的三个数，且模长不超过 0.30 m")
        command += ["--tcp-offset-m", *(str(value) for value in tcp_offset)]
        if req.start_pose:
            command.append("--start-pose")
    manager.launch(command, mode, checkpoint_dir=selected)
    return {"ok": True, "mode": mode}


@app.post("/api/confirm")
def confirm(x_control_token: str | None = Header(default=None)) -> dict:
    authorize(x_control_token)
    manager.confirm()
    return {"ok": True}


@app.post("/api/confirm-start-pose")
def confirm_start_pose(x_control_token: str | None = Header(default=None)) -> dict:
    authorize(x_control_token)
    manager.confirm_start_pose()
    return {"ok": True}


@app.post("/api/pause")
def pause(x_control_token: str | None = Header(default=None)) -> dict:
    authorize(x_control_token)
    manager.pause()
    return {"ok": True}


@app.post("/api/stop")
def stop(x_control_token: str | None = Header(default=None)) -> dict:
    authorize(x_control_token)
    manager.stop()
    return {"ok": True}


@app.post("/api/reset")
def reset(x_control_token: str | None = Header(default=None)) -> dict:
    authorize(x_control_token)
    if not RESET_WRAPPER.is_file():
        raise HTTPException(409, "ARX 复位脚本不存在")
    if fold_task_active():
        raise HTTPException(409, "Fold Box 控制或复位任务正在运行")
    manager.launch([str(RESET_WRAPPER)], "resetting", confirmation="RESET")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8089, log_level="info", timeout_graceful_shutdown=5)
