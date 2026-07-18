# -*- coding: utf-8 -*-
"""Local browser UI for MFR multi-view inference visualization."""

import json
import os
import tempfile
import threading
import traceback
import uuid
import cgi
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
STATIC_DIR = CURRENT_DIR / "web_static"

import sys

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from multiview_runner import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_M2F_ROOT,
    DEFAULT_SINGLEVIEW_CONFIG,
    DEFAULT_SINGLEVIEW_WEIGHTS,
    DEFAULT_VIDEO_CONFIG,
    DEFAULT_WEIGHTS,
    load_step_faces,
    mesh_to_payload,
    run_detectron2_multiview,
    run_detectron2_singleview,
    summarize_labels,
)


STATE = {"jobs": {}, "lock": threading.Lock()}
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "m2f_step_uploads")


def load_step_job(step_path):
    job = make_job(step_path)
    try:
        append_log(job, f"加载 STEP: {step_path}")
        step_data = load_step_faces(step_path)
        if not step_data or not step_data["faces"]:
            raise RuntimeError("STEP 加载失败或未提取到面")
        job["step_data"] = step_data
        job["mesh"] = mesh_to_payload(step_data)
        job["mesh_version"] += 1
        job["summary"] = []
        job["partial_frame"] = 0
        job["status"] = "ready"
        append_log(job, f"加载完成: {len(step_data['faces'])} 个面")
        return job, None
    except Exception:
        job["status"] = "failed"
        job["error"] = traceback.format_exc()
        return job, job["error"]


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def make_job(step_path):
    job_id = uuid.uuid4().hex[:12]
    output_dir = os.path.join(tempfile.gettempdir(), f"m2f_web_{job_id}")
    os.makedirs(output_dir, exist_ok=True)
    job = {
        "id": job_id,
        "step_path": step_path,
        "output_dir": output_dir,
        "status": "loading",
        "logs": [],
        "log_path": os.path.join(output_dir, "run.log"),
        "error": None,
        "step_data": None,
        "mesh": None,
        "mesh_version": 0,
        "summary": [],
        "partial_frame": 0,
        "result": None,
    }
    with STATE["lock"]:
        STATE["jobs"][job_id] = job
    return job


def append_log(job, message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with STATE["lock"]:
        job["logs"].append(line)
        job["logs"] = job["logs"][-300:]
        log_path = job.get("log_path")
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(line + "\n")


def public_job(job):
    return {
        "id": job["id"],
        "step_path": job["step_path"],
        "output_dir": job["output_dir"],
        "log_path": job["log_path"],
        "status": job["status"],
        "logs": job["logs"],
        "error": job["error"],
        "mesh_version": job["mesh_version"],
        "summary": job["summary"],
        "partial_frame": job["partial_frame"],
        "result": job["result"],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "MFRWebInference/0.1"

    def do_HEAD(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self._send_file_headers(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            target = (STATIC_DIR / rel).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return
            content_type = "application/javascript" if target.suffix == ".js" else "text/css"
            self._send_file_headers(target, content_type)
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            target = (STATIC_DIR / rel).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = "application/javascript" if target.suffix == ".js" else "text/css"
            self._send_file(target, content_type)
            return
        if path == "/api/defaults":
            self._send_json(
                {
                    "m2f_root": DEFAULT_M2F_ROOT,
                    "config_path": DEFAULT_CONFIG,
                    "weights_path": DEFAULT_WEIGHTS,
                    "mode": "multiview",
                    "modes": {
                        "multiview": {
                            "label": "视频多视角",
                            "config_path": DEFAULT_VIDEO_CONFIG,
                            "weights_path": DEFAULT_WEIGHTS,
                        },
                        "singleview": {
                            "label": "单图投票",
                            "config_path": DEFAULT_SINGLEVIEW_CONFIG,
                            "weights_path": DEFAULT_SINGLEVIEW_WEIGHTS,
                        },
                    },
                    "device": "cuda",
                    "score_threshold": 0.5,
                    "min_ratio": 0.5,
                    "min_face_area": 10,
                }
            )
            return
        if path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3:
                job = self._get_job(parts[2])
                self._send_json(public_job(job) if job else {"error": "job not found"}, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND)
                return
            if len(parts) == 4 and parts[3] == "mesh":
                job = self._get_job(parts[2])
                if not job or not job.get("mesh"):
                    self._send_json({"error": "mesh not ready"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json(job["mesh"])
                return
        if path.startswith("/outputs/"):
            rel = path[len("/outputs/") :]
            job_id, _, rest = rel.partition("/")
            job = self._get_job(job_id)
            if not job or not rest:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            target = Path(job["output_dir"], rest).resolve()
            if not str(target).startswith(str(Path(job["output_dir"]).resolve())) or not target.exists():
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_file(target, "application/octet-stream")
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/load-step":
            payload = self._read_json()
            step_path = os.path.abspath(os.path.expanduser(payload.get("step_path", "").strip()))
            if not os.path.exists(step_path):
                self._send_json({"error": f"STEP 文件不存在: {step_path}"}, HTTPStatus.BAD_REQUEST)
                return
            job, error = load_step_job(step_path)
            status = HTTPStatus.INTERNAL_SERVER_ERROR if error else HTTPStatus.OK
            payload = {"job": public_job(job), "mesh": job.get("mesh")}
            if error:
                payload["error"] = error
            self._send_json(payload, status)
            return
        if parsed.path == "/api/upload-step":
            try:
                step_path = self._save_uploaded_step()
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            job, error = load_step_job(step_path)
            status = HTTPStatus.INTERNAL_SERVER_ERROR if error else HTTPStatus.OK
            payload = {"job": public_job(job), "mesh": job.get("mesh")}
            if error:
                payload["error"] = error
            self._send_json(payload, status)
            return
        if parsed.path == "/api/run":
            payload = self._read_json()
            job = self._get_job(payload.get("job_id", ""))
            if not job:
                self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            if job["status"] == "running":
                self._send_json({"job": public_job(job)})
                return
            for key in ("weights_path", "config_path", "m2f_root"):
                if not os.path.exists(os.path.expanduser(payload.get(key, ""))):
                    self._send_json({"error": f"路径不存在: {key}={payload.get(key, '')}"}, HTTPStatus.BAD_REQUEST)
                    return
            thread = threading.Thread(target=self._run_job, args=(job, payload), daemon=True)
            thread.start()
            self._send_json({"job": public_job(job)})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _run_job(self, job, payload):
        with STATE["lock"]:
            job["status"] = "running"
            job["error"] = None
            job["result"] = None
        try:
            output_dir = payload.get("output_dir") or job["output_dir"]
            output_dir = os.path.abspath(os.path.expanduser(output_dir))
            os.makedirs(output_dir, exist_ok=True)
            job["output_dir"] = output_dir
            job["log_path"] = os.path.join(output_dir, "run.log")
            append_log(job, f"日志文件: {job['log_path']}")
            mode = payload.get("mode") or "multiview"
            append_log(job, f"推理模式: {mode}")
            runner = run_detectron2_singleview if mode == "singleview" else run_detectron2_multiview
            result = runner(
                job["step_data"],
                os.path.abspath(os.path.expanduser(payload["m2f_root"])),
                os.path.abspath(os.path.expanduser(payload["config_path"])),
                os.path.abspath(os.path.expanduser(payload["weights_path"])),
                output_dir,
                payload.get("device") or "cuda",
                float(payload.get("score_threshold", 0.5)),
                float(payload.get("min_ratio", 0.5)),
                int(payload.get("min_face_area", 10)),
                progress=lambda message: append_log(job, message),
            )
            self._update_final_mesh(job, result["face_labels"])
            colored_faces = sum(1 for item in result["face_labels"].values() if int(item.get("class_id", -1)) >= 0)
            append_log(job, f"最终综合结果: {colored_faces}/{len(result['face_labels'])} 个面为特征类别")
            job["result"] = {
                "output_dir": output_dir,
                "log_path": job["log_path"],
                "face_labels": result["face_labels"],
                "summary": job["summary"],
                "label_url": f"/outputs/{job['id']}/face_labels.json",
                "encoded_url": f"/outputs/{job['id']}/encoded_views/000001.png",
                "frame_results_dir": os.path.join(output_dir, "frame_results"),
                "raw_predictions": result["raw_predictions"],
            }
            append_log(job, f"完成。输出目录: {output_dir}")
            job["status"] = "done"
        except Exception:
            job["status"] = "failed"
            job["error"] = traceback.format_exc()
            append_log(job, job["error"])

    def _update_final_mesh(self, job, face_labels):
        mesh = mesh_to_payload(job["step_data"], face_labels)
        summary = summarize_labels(face_labels)
        with STATE["lock"]:
            job["mesh"] = mesh
            job["mesh_version"] += 1
            job["summary"] = summary
            job["partial_frame"] = 14

    def _get_job(self, job_id):
        with STATE["lock"]:
            return STATE["jobs"].get(job_id)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(body or "{}")

    def _save_uploaded_step(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
        })
        file_item = form["step_file"] if "step_file" in form else None
        if file_item is None or not file_item.filename:
            raise ValueError("没有收到 STEP 文件")
        suffix = Path(file_item.filename).suffix.lower()
        if suffix not in (".step", ".stp"):
            raise ValueError("请选择 .step 或 .stp 文件")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(file_item.filename).name)
        target = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:12]}_{safe_name}")
        with open(target, "wb") as output:
            while True:
                chunk = file_item.file.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        return target

    def _send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path, content_type):
        data = Path(path).read_bytes()
        self._send_file_headers(path, content_type, len(data))
        self.wfile.write(data)

    def _send_file_headers(self, path, content_type, content_length=None):
        if content_length is None:
            content_length = Path(path).stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run MFR multi-view browser inference UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Web UI: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
