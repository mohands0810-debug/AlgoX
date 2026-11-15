"""AlgoX – Mars Colony Emergency Response
Flask web application: upload a CSV → solve → view + download JSON.
Run:  python app.py
Then open http://127.0.0.1:5000 in your browser.
"""
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from judge import judge as run_judge
from solve import Problem, advanced_greedy, best_greedy, anneal_worker, build_schedule

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

CAT_NAMES = ["STRUCTURAL", "MEDICAL", "LIFE_SUPPORT"]
CREW_OF = {0: "E", 1: "M", 2: "L"}


def _run_solve(job_id: str, csv_path: str, out_path: str, iters: int, seed: int):
    """Background thread: run the full solver pipeline and update job status."""
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "greedy"

        P = Problem(csv_path)
        init_qs, init_score = best_greedy(P)

        with _jobs_lock:
            _jobs[job_id].update({"status": "annealing", "greedy_score": round(init_score, 4)})

        # Active / drop split
        active = [[], [], []]
        drop = []
        for c in range(3):
            t = 0
            for i in init_qs[c]:
                s = P.arr[i] if P.arr[i] > t else t
                if s - P.arr[i] < P.win[i]:
                    active[c].append(i)
                    t = s + P.d3[c][i]
                else:
                    drop.append(i)

        # SA runs: sequential on Vercel to respect serverless sandbox, multiprocessing locally
        configs = [
            (csv_path, active, drop, iters, seed,     2.0, 0.01),
            (csv_path, active, drop, iters, seed + 1, 1.0, 0.01),
            (csv_path, active, drop, iters, seed + 2, 4.0, 0.005),
            (csv_path, active, drop, iters, seed + 3, 2.0, 0.003),
        ]
        if os.environ.get("VERCEL"):
            results = [anneal_worker(cfg) for cfg in configs[:2]]
        else:
            try:
                import multiprocessing as mp
                with mp.Pool(len(configs)) as pool:
                    results = pool.map(anneal_worker, configs)
            except Exception:
                results = [anneal_worker(cfg) for cfg in configs]

        run_scores = [round(r[0], 4) for r in results]
        best_score, (best_qs, best_dump) = max(results, key=lambda r: r[0])

        rows = build_schedule(P, best_qs, best_dump)
        schedule = {"schedule": rows}
        with open(out_path, "w") as f:
            json.dump(schedule, f)

        # Validate with judge
        judge_score, judge_msg = run_judge(csv_path, out_path)

        # Detailed row telemetry for timeline and table (first 500 rows)
        enriched_rows = []
        for r in rows[:500]:
            i = r["id"]
            cat_idx = P.cat[i]
            crew = r["crew"]
            is_match = (CREW_OF[cat_idx] == crew)
            wait = r["start"] - P.arr[i]
            window = P.win[i]
            pts = P.sev[i] * max(0.0, 1.0 - wait / window) if wait < window else 0.0
            enriched_rows.append({
                "id": i,
                "crew": crew,
                "start_time": r["start"],
                "end_time": r["end"],
                "duration": r["end"] - r["start"],
                "arrival": P.arr[i],
                "severity": P.sev[i],
                "cat": CAT_NAMES[cat_idx],
                "is_match": is_match,
                "score_contrib": round(pts, 3),
            })

        max_span = max((r["end"] for r in rows), default=1)
        total_possible = sum(P.sev[i] for i in P.ids)
        efficiency = round((judge_score / total_possible * 100), 1) if total_possible > 0 else 0.0

        rules_audit = [
            {
                "num": 1,
                "name": "Incident Completeness",
                "desc": f"All {len(P.ids)} incidents scheduled exactly once (no duplicates, no omissions)",
                "passed": (judge_msg == "valid"),
            },
            {
                "num": 2,
                "name": "Valid Crew Types",
                "desc": "All assignments use authorized crews: E (Structural), M (Medical), L (Life Support)",
                "passed": (judge_msg == "valid"),
            },
            {
                "num": 3,
                "name": "Temporal Causality",
                "desc": "Zero starts before arrival (start >= arrival strictly verified)",
                "passed": (judge_msg == "valid"),
            },
            {
                "num": 4,
                "name": "Duration Calculation",
                "desc": "end = start + dur (matching) or start + 2*dur (cross-crew)",
                "passed": (judge_msg == "valid"),
            },
            {
                "num": 5,
                "name": "Non-overlapping Execution",
                "desc": "Zero crew conflicts (subsequent start >= previous end)",
                "passed": (judge_msg == "valid"),
            },
            {
                "num": 6,
                "name": "Chronological Monotonicity",
                "desc": "Schedule strictly sorted by start time",
                "passed": (judge_msg == "valid"),
            },
            {
                "num": 7,
                "name": "Whole-Number Quantization",
                "desc": "All id, start, end fields are integer minutes",
                "passed": (judge_msg == "valid"),
            },
        ]

        with _jobs_lock:
            _jobs[job_id].update({
                "status": "done",
                "score": round(judge_score, 4),
                "seed": seed,
                "incidents": len(P.ids),
                "greedy_score": round(init_score, 4),
                "run_scores": run_scores,
                "annealed_score": round(best_score, 4),
                "judge_score": round(judge_score, 4),
                "judge_msg": judge_msg,
                "incident_count": len(P.ids),
                "active_count": sum(len(a) for a in best_qs),
                "drop_count": len(best_dump),
                "out_path": out_path,
                "schedule": rows,
                "schedule_preview": rows[:30],
                "enriched_rows": enriched_rows,
                "total_rows": len(rows),
                "max_span": max_span,
                "efficiency": efficiency,
                "rules_audit": rules_audit,
            })
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update({"status": "error", "error": str(exc)})


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import make_response
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/preview")
def preview_page():
    from flask import make_response
    p = BASE / "data" / "demo.json"
    demo_json = p.read_text(encoding="utf-8") if p.exists() else None
    s = BASE / "data" / "sample1.csv"
    sample1_csv = s.read_text(encoding="utf-8") if s.exists() else None
    resp = make_response(render_template("index.html", preview=True, demo_json=demo_json, sample1_csv=sample1_csv))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/solve", methods=["POST"])
def solve():
    iters = int(request.form.get("iters", 500_000))
    seed = int(request.form.get("seed", 2026))
    sample_name = request.form.get("sample", "").strip()

    iters = max(10_000, min(iters, 10_000_000))
    seed = max(0, min(seed, 2**31 - 1))

    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "input.csv")
    out_path = os.path.join(tmp_dir, "submission.json")

    if sample_name:
        src = BASE / "data" / f"{sample_name}.csv"
        if not src.exists():
            src = BASE / f"{sample_name}.csv"
        if not src.exists():
            return jsonify({"error": f"Sample file {sample_name}.csv not found"}), 404
        import shutil
        shutil.copy(src, csv_path)
    elif "file" in request.files and request.files["file"].filename != "":
        request.files["file"].save(csv_path)
    elif "csv" in request.files and request.files["csv"].filename != "":
        request.files["csv"].save(csv_path)
    else:
        return jsonify({"error": "No CSV file or sample specified"}), 400

    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "created": time.time()}

    if os.environ.get("VERCEL"):
        # On Vercel serverless, run synchronously to avoid background thread freezing
        iters = min(iters, 250_000)
        _run_solve(job_id, csv_path, out_path, iters, seed)
        with _jobs_lock:
            return jsonify(_jobs.get(job_id, {"status": "error"}))

    t = threading.Thread(
        target=_run_solve,
        args=(job_id, csv_path, out_path, iters, seed),
        daemon=True
    )
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    # Convert rules_audit list → dict for frontend
    if isinstance(job.get("rules_audit"), list):
        audit_dict = {}
        for idx, r in enumerate(job["rules_audit"], start=1):
            audit_dict[f"rule{idx}"] = r.get("passed", True)
        job["rules_audit"] = audit_dict
    # Add top-level fields frontend expects
    if "judge_score" in job and "score" not in job:
        job["score"] = job["judge_score"]
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job.get("status") != "done":
        return jsonify({"error": "Job not ready"}), 404
    return send_file(
        job["out_path"],
        as_attachment=True,
        download_name="submission.json",
        mimetype="application/json"
    )


@app.route("/sample/<num>")
def sample(num):
    p = BASE / "data" / f"sample{num}.csv"
    if not p.exists():
        p = BASE / f"sample{num}.csv"
    if not p.exists():
        return "Not found", 404
    return send_file(p, mimetype="text/csv")


@app.route("/demo-data")
def demo_data():
    p = BASE / "data" / "demo.json"
    if p.exists():
        return send_file(p, mimetype="application/json")
    return "Not found", 404


if __name__ == "__main__":
    import webbrowser
    os.makedirs(BASE / "templates", exist_ok=True)
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    print("=" * 60)
    print("  AlgoX Solver Web App running at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000)
