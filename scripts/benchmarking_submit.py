#!/usr/bin/env python3
from __future__ import print_function
import argparse, csv, datetime, json, os, re, subprocess, sys
from pathlib import Path

def project_root(): return Path(__file__).resolve().parents[1]
def instance_id(path):
    m = re.search(r"(instance_\d+)", path.name)
    if m: return m.group(1)
    name = path.name
    if name.endswith(".original.mps"): return name[:-len(".original.mps")]
    if name.endswith(".mps"): return name[:-len(".mps")]
    return path.stem

def run_logged(cmd, cwd, env, log_path):
    if not log_path.parent.exists(): log_path.parent.mkdir(parents=True)
    with log_path.open("w") as f:
        f.write("[cmd] " + " ".join(str(x) for x in cmd) + "\n")
        f.write("[cwd] " + str(cwd) + "\n\n"); f.flush()
        p = subprocess.Popen([str(x) for x in cmd], cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, universal_newlines=True)
        rc = p.wait(); f.write("\n[exit_code] {0}\n".format(rc)); return int(rc)

def write_manifest_csv(rows, path):
    fields = ["instance","instance_path","heuristic_returncode","baseline_returncode","heuristic_run_dir","heuristic_log","baseline_log","baseline_event_log","baseline_result_json"]
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})

def main():
    root = project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-dir", default="tests")
    ap.add_argument("--pattern", default="instance_*.original.mps")
    ap.add_argument("--config", default="config/default_config.json")
    ap.add_argument("--seconds", type=float, default=300.0, help="Total baseline window.")
    ap.add_argument("--heuristic-seconds", type=float, default=None, help="Wall seconds for phase1+phase2. Default seconds/2.")
    ap.add_argument("--final-seconds", type=float, default=None, help="Default seconds - heuristic_seconds.")
    ap.add_argument("--compute-units", type=int, default=16)
    ap.add_argument("--workflow-script", default="slurm/submit_workflow.sh")
    ap.add_argument("--baseline-script", default="scripts/run_baseline.sh")
    ap.add_argument("--run-base-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-heuristic", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--keep-going", action="store_true", default=True)
    args = ap.parse_args()

    heuristic_seconds = args.heuristic_seconds if args.heuristic_seconds is not None else args.seconds/2.0
    final_seconds = args.final_seconds if args.final_seconds is not None else max(1.0, args.seconds - heuristic_seconds)
    phase1_seconds, phase2_seconds = heuristic_seconds/2.0, heuristic_seconds/2.0

    tests_dir = (root/args.tests_dir).resolve()
    config = (root/args.config).resolve()
    workflow = (root/args.workflow_script).resolve()
    baseline = (root/args.baseline_script).resolve()
    run_base = Path(args.run_base_dir).resolve() if args.run_base_dir else Path(os.environ.get("RUN_BASE_DIR", str(root))).resolve()
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root/"results"/"benchmarking_raw"/stamp).resolve()
    logs_dir = out_dir/"logs"
    if not logs_dir.exists(): logs_dir.mkdir(parents=True)

    instances = sorted(tests_dir.glob(args.pattern))
    if args.limit is not None: instances = instances[:max(0,args.limit)]
    if not instances: print("ERROR: no instances found", file=sys.stderr); return 2

    print("Output dir:         {0}".format(out_dir))
    print("Baseline seconds:   {0}".format(args.seconds))
    print("Heuristic seconds:  {0}".format(heuristic_seconds))
    print("Phase1 wall target: {0}".format(phase1_seconds))
    print("Phase2 wall target: {0}".format(phase2_seconds))
    print("Final seconds:      {0}".format(final_seconds))

    rows = []
    for idx, inst in enumerate(instances, 1):
        inst = inst.resolve(); iid = instance_id(inst)
        print("\n[{0}/{1}] {2}".format(idx, len(instances), iid))
        run_dir = run_base/"milp_runs"/iid
        heur_log = logs_dir/(iid+"_heuristic_workflow.log")
        base_log = logs_dir/(iid+"_baseline.log")
        base_event_log = logs_dir/(iid+"_baseline_incumbents.csv")
        base_result_json = logs_dir/(iid+"_baseline.json")
        heur_rc = 0; base_rc = 0
        if not args.skip_heuristic:
            env = os.environ.copy()
            env.update({
                "HEURISTIC_WALL_SECONDS": str(heuristic_seconds),
                "PHASE1_WALL_SECONDS": str(phase1_seconds),
                "PHASE2_WALL_SECONDS": str(phase2_seconds),
                "FINAL_SECONDS": str(final_seconds),
                "FINAL_THREADS": str(args.compute_units),
                "ARRAY_TASKS": str(args.compute_units),
                "RUN_BASE_DIR": str(run_base),
                "PYTHONPATH": str(root),
                "ALLOW_PARTIAL_PHASE_FAILURES": "1",
            })
            heur_rc = run_logged(["bash", str(workflow), str(inst), str(config)], root, env, heur_log)
            print("  heuristic rc: {0}".format(heur_rc))
            if heur_rc != 0 and not args.keep_going: return heur_rc
        if not args.skip_baseline:
            env = os.environ.copy(); env.update({
                "TIME_LIMIT": str(args.seconds),
                "THREADS": str(args.compute_units),
                "BASELINE_USE_PYTHON": "1",
                "BASELINE_EVENT_LOG": str(base_event_log),
                "BASELINE_RESULT_JSON": str(base_result_json),
                "BASELINE_SOLVER_LOG": str(logs_dir/(iid+"_baseline_solver.log")),
            })
            base_rc = run_logged(["bash", str(baseline), str(inst)], root, env, base_log)
            print("  baseline rc: {0}".format(base_rc))
            if base_rc != 0 and not args.keep_going: return base_rc
        rows.append({"instance": iid, "instance_path": str(inst), "heuristic_returncode": heur_rc, "baseline_returncode": base_rc, "heuristic_run_dir": str(run_dir), "heuristic_log": str(heur_log), "baseline_log": str(base_log), "baseline_event_log": str(base_event_log), "baseline_result_json": str(base_result_json)})
        manifest = {"created_utc": stamp, "project_root": str(root), "config": str(config), "seconds": args.seconds, "heuristic_seconds": heuristic_seconds, "phase1_wall_seconds": phase1_seconds, "phase2_wall_seconds": phase2_seconds, "final_seconds": final_seconds, "compute_units": args.compute_units, "run_base_dir": str(run_base), "rows": rows}
        with (out_dir/"manifest.json").open("w") as f: json.dump(manifest, f, indent=2)
        write_manifest_csv(rows, out_dir/"manifest.csv")
    print("\nDone.")
    print("Manifest: {0}".format(out_dir/"manifest.json"))
    print("Gather inside container:")
    print("  python3 scripts/benchmarking_gathering.py --manifest {0}".format(out_dir/"manifest.json"))
    return 0
if __name__ == "__main__": sys.exit(main())
