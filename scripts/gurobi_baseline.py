#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

def dump(o,p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p,'w') as f: json.dump(o,f,indent=2,default=str)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('instance'); ap.add_argument('--out',required=True); ap.add_argument('--event-log',required=True); ap.add_argument('--solver-log',default=None)
    ap.add_argument('--time-limit',type=float,default=300.0); ap.add_argument('--threads',type=int,default=16); ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    env=gp.Env(empty=True); env.setParam('OutputFlag',0); env.start(); m=gp.read(args.instance, env=env)
    m.Params.TimeLimit=args.time_limit; m.Params.Threads=args.threads; m.Params.Seed=args.seed
    if args.solver_log: m.Params.LogFile=args.solver_log
    Path(args.event_log).parent.mkdir(parents=True, exist_ok=True)
    f=open(args.event_log,'w',newline='')
    w=csv.DictWriter(f,fieldnames=['time_sec','objective','incumbent_objective','method','phase','source','task_id'])
    w.writeheader(); f.flush()
    best=None; sense='min' if int(m.ModelSense)==1 else 'max'; t0=time.time()
    def cb(model, where):
        nonlocal best
        if where == GRB.Callback.MIPSOL:
            try: obj=float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            except Exception: return
            try: runtime=float(model.cbGet(GRB.Callback.RUNTIME))
            except Exception: runtime=time.time()-t0
            if best is None or (obj < best if sense=='min' else obj > best):
                best=obj; w.writerow({'time_sec':runtime,'objective':obj,'incumbent_objective':obj,'method':'gurobi_baseline','phase':'baseline','source':'mipsol_callback','task_id':'callback'}); f.flush()
    m.optimize(cb); f.close()
    out={'status':'ok','instance_path':args.instance,'time_limit':args.time_limit,'threads':args.threads,'runtime_sec':time.time()-t0,
         'solver_status':int(m.Status),'event_log':args.event_log,'solver_log':args.solver_log,'solution_count':int(m.SolCount)}
    if m.SolCount:
        out['objective']=float(m.ObjVal); out['best_obj']=float(m.ObjVal); out['best_objective']=float(m.ObjVal); out['values']={v.VarName:float(v.X) for v in m.getVars()}
    for k,a in [('bound','ObjBound'),('mip_gap','MIPGap')]:
        try: out[k]=float(getattr(m,a))
        except Exception: out[k]=None
    dump(out,args.out)
if __name__=='__main__': raise SystemExit(main())
