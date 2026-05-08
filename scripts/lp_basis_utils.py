"""
scripts/lp_basis_utils.py

Helpers for LP basis warm starts in gurobipy.

Important:
- VBasis/CBasis apply to LP models, not MIP starts.
- Use basis reuse when the model is the same or only lightly modified.
- If presolve is important, set LPWarmStart=2.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gurobipy as gp
from gurobipy import GRB


def capture_basis(model: gp.Model) -> Optional[Dict[str, Any]]:
    try:
        vars_ = model.getVars()
        cons = model.getConstrs()
        return {
            "available": True,
            "var_names": [v.VarName for v in vars_],
            "constr_names": [c.ConstrName for c in cons],
            "vbasis": [int(x) for x in model.getAttr(GRB.Attr.VBasis, vars_)],
            "cbasis": [int(x) for x in model.getAttr(GRB.Attr.CBasis, cons)],
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def apply_basis_if_valid(model: gp.Model, basis: Optional[Dict[str, Any]], lp_warm_start: int = 2) -> bool:
    if not basis or not basis.get("available"):
        return False

    vars_ = model.getVars()
    cons = model.getConstrs()

    var_names = basis.get("var_names") or []
    con_names = basis.get("constr_names") or []
    vbasis = basis.get("vbasis") or []
    cbasis = basis.get("cbasis") or []

    if len(vars_) != len(var_names) or len(cons) != len(con_names):
        return False

    if [v.VarName for v in vars_] != list(var_names):
        return False
    if [c.ConstrName for c in cons] != list(con_names):
        return False

    try:
        model.Params.LPWarmStart = int(lp_warm_start)
        model.setAttr(GRB.Attr.VBasis, vars_, [int(x) for x in vbasis])
        model.setAttr(GRB.Attr.CBasis, cons, [int(x) for x in cbasis])
        model.update()
        return True
    except Exception:
        return False