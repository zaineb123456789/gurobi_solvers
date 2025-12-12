import itertools
from collections import defaultdict
from gurobipy import Model, GRB, quicksum
from data import DAYS, SHIFTS, COMPETENCES, MIN_SHIFTS, MAX_SHIFTS, MAX_CONSEC_MATIN

class HiringScheduler:
    def __init__(self, candidates, demand, availability):
        self.candidates = candidates
        self.ids = [c["id"] for c in candidates]
        self.demand = demand
        self.avail = availability
        self.min_shifts = {c["id"]: MIN_SHIFTS for c in candidates}
        self.max_shifts = {c["id"]: MAX_SHIFTS for c in candidates}

    def solve(self, time_limit=30, verbose=False):
        E,D,S,K = self.ids, DAYS, SHIFTS, COMPETENCES
        m = Model("HiringShiftScheduling")
        if not verbose:
            m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', time_limit)

        x = {e:m.addVar(vtype=GRB.BINARY, name=f"x[{e}]") for e in E}
        y = {(e,d,s): m.addVar(vtype=GRB.BINARY, name=f"y[{e},{d},{s}]") for e,d,s in itertools.product(E,D,S)}
        m.update()

        m.setObjective(quicksum(next(c for c in self.candidates if c["id"]==e)["hire_cost"]*x[e] for e in E), GRB.MINIMIZE)

        for d in D:
            for s in S:
                for k in K:
                    req = self.demand[d][s].get(k,0)
                    if req>0:
                        m.addConstr(quicksum(next(c for c in self.candidates if c["id"]==e)["qual"].get(k,0)*y[(e,d,s)] for e in E)>=req,
                                    name=f"cover_{d}_{s}_{k}")


        for e,d,s in itertools.product(E,D,S):
            m.addConstr(y[(e,d,s)]<=x[e], name=f"link_{e}_{d}_{s}")

        for e,d,s in itertools.product(E,D,S):
            if self.avail.get(e,{}).get(d,1)==0:
                m.addConstr(y[(e,d,s)]==0, name=f"unavailable_{e}_{d}_{s}")


        for e,d in itertools.product(E,D):
            m.addConstr(quicksum(y[(e,d,s)] for s in S)<=1, name=f"one_per_day_{e}_{d}")

        for e in E:
            for i in range(len(D)-1):
                m.addConstr(y[(e,D[i],"Garde")]+y[(e,D[i+1],"Matin")]<=1,
                            name=f"rest_guard_matin_{e}_{D[i]}_{D[i+1]}")

        window = 3
        for e in E:
            for start in range(len(D)-window+1):
                span = "_".join(D[start:start+window])
                m.addConstr(quicksum(y[(e,D[i],"Garde")] for i in range(start,start+window))<=1,
                            name=f"spacing_garde_{e}_{span}")

        win_len = MAX_CONSEC_MATIN+1
        for e in E:
            for start in range(len(D)-win_len+1):
                span = "_".join(D[start:start+win_len])
                m.addConstr(quicksum(y[(e,D[i],"Matin")] for i in range(start,start+win_len))<=MAX_CONSEC_MATIN,
                            name=f"max_matin_seq_{e}_{span}")

        for e in E:
            m.addConstr(quicksum(y[(e,d,s)] for d in D for s in S)<=self.max_shifts[e]*x[e],
                        name=f"max_total_{e}")
            m.addConstr(quicksum(y[(e,d,s)] for d in D for s in S)>=self.min_shifts[e]*x[e],
                        name=f"min_total_{e}")

        m.optimize()
        status_map = {
            GRB.OPTIMAL: "optimal",
            GRB.SUBOPTIMAL: "suboptimal",
            GRB.TIME_LIMIT: "time_limit",
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible_or_unbounded"
        }
        status = status_map.get(m.status, f"other({m.status})")

        if m.status == GRB.INFEASIBLE or m.status == GRB.INF_OR_UNBD:
            try:
                m.computeIIS()
                iis = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
            except Exception:
                iis = []
            return {"status": status, "iis": iis, "messages": self.explain_iis(iis), "model": m}

        if m.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            return {"status": status, "model": m}

        hired = [e for e in E if x[e].X>0.5]
        assigns = defaultdict(list)
        emp_assignments = defaultdict(list)
        for e,d,s in itertools.product(E,D,S):
            if y[(e,d,s)].X>0.5:
                assigns[(d,s)].append(e)
                emp_assignments[e].append((d,s))

        total_infirmiers = sum(1 for e in hired if next(c for c in self.candidates if c["id"]==e)["qual"]["infirmier"]==1)
        total_medecins = sum(1 for e in hired if next(c for c in self.candidates if c["id"]==e)["qual"]["medecin"]==1)

        return {"status": status,
                "hired":hired,"assigns":assigns,"emp_assignments":emp_assignments,
                "objective":m.objVal,"model":m,
                "total_infirmiers":total_infirmiers,"total_medecins":total_medecins}

    def explain_iis(self, iis):
        msgs=[]
        for name in iis:
            parts = name.split("_")
            if name.startswith("cover_") and len(parts)>=4:
                _, day, shift, comp = parts[:4]
                msgs.append(f"Sous-effectif: manque compétence '{comp}' pour le créneau {day}_{shift}.")
            elif name.startswith("unavailable_") and len(parts)>=4:
                _, emp, day, shift = parts[:4]
                msgs.append(f"Indispo: {emp} n'est pas disponible {day} ({shift}).")
            elif name.startswith("one_per_day_") and len(parts)>=3:
                _, emp, day = parts[:3]
                msgs.append(f"Conflit: {emp} doit avoir ≤1 shift le jour {day}.")
            elif name.startswith("rest_guard_matin_") and len(parts)>=6:
                emp, d0, d1 = parts[3], parts[4], parts[5]
                msgs.append(f"Repos insuffisant: {emp} garde le {d0} empêche le day suivant.")
            elif name.startswith("spacing_garde_") and len(parts)>=5:
                _, _, emp, d0, d1, *rest = parts
                last = rest[-1] if rest else d1
                msgs.append(f"Spacing gardes: {emp} ne peut pas avoir gardes proches ({d0}, {last}).")
            elif name.startswith("max_matin_seq_") and len(parts)>=5:
                _, _, emp, d0, d1, *rest = parts
                last = rest[-1] if rest else d1
                msgs.append(f"Max Matin: trop de Matin consécutifs pour {emp} ({d0} à {last}).")
            elif name.startswith("min_total_") and len(parts)>=2:
                _, emp = parts[:2]
                msgs.append(f"Min shifts impossible pour {emp}.")
            elif name.startswith("max_total_") and len(parts)>=2:
                _, emp = parts[:2]
                msgs.append(f"Max shifts violée pour {emp}.")
            else:
                msgs.append(f"Contrainte en conflit: {name}")
        return msgs
