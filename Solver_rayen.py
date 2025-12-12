from PyQt5.QtCore import QThread, pyqtSignal
from gurobipy import Model, GRB


class SolverThread(QThread):
    result_ready = pyqtSignal(dict, float)

    def __init__(self, produits, profit, ressources, besoins, disponibilite):
        super().__init__()
        self.produits = produits
        self.profit = profit
        self.ressources = ressources
        self.besoins = besoins
        self.disponibilite = disponibilite

    def run(self):
        try:
            model = Model("Production_PL")
            model.setParam('OutputFlag', 0)

            # Variables
            x = {p: model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{p}", lb=0)
                 for p in self.produits}

            # Objectif : maximiser profit
            model.setObjective(sum(self.profit[p] * x[p] for p in self.produits), GRB.MAXIMIZE)

            # Contraintes
            for r in self.ressources:
                model.addConstr(sum(self.besoins[(p, r)] * x[p] for p in self.produits)
                                <= self.disponibilite[r], name=f"R_{r}")

            model.optimize()

            if model.status == GRB.OPTIMAL:
                solution = {p: x[p].x for p in self.produits}
                self.result_ready.emit(solution, model.objVal)
            else:
                self.result_ready.emit({}, 0.0)

        except Exception as e:
            print("Erreur solveur :", e)
            self.result_ready.emit({}, 0.0)