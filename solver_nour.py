"""
Solveur Gurobi pour le problème de Flot Maximum avec contrainte de temps
Objectif bi-critère: Maximiser le flot + Minimiser le temps total d'évacuation
"""

from gurobipy import Model, GRB, quicksum
from typing import Dict, List, Tuple, Any


class EvacuationSolver:
    def __init__(self):
        self.model = None
        self.solution = None
        
    def solve(
        self,
        sources: Dict[str, int],  # {nom_source: nb_personnes}
        sinks: List[str],  # Liste des puits
        edges: List[Tuple[str, str, int, int]],  # (from, to, capacity, time)
        max_time: int,  # Temps maximum d'évacuation
        alpha: float = 0.7  # Poids pour le flot (1-alpha pour le temps)
    ) -> Dict[str, Any]:
        """
        Résout le problème de flot maximum avec contrainte de temps.
        
        Retourne un dictionnaire avec:
        - flow_values: flot sur chaque arête
        - edge_used: 0/1 pour chaque arête
        - total_flow: flot total
        - total_time: temps total d'évacuation
        - evacuated_per_source: personnes évacuées par source
        - received_per_sink: personnes reçues par puits
        - time_per_sink: temps pour atteindre chaque puits
        - percentage: pourcentage d'évacuation
        - status: statut de la solution
        """
        
        try:
            # Créer le modèle Gurobi
            self.model = Model("Evacuation_MaxFlow")
            self.model.setParam('OutputFlag', 0)  # Désactiver les logs
            
            # Collecter tous les noeuds
            nodes = set(sources.keys()) | set(sinks)
            for (u, v, _, _) in edges:
                nodes.add(u)
                nodes.add(v)
            
            # Variables de décision
            # x[i,j] = flot sur l'arête (i,j)
            x = {}
            # y[i,j] = 1 si l'arête est utilisée, 0 sinon
            y = {}
            # t[j] = temps pour atteindre le puits j
            t = {}
            
            for (u, v, cap, time) in edges:
                x[u, v] = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=cap, name=f"x_{u}_{v}")
                y[u, v] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{u}_{v}")
            
            for sink in sinks:
                t[sink] = self.model.addVar(vtype=GRB.INTEGER, lb=0, name=f"t_{sink}")
            
            # Variable pour le temps maximum parmi tous les puits
            t_max = self.model.addVar(vtype=GRB.INTEGER, lb=0, name="t_max")
            
            self.model.update()
            
            # Construire les listes d'arêtes entrantes et sortantes pour chaque noeud
            edges_in = {n: [] for n in nodes}
            edges_out = {n: [] for n in nodes}
            edge_dict = {}
            
            for (u, v, cap, time) in edges:
                edges_out[u].append((u, v))
                edges_in[v].append((u, v))
                edge_dict[u, v] = (cap, time)
            
            # Contraintes de conservation du flot
            for node in nodes:
                if node in sources:
                    # Source: flot sortant - flot entrant <= nb_personnes
                    out_flow = quicksum(x[e] for e in edges_out[node]) if edges_out[node] else 0
                    in_flow = quicksum(x[e] for e in edges_in[node]) if edges_in[node] else 0
                    self.model.addConstr(out_flow - in_flow <= sources[node], f"source_{node}")
                    # Au moins une personne doit sortir de chaque source
                    self.model.addConstr(out_flow >= 1, f"min_source_{node}")
                elif node in sinks:
                    # Puits: pas de contrainte particulière (accumulation)
                    pass
                else:
                    # Noeud intermédiaire: conservation du flot
                    out_flow = quicksum(x[e] for e in edges_out[node]) if edges_out[node] else 0
                    in_flow = quicksum(x[e] for e in edges_in[node]) if edges_in[node] else 0
                    self.model.addConstr(out_flow == in_flow, f"conservation_{node}")
            
            # Lier x et y: si x[i,j] > 0 alors y[i,j] = 1
            M = sum(sources.values())  # Grande valeur
            for (u, v) in x:
                cap, _ = edge_dict[u, v]
                self.model.addConstr(x[u, v] <= cap * y[u, v], f"link_{u}_{v}")
            
            # Calcul du temps pour atteindre chaque puits (chemin le plus long utilisé)
            # On utilise une formulation big-M pour calculer le temps
            for sink in sinks:
                # Temps = somme des temps des arêtes utilisées sur le chemin
                # Simplification: on prend le temps maximum parmi toutes les arêtes utilisées menant au puits
                relevant_edges = []
                visited = set()
                queue = [sink]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    for (u, v) in edges_in.get(current, []):
                        relevant_edges.append((u, v))
                        if u not in visited:
                            queue.append(u)
                
                if relevant_edges:
                    for (u, v) in relevant_edges:
                        _, time = edge_dict[u, v]
                        self.model.addConstr(t[sink] >= time * y[u, v], f"time_{sink}_{u}_{v}")
            
            # t_max >= t[j] pour tout puits j
            for sink in sinks:
                self.model.addConstr(t_max >= t[sink], f"tmax_{sink}")
            
            # Contrainte de temps total
            self.model.addConstr(t_max <= max_time, "max_time_constraint")
            
            # Fonction objectif bi-critère
            # Maximiser le flot total (somme des flots entrant dans les puits)
            total_flow_expr = quicksum(
                x[e] for sink in sinks for e in edges_in.get(sink, [])
            )
            
            # Normalisation pour combiner les objectifs
            max_possible_flow = sum(sources.values())
            
            # Objectif: alpha * (flot/max_flot) - (1-alpha) * (temps/max_temps)
            self.model.setObjective(
                alpha * total_flow_expr - (1 - alpha) * t_max,
                GRB.MAXIMIZE
            )
            
            # Résoudre
            self.model.optimize()
            
            if self.model.status == GRB.OPTIMAL or self.model.status == GRB.SUBOPTIMAL:
                # Extraire la solution
                flow_values = {}
                edge_used = {}
                for (u, v) in x:
                    flow_values[u, v] = int(x[u, v].X)
                    edge_used[u, v] = int(y[u, v].X)
                
                # Calculer le flot total
                total_flow = sum(
                    flow_values.get(e, 0) for sink in sinks for e in edges_in.get(sink, [])
                )
                
                # Personnes évacuées par source
                evacuated_per_source = {}
                for source in sources:
                    evacuated = sum(flow_values.get(e, 0) for e in edges_out.get(source, []))
                    evacuated_per_source[source] = evacuated
                
                # Personnes reçues par puits
                received_per_sink = {}
                for sink in sinks:
                    received = sum(flow_values.get(e, 0) for e in edges_in.get(sink, []))
                    received_per_sink[sink] = received
                
                # Temps par puits
                time_per_sink = {}
                for sink in sinks:
                    time_per_sink[sink] = int(t[sink].X)
                
                # Pourcentage d'évacuation
                total_people = sum(sources.values())
                percentage = (total_flow / total_people * 100) if total_people > 0 else 0
                
                self.solution = {
                    'flow_values': flow_values,
                    'edge_used': edge_used,
                    'total_flow': total_flow,
                    'total_time': int(t_max.X),
                    'evacuated_per_source': evacuated_per_source,
                    'received_per_sink': received_per_sink,
                    'time_per_sink': time_per_sink,
                    'percentage': percentage,
                    'total_people': total_people,
                    'status': 'optimal'
                }
                
                return self.solution
            
            elif self.model.status == GRB.INFEASIBLE:
                return {
                    'status': 'infeasible',
                    'message': "Aucune solution réalisable. Vérifiez les contraintes de temps ou de capacité."
                }
            else:
                return {
                    'status': 'error',
                    'message': f"Statut du solveur: {self.model.status}"
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


def create_factory_example():
    """
    Exemple: Incendie dans une usine - Évacuation des travailleurs
    """
    # Sources: différentes zones de l'usine avec des travailleurs
    sources = {
        "Atelier_A": 25,      # 25 ouvriers dans l'atelier A
        "Atelier_B": 30,      # 30 ouvriers dans l'atelier B
        "Bureau": 15,         # 15 employés de bureau
        "Entrepot": 20,       # 20 magasiniers
    }
    
    # Puits: sorties de secours
    sinks = ["Sortie_Nord", "Sortie_Sud", "Sortie_Est"]
    
    # Arêtes: couloirs et passages (from, to, capacité, temps en secondes)
    edges = [
        # Depuis Atelier A
        ("Atelier_A", "Couloir_1", 15, 10),
        ("Atelier_A", "Couloir_2", 12, 8),
        
        # Depuis Atelier B
        ("Atelier_B", "Couloir_2", 18, 12),
        ("Atelier_B", "Couloir_3", 15, 15),
        
        # Depuis Bureau
        ("Bureau", "Couloir_1", 10, 5),
        ("Bureau", "Hall", 8, 7),
        
        # Depuis Entrepot
        ("Entrepot", "Couloir_3", 12, 20),
        ("Entrepot", "Couloir_4", 10, 18),
        
        # Couloirs intermédiaires
        ("Couloir_1", "Hall", 20, 8),
        ("Couloir_2", "Hall", 25, 10),
        ("Couloir_3", "Couloir_4", 15, 5),
        ("Couloir_4", "Hall", 18, 12),
        
        # Vers les sorties
        ("Hall", "Sortie_Nord", 30, 15),
        ("Hall", "Sortie_Sud", 25, 12),
        ("Couloir_3", "Sortie_Est", 20, 10),
        ("Couloir_4", "Sortie_Est", 15, 8),
    ]
    
    # Temps maximum d'évacuation: 60 secondes
    max_time = 60
    
    return sources, sinks, edges, max_time


if __name__ == "__main__":
    # Test du solveur avec l'exemple de l'usine
    solver = EvacuationSolver()
    sources, sinks, edges, max_time = create_factory_example()
    
    result = solver.solve(sources, sinks, edges, max_time)
    
    if result['status'] == 'optimal':
        print("=" * 50)
        print("RÉSULTATS DE L'ÉVACUATION")
        print("=" * 50)
        print(f"\nFlot total évacué: {result['total_flow']} personnes")
        print(f"Temps total d'évacuation: {result['total_time']} secondes")
        print(f"Pourcentage d'évacuation: {result['percentage']:.1f}%")
        
        print("\n--- Par source ---")
        for source, count in result['evacuated_per_source'].items():
            print(f"  {source}: {count} personnes évacuées")
        
        print("\n--- Par puits ---")
        for sink, count in result['received_per_sink'].items():
            time = result['time_per_sink'][sink]
            print(f"  {sink}: {count} personnes (temps: {time}s)")
        
        print("\n--- Arêtes utilisées ---")
        for (u, v), used in result['edge_used'].items():
            if used:
                flow = result['flow_values'][u, v]
                print(f"  {u} -> {v}: {flow} personnes")
    else:
        print(f"Erreur: {result.get('message', 'Inconnue')}")
