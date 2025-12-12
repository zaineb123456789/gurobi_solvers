import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np

class NetworkOptimizer:
    """
    Classe pour optimiser le routage de données dans un réseau
    avec minimisation des coûts de transmission
    """
    
    def __init__(self, num_nodes, edges, demand, objective_type=0, 
                 use_reliability=True, use_balance=False):
        """
        Initialisation de l'optimiseur
        
        Args:
            num_nodes: Nombre de nœuds dans le réseau
            edges: Liste de tuples (source, dest, capacity, cost, latency)
            demand: Demande totale à acheminer de la source (0) à la destination (n-1)
            objective_type: 0=coût, 1=latence, 2=multi-critère
            use_reliability: Utiliser contraintes de fiabilité
            use_balance: Utiliser équilibrage de charge
        """
        self.num_nodes = num_nodes
        self.edges = edges
        self.demand = demand
        self.objective_type = objective_type
        self.use_reliability = use_reliability
        self.use_balance = use_balance
        
        # Nœud source et destination
        self.source = 0
        self.destination = num_nodes - 1
        
        # Dictionnaires pour accès rapide
        self.edge_dict = {}
        for source, dest, capacity, cost, latency in edges:
            self.edge_dict[(source, dest)] = {
                'capacity': capacity,
                'cost': cost,
                'latency': latency
            }
        
        # Créer le modèle Gurobi
        self.model = gp.Model("Network_Routing")
        self.model.setParam('OutputFlag', 0)  # Désactiver sortie console
        
        # Variables et contraintes
        self.flow_vars = {}
        self.results = {}
        
    def build_model(self):
        """Construire le modèle d'optimisation"""
        
        # ============================================
        # VARIABLES DE DÉCISION
        # ============================================
        
        # x[i,j] = flux sur l'arête (i,j)
        for (i, j) in self.edge_dict.keys():
            self.flow_vars[(i, j)] = self.model.addVar(
                lb=0.0, 
                ub=self.edge_dict[(i, j)]['capacity'],
                vtype=GRB.CONTINUOUS,
                name=f"flow_{i}_{j}"
            )
        
        # Variables binaires pour savoir si un lien est utilisé
        self.link_used = {}
        for (i, j) in self.edge_dict.keys():
            self.link_used[(i, j)] = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"used_{i}_{j}"
            )
        
        self.model.update()
        
        # ============================================
        # CONTRAINTES DE CONSERVATION DU FLUX
        # ============================================
        
        for node in range(self.num_nodes):
            # Flux entrant
            inflow = gp.quicksum(
                self.flow_vars[(i, node)] 
                for (i, j) in self.edge_dict.keys() 
                if j == node
            )
            
            # Flux sortant
            outflow = gp.quicksum(
                self.flow_vars[(node, j)] 
                for (i, j) in self.edge_dict.keys() 
                if i == node
            )
            
            if node == self.source:
                # Nœud source: sortie = demande
                self.model.addConstr(
                    outflow - inflow == self.demand,
                    name=f"flow_balance_source_{node}"
                )
            elif node == self.destination:
                # Nœud destination: entrée = demande
                self.model.addConstr(
                    inflow - outflow == self.demand,
                    name=f"flow_balance_dest_{node}"
                )
            else:
                # Nœuds intermédiaires: conservation du flux
                self.model.addConstr(
                    inflow == outflow,
                    name=f"flow_balance_{node}"
                )
        
        # ============================================
        # CONTRAINTES DE CAPACITÉ
        # ============================================
        
        for (i, j), edge_data in self.edge_dict.items():
            capacity = edge_data['capacity']
            
            # Le flux ne peut pas dépasser la capacité
            self.model.addConstr(
                self.flow_vars[(i, j)] <= capacity,
                name=f"capacity_{i}_{j}"
            )
            
            # Lier la variable binaire au flux
            self.model.addConstr(
                self.flow_vars[(i, j)] <= capacity * self.link_used[(i, j)],
                name=f"link_activation_{i}_{j}"
            )
        
        # ============================================
        # CONTRAINTES DE FIABILITÉ (optionnel)
        # ============================================
        
        if self.use_reliability:
            # Assurer qu'au moins 2 chemins différents existent si possible
            # En limitant le flux sur chaque arête à 80% de la demande
            for (i, j) in self.edge_dict.keys():
                self.model.addConstr(
                    self.flow_vars[(i, j)] <= 0.8 * self.demand,
                    name=f"reliability_{i}_{j}"
                )
        
        # ============================================
        # CONTRAINTES D'ÉQUILIBRAGE (optionnel)
        # ============================================
        
        if self.use_balance:
            # Équilibrage: éviter qu'un lien soit trop chargé
            for (i, j), edge_data in self.edge_dict.items():
                capacity = edge_data['capacity']
                # Limiter l'utilisation à 70% de la capacité
                self.model.addConstr(
                    self.flow_vars[(i, j)] <= 0.7 * capacity,
                    name=f"balance_{i}_{j}"
                )
        
        # ============================================
        # FONCTION OBJECTIF
        # ============================================
        
        if self.objective_type == 0:
            # Minimiser le coût total
            total_cost = gp.quicksum(
                self.flow_vars[(i, j)] * self.edge_dict[(i, j)]['cost']
                for (i, j) in self.edge_dict.keys()
            )
            self.model.setObjective(total_cost, GRB.MINIMIZE)
            
        elif self.objective_type == 1:
            # Minimiser la latence moyenne pondérée
            total_latency = gp.quicksum(
                self.flow_vars[(i, j)] * self.edge_dict[(i, j)]['latency']
                for (i, j) in self.edge_dict.keys()
            )
            self.model.setObjective(total_latency, GRB.MINIMIZE)
            
        else:
            # Multi-critère: minimiser coût + alpha * latence
            alpha = 0.1  # Poids de la latence
            
            total_cost = gp.quicksum(
                self.flow_vars[(i, j)] * self.edge_dict[(i, j)]['cost']
                for (i, j) in self.edge_dict.keys()
            )
            
            total_latency = gp.quicksum(
                self.flow_vars[(i, j)] * self.edge_dict[(i, j)]['latency']
                for (i, j) in self.edge_dict.keys()
            )
            
            # Normaliser la latence pour qu'elle soit comparable au coût
            normalized_latency = total_latency / 100.0
            
            self.model.setObjective(
                total_cost + alpha * normalized_latency, 
                GRB.MINIMIZE
            )
        
        self.model.update()
    
    def solve(self):
        """Résoudre le problème d'optimisation"""
        
        # Construire le modèle
        self.build_model()
        
        # Mesurer le temps de résolution
        start_time = time.time()
        
        # Résoudre
        self.model.optimize()
        
        solve_time = time.time() - start_time
        
        # Extraire les résultats
        results = {
            'status': self.get_status_string(),
            'solve_time': solve_time
        }
        
        if self.model.Status == GRB.OPTIMAL:
            # Récupérer les flux optimaux
            flows = {}
            for (i, j), var in self.flow_vars.items():
                flows[(i, j)] = var.X
            
            results['flows'] = flows
            
            # Calculer les métriques
            total_cost = sum(
                flows[(i, j)] * self.edge_dict[(i, j)]['cost']
                for (i, j) in flows.keys()
            )
            
            total_latency = sum(
                flows[(i, j)] * self.edge_dict[(i, j)]['latency']
                for (i, j) in flows.keys() if flows[(i, j)] > 0.01
            )
            
            active_flows = sum(1 for flow in flows.values() if flow > 0.01)
            total_flow = sum(flows.values())
            
            avg_latency = total_latency / total_flow if total_flow > 0 else 0
            
            # Calculer l'utilisation moyenne
            total_capacity = sum(e['capacity'] for e in self.edge_dict.values())
            total_capacity_used = sum(flows.values())
            avg_utilization = total_capacity_used / total_capacity if total_capacity > 0 else 0
            
            results['total_cost'] = total_cost
            results['avg_latency'] = avg_latency
            results['active_links'] = active_flows
            results['total_flow'] = total_flow
            results['avg_utilization'] = avg_utilization
            results['total_capacity'] = total_capacity
            results['total_capacity_used'] = total_capacity_used
            
            # Trouver les chemins principaux
            main_paths = self.find_main_paths(flows)
            results['main_paths'] = main_paths
            
        else:
            results['message'] = "Aucune solution optimale trouvée. Vérifiez les contraintes."
        
        return results
    
    def get_status_string(self):
        """Convertir le statut Gurobi en string lisible"""
        status_dict = {
            GRB.OPTIMAL: 'optimal',
            GRB.INFEASIBLE: 'infeasible',
            GRB.INF_OR_UNBD: 'inf_or_unbounded',
            GRB.UNBOUNDED: 'unbounded',
            GRB.CUTOFF: 'cutoff',
            GRB.ITERATION_LIMIT: 'iteration_limit',
            GRB.NODE_LIMIT: 'node_limit',
            GRB.TIME_LIMIT: 'time_limit',
            GRB.SOLUTION_LIMIT: 'solution_limit',
            GRB.INTERRUPTED: 'interrupted',
            GRB.NUMERIC: 'numeric',
            GRB.SUBOPTIMAL: 'suboptimal'
        }
        return status_dict.get(self.model.Status, 'unknown')
    
    def find_main_paths(self, flows):
        """Trouver les chemins principaux utilisés"""
        paths = []
        
        # Utiliser un algorithme de recherche en profondeur pour trouver les chemins
        def dfs_path(node, visited, path, remaining_flow):
            if node == self.destination and remaining_flow > 0.01:
                # Chemin trouvé
                path_str = " → ".join(str(n) for n in path)
                paths.append(f"Chemin: {path_str} | Flux: {remaining_flow:.2f}")
                return
            
            for (i, j) in flows.keys():
                if i == node and j not in visited and flows[(i, j)] > 0.01:
                    flow_to_use = min(remaining_flow, flows[(i, j)])
                    visited.add(j)
                    path.append(j)
                    dfs_path(j, visited.copy(), path.copy(), flow_to_use)
                    
        # Limiter à 5 chemins principaux
        visited_start = {self.source}
        dfs_path(self.source, visited_start, [self.source], self.demand)
        
        return paths[:5]  # Retourner au plus 5 chemins
    
    def get_model_statistics(self):
        """Obtenir des statistiques sur le modèle"""
        return {
            'num_variables': self.model.NumVars,
            'num_constraints': self.model.NumConstrs,
            'num_binary_vars': self.model.NumBinVars,
            'num_continuous_vars': self.model.NumVars - self.model.NumBinVars
        }