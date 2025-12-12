"""
Module d'optimisation pour l'ordonnancement militaire avec Gurobi
"""

from gurobipy import Model, GRB, quicksum

def optimize_military_schedule(missions, resources, horizon=24):
    """
    Résout le problème d'ordonnancement militaire avec Gurobi
    
    Paramètres:
    -----------
    missions : dict
        Dictionnaire des missions avec leurs caractéristiques
        Format: {
            "mission_id": {
                "dur": durée (int),
                "priority": priorité (float),
                "deadline": échéance (int),
                "req": {"ressource1": quantité, ...}
            }
        }
    
    resources : dict
        Dictionnaire des ressources disponibles
        Format: {"ressource1": quantité, ...}
    
    horizon : int
        Horizon temporel pour la planification (en heures)
    
    Retourne:
    --------
    tuple: (schedule, objective_value)
        schedule: dict des missions programmées {mission_id: heure_de_début ou None}
        objective_value: valeur de la fonction objectif optimale
    """
    
    # Création du modèle
    model = Model("MilitaryAssignment")
    model.Params.OutputFlag = 0  # Désactiver la sortie console
    
    # ============================================
    # 1. Variables de décision
    # ============================================
    # x[m][t] = 1 si la mission m commence à l'heure t, 0 sinon
    x = {}
    for m in missions:
        dur = missions[m]["dur"]
        x[m] = {}
        for t in range(max(0, horizon - dur + 1)):
            x[m][t] = model.addVar(vtype=GRB.BINARY, name=f"x_{m}_{t}")
    
    model.update()
    
    # ============================================
    # 2. Contraintes de ressources
    # ============================================
    # Pour chaque ressource et chaque heure, la consommation totale ≤ disponibilité
    for r in resources:
        for t in range(horizon):
            expr = quicksum(
                x[m][s] * missions[m]["req"].get(r, 0)
                for m in missions
                for s in range(
                    max(0, t - missions[m]["dur"] + 1),
                    min(t + 1, horizon - missions[m]["dur"] + 1)
                )
            )
            model.addConstr(expr <= resources[r], name=f"res_{r}_t{t}")
    
    # ============================================
    # 3. Contraintes temporelles
    # ============================================
    # Contraintes de deadline
    for m in missions:
        dur = missions[m]["dur"]
        deadline = missions[m].get("deadline", horizon)
        
        for t in x[m]:
            if t + dur > deadline:
                model.addConstr(x[m][t] == 0, name=f"dead_{m}_{t}")
    
    # ============================================
    # 4. Contraintes d'assignement
    # ============================================
    # Chaque mission est exécutée au plus une fois
    for m in missions:
        model.addConstr(
            quicksum(x[m][t] for t in x[m]) <= 1,
            name=f"once_{m}"
        )
    
    # ============================================
    # 5. Fonction objectif
    # ============================================
    # Maximiser la somme des priorités des missions exécutées
    priority_obj = quicksum(
        x[m][t] * missions[m]["priority"]
        for m in missions
        for t in x[m]
    )
    
    model.setObjective(priority_obj, GRB.MAXIMIZE)
    
    # ============================================
    # 6. Résolution
    # ============================================
    model.optimize()
    
    # Vérification du statut
    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(
            f"Gurobi n'a pas trouvé de solution valide (statut: {model.status})"
        )
    
    # ============================================
    # 7. Extraction de la solution
    # ============================================
    schedule = {}
    for m in missions:
        schedule[m] = None
        for t in x[m]:
            if x[m][t].X > 0.5:  # Variable binaire activée
                schedule[m] = int(t)
    
    objective_value = model.ObjVal if model.SolCount > 0 else 0.0
    
    return schedule, objective_value


# ============================================
# Exemple d'utilisation
# ============================================
if __name__ == "__main__":
    # Données d'exemple
    missions = {
        "M1": {
            "dur": 6,
            "priority": 3.0,
            "deadline": 24,
            "req": {"soldats": 10, "drones": 2}
        },
        "M2": {
            "dur": 4,
            "priority": 2.0,
            "deadline": 12,
            "req": {"soldats": 5, "véhicules": 1}
        },
        "M3": {
            "dur": 8,
            "priority": 1.0,
            "deadline": 20,
            "req": {"soldats": 8, "hélicoptères": 1, "médecins": 1}
        },
        "M4": {
            "dur": 5,
            "priority": 2.0,
            "deadline": 18,
            "req": {"soldats": 6, "drones": 1}
        },
        "M5": {
            "dur": 3,
            "priority": 4.0,
            "deadline": 10,
            "req": {"soldats": 4, "véhicules": 1, "médecins": 1}
        }
    }
    
    resources = {
        "soldats": 25,
        "drones": 4,
        "véhicules": 3,
        "hélicoptères": 2,
        "médecins": 2
    }
    
    # Résolution
    try:
        schedule, objective_value = optimize_military_schedule(
            missions, resources, horizon=24
        )
        
        # Affichage des résultats
        print("=" * 60)
        print("RÉSULTATS DE L'OPTIMISATION")
        print("=" * 60)
        print(f"Valeur objectif (priorité totale): {objective_value:.2f}")
        print("\nPlanning obtenu:")
        print("-" * 60)
        
        scheduled_count = 0
        for mission, start_time in schedule.items():
            if start_time is not None:
                end_time = start_time + missions[mission]["dur"]
                print(f"{mission}: H{start_time:02d} → H{end_time:02d} "
                      f"(durée: {missions[mission]['dur']}h, "
                      f"priorité: {missions[mission]['priority']})")
                scheduled_count += 1
            else:
                print(f"{mission}: Non planifiée")
        
        print("-" * 60)
        print(f"Missions planifiées: {scheduled_count}/{len(missions)}")
        
        # Analyse d'utilisation des ressources
        print("\nAnalyse d'utilisation des ressources:")
        print("-" * 60)
        
        for hour in range(24):
            resource_usage = {r: 0 for r in resources}
            active_missions = []
            
            for mission, start_time in schedule.items():
                if start_time is not None:
                    dur = missions[mission]["dur"]
                    if start_time <= hour < start_time + dur:
                        active_missions.append(mission)
                        for resource, amount in missions[mission]["req"].items():
                            resource_usage[resource] += amount
            
            if active_missions:
                usage_str = ", ".join([
                    f"{r}: {resource_usage[r]}/{resources[r]}"
                    for r in resources
                    if resource_usage[r] > 0
                ])
                print(f"H{hour:02d}: {', '.join(active_missions)} | {usage_str}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Erreur lors de l'optimisation: {str(e)}")