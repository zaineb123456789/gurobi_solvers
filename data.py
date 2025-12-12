DAYS = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
SHIFTS = ["Matin","Garde"]
COMPETENCES = ["infirmier","medecin"]

MIN_SHIFTS = 2
MAX_SHIFTS = 8
MAX_CONSEC_MATIN = 5

def default_demand():
    d = {}
    for day in DAYS:
        d[day] = {
            "Matin": {"infirmier": 3, "medecin": 1},
            "Garde": {"infirmier": 2, "medecin": 1}
        }
    return d

DEMAND = default_demand()

DEFAULT_CANDIDATES = []
for i in range(1, 11):
    DEFAULT_CANDIDATES.append({
        "id": f"I{i}", "name": f"Inf_{i}",
        "qual": {"infirmier":1, "medecin":0},
        "hire_cost": 1000 + 10*i,
        "shift_cost": {"Matin":80,"Garde":120}})
for i in range(1,5):
    DEFAULT_CANDIDATES.append({
        "id": f"M{i}", "name": f"Med_{i}",
        "qual": {"infirmier":0, "medecin":1},
        "hire_cost": 2000 + 20*i,
        "shift_cost": {"Matin":120,"Garde":180}})

def default_availability(candidate_list):
    return {c["id"]:{d:1 for d in DAYS} for c in candidate_list}
