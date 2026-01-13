# Dictionary of actions
actions = {
    'attack': {'damage': 300, 'mp_cost': 0, 'heal': 0, 'quantity': 1},
    'fire spell': {'damage': 600, 'mp_cost': 25, 'heal': 0, 'quantity': 1},
    'thunder spell': {'damage': 700, 'mp_cost': 30, 'heal': 0, 'quantity': 1},
    'blizzard spell': {'damage': 800, 'mp_cost': 35, 'heal': 0, 'quantity': 1},
    'meteor spell': {'damage': 1000, 'mp_cost': 40, 'heal': 0, 'quantity': 1},
    'cura spell': {'damage': 0, 'mp_cost': 32, 'heal': 1500, 'quantity': 1},
    'potion': {'damage': 0, 'mp_cost': 0, 'heal': 50, 'mp_heal': 0, "quantity": 3},
    'grenade': {'damage': 500, 'mp_cost': 0, 'heal': 0, 'quantity': 2},
    'elixir': {'damage': 0, 'mp_cost': 0, 'heal': 3260, 'mp_heal': 132, 'quantity': 1}
}

def reset_quantity():
    for v, param in actions.items():
        if v == "potion":
            actions[v]['quantity'] = 3
        elif v == "grenade":
            actions[v]['quantity'] = 2
        elif v == "elixir":
            actions[v]['quantity'] = 1
        else:
            actions[v]['quantity'] = 1

def updage_quantity(action, mp_player):
    for v, param in actions.items():
        if actions[v]['mp_cost'] > mp_player:
            actions[v]['quantity'] = 0
        if v == "potion" == action or v == "grenade" == action or v == "elixir" == action:
            actions[v]['quantity'] -= 1
            if actions[v]['quantity'] < 0:
                actions[v]['quantity'] = 0
    if action == "elixir":
        actions['attack']['quantity'] = 1
        actions['fire spell']['quantity'] = 1
        actions['thunder spell']['quantity'] = 1
        actions['blizzard spell']['quantity'] = 1
        actions['meteor spell']['quantity'] = 1
        actions['cura spell']['quantity'] = 1


# Function for calculating normalised scores
def calculate_scores(hp, mp, hp_nemico):
    hp_max = 3260
    mp_max = 132
    hp_nemico_max = 5000

    score_dict = {}

    for nome, a in actions.items():
        # Action not executable
        if a['quantity'] == 0 or mp < a['mp_cost']:
            score_dict[nome] = 0
            continue

        # Basic calculations
        danno_effettivo = min(a.get('damage', 0), hp_nemico)
        cura_effettiva = min(a.get('heal', 0), hp_max - hp)
        recupero_mp = a.get('mp_heal', 0)

        # Dynamic weights
        p_danno = 1.0 + (1 - hp_nemico / hp_nemico_max)
        p_cura = 2.0 if hp < 0.3 * hp_max else 0.1
        p_mp = 1.5 if mp < 25 else 0.1
        p_costo_mp = 0.4

        # Score
        efficacia = (
            p_danno * danno_effettivo +
            p_cura * cura_effettiva +
            p_mp * recupero_mp -
            p_costo_mp * a['mp_cost'] -0
        )
        score_dict[nome] = max(efficacia, 0) 

    # Normalisation
    max_score = max(score_dict.values())
    if max_score > 0:
        score_dict = {k: v / max_score for k, v in score_dict.items()}
    else:
        score_dict = {k: 0 for k in score_dict}

    print(score_dict)
    return score_dict


