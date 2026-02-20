import csv
import random


actions = {
    'attack': {'damage': 300, 'mp_cost': 0, 'heal': 0},
    'fire spell': {'damage': 600, 'mp_cost': 25, 'heal': 0},
    'thunder spell': {'damage': 700, 'mp_cost': 30, 'heal': 0},
    'blizzard spell': {'damage': 800, 'mp_cost': 35, 'heal': 0},
    'meteor spell': {'damage': 1000, 'mp_cost': 40, 'heal': 0},
    'cura spell': {'damage': 0, 'mp_cost': 32, 'heal': 1500},
    'potion': {'damage': 0, 'mp_cost': 0, 'heal': 50},
    'grenade': {'damage': 500, 'mp_cost': 0, 'heal': 0},
    'elixir': {'damage': 0, 'mp_cost': 0, 'heal': 'full'}
}


def generate_game_scenario():
    player_hp = random.randint(10, 5000)
    player_mp = random.randint(0, 150)
    enemy_hp = random.randint(10, 5000)
    enemy_mp = random.randint(0, 800)
    available_items = {
        'potion': random.randint(0, 3),
        'grenade': random.randint(0, 2),
        'elixer': random.randint(0, 1)
    }
    last_enemy_move = random.choice(['attack', 'fire spell', 'cura spell'])
    available_actions = {action: info for action, info in actions.items() if player_mp >= info['mp_cost']}

    return {
        'player_hp': player_hp,
        'player_mp': player_mp,
        'enemy_hp': enemy_hp,
        'enemy_mp': enemy_mp,
        'available_items': available_items,
        'last_enemy_move': last_enemy_move,
        'available_actions': available_actions
    }


def generate_response(scenario):
    action = random.choice(list(scenario['available_actions'].keys()))
    return f"The next action to take is [{action}]"


def get_best_attack():
    best = None
    for action, info in actions.items():
        if info["damage"] > 0:
            if best is None or info["damage"] > actions[best]["damage"]:
                best = action
    return best


def generate_instructions(scenario, response):
    player_hp = scenario['player_hp']
    player_mp = scenario['player_mp']
    enemy_hp = scenario['enemy_hp']

    if player_hp < 1000 and enemy_hp < 1000:
        best_attack = get_best_attack()
        if best_attack:
            if response == best_attack:
                return "The action taken seems reasonable given the current state."
            return f"Consider using [{best_attack}] for more damage."
        return "Use [attack] since you don't have enough MP."

    # Low HP + low MP
    if player_hp < 800 and player_mp < 25:
        if items.get("elixer", 0) > 0:
            if response == "elixir":
                return "The action taken seems reasonable given the current state."
            return "You should use [elixir] to fully restore health and MP."
        if player_mp >= 32:
            if response == "cura spell":
                return "The action taken seems reasonable given the current state."
            return "You should use [cura spell] to heal more HP."
        if items.get("potion", 0) > 0:
            if response == "potion":
                return "The action taken seems reasonable given the current state."
            return "You should use a [potion] to heal."

    # Low HP only
    if player_hp < 800:
        if player_mp >= 32:
            if response == "cura spell":
                return "The action taken seems reasonable given the current state."
            return "You should use [cura spell] to heal more HP."
        if items.get("potion", 0) > 0:
            if response == "potion":
                return "The action taken seems reasonable given the current state."
            return "You should use a [potion] to heal."
        if items.get("elixer", 0) > 0:
            if response == "elixir":
                return "The action taken seems reasonable given the current state."
            return "You should use [elixir] to fully restore health and MP."

    # Default: attack recommendation
    best_attack = get_best_attack()
    if best_attack:
        if response == best_attack:
            return "The action taken seems reasonable given the current state."
        return f"Consider using [{best_attack}] for more damage."

    return "Use [attack] since you don't have enough MP."


def generate_dataset(n=5000):
    dataset = []
    for _ in range(n):
        scenario = generate_game_scenario()
        response = generate_response(scenario)
        instructions = generate_instructions(scenario, response)
        prompt = (
            f"Player has {scenario['player_hp']} hp and {scenario['player_mp']} mp. "
            f"Enemy has {scenario['enemy_hp']} hp and {scenario['enemy_mp']} mp. "
            f"Available actions: {', '.join(scenario['available_actions'].keys())}. "
            f"Last enemy move was {scenario['last_enemy_move']}."
        )
        dataset.append({
            'prompt': prompt,
            'response': response,
            'instructions': instructions
        })
    return dataset


def save_dataset_to_csv(dataset, filename='game_scenarios_dataset_3.csv'):
    keys = dataset[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)


dataset = generate_dataset()

save_dataset_to_csv(dataset)
