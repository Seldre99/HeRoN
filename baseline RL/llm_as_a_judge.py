import numpy as np
import matplotlib.pyplot as plt
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.agent import DQNAgent
from classes.environment import BattleEnv
import pandas as pd
import re
import lmstudio as lms
import action_score as score
import os
import csv

SERVER_API_HOST = "localhost:1234"

lms.get_default_client(SERVER_API_HOST)


def map_llm_action_to_agent_action(llm_response):
    match = re.search(r'\[(.*?)\]', llm_response)
    if match:
        action = match.group(1).strip().lower()
        if action == "attack":
            return 0
        elif action == "fire spell":
            return 1
        elif action == "thunder spell":
            return 2
        elif action == "blizzard spell":
            return 3
        elif action == "meteor spell":
            return 4
        elif action == "cura spell":
            return 5
        elif action == "potion":
            return 6
        elif action == "grenade":
            return 7
        elif action == "elixir" or action == "elixer":
            return 8
    return None

def map_action(action):
    if action == 0:
        return "attack"
    elif action == 1:
        return "fire spell"
    elif action == 2:
        return "thunder spell"
    elif action == 3:
        return "blizzard spell"
    elif action == 4:
        return "meteor spell"
    elif action == 5:
        return "cura spell"
    elif action == 6:
        return "potion"
    elif action == 7:
        return "grenade"
    elif action == 8:
        return "elixir"
    return None



def normalize_score(score, min_score=1, max_score=10):
    score = np.clip(score, min_score, max_score)
    return 2 * (score - min_score) / (max_score - min_score) - 1


def train_dqn(episodes, model_save, batch_size=32):
    player_spells = [fire, thunder, blizzard, meteor, cura]
    player_items = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                    {"item": hielixer, "quantity": 1}]
    player1 = Person("Valos", 3260, 132, 300, 34, player_spells, player_items)
    enemy1 = Person("Magus", 5000, 701, 525, 25, [fire, cura], [])

    players = [player1]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    agent = DQNAgent(env.state_size, env.action_size, None)


    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    action_scores = []
    allucination = 0
    total_agent_wins = 0

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0
        moves = 0
        match_score = []
        last_enemy_move = "No action"
        score.reset_quantity()
        while not done:
            # Descrizione environment ed azione LLM #
            game_description = env.describe_game_state(last_enemy_move)
            action = agent.act(state, env)

            input_text = "You are an expert evaluator of Reinforcement Learning agent decisions in turn-based combat environments." \
                         "Your role is to provide a SCORE from 1 to 10 critically assess the QUALITY of a single ACTION selected by an RL agent given the current game state." \
                         "Where SCORE of 1 refers to a bad choise, and a SCORE of 10 is a perfect choise." \
                         f"Given the current game state '{game_description}', the action selected by RL is {action}"\
                         "You must respond ONLY writing the chosen score. /no_think"


            with lms.Client() as client:
                model = client.llm.model("") #change LLM
                llm_response = model.respond(input_text)

            llm_response = str(llm_response)
            llm_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
            print(f"LLM response: {llm_response}")
          
            match = map_action(action)
            total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
            match_score.append(round(total_score.get(match), 2))

            # Esecuzione azione RL #
            next_state, reward, done, a_win, e_win, last_enemy_move = env.step(action)
            score.updage_quantity(match, players[0].get_mp())
            reward = normalize_score(int(llm_response))
            total_reward += reward
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves += 1
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, env)

            if done:
                print(f"Episode: {e}/{episodes}, Score: {total_reward}, Moves: {moves}, Epsilon: {agent.epsilon}")
                if a_win:
                    agent_wins.append(1)
                    enemy_wins.append(0)
                    total_agent_wins += 1
                else:
                    agent_wins.append(0)
                    enemy_wins.append(1)

                success_rate.append(total_agent_wins / (e + 1))
                print("Vittorie agente: ", agent_wins.count(1), " Vittorie nemico: ", enemy_wins.count(1))
        rewards_per_episode.append(total_reward)
        agent_moves_per_episode.append(moves)
        action_scores.append(np.mean(match_score))

    agent.save(model_save)
    print("Media delle ricompense: ", np.mean(rewards_per_episode))
    print("Media delle mosse: ", np.mean(agent_moves_per_episode))
    print("Media score mosse: ", np.mean(action_scores))

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores


def save_metrics_to_csv(filepath: str, agent_wins, rewards_per_episode, agent_moves_per_episode, action_scores):

    row = {
        "Vittorie agente": agent_wins.count(1),
        "Media delle ricompense": np.mean(rewards_per_episode),
        "Media delle mosse": np.mean(agent_moves_per_episode),
        "Media score mosse": np.mean(action_scores),
    }

    file_exists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

if __name__ == "__main__":
    # Spells and items setup
    fire = Spell("Fire", 25, 600, "black")
    thunder = Spell("Thunder", 30, 700, "black")
    blizzard = Spell("Blizzard", 35, 800, "black")
    meteor = Spell("Meteor", 40, 1000, "black")
    cura = Spell("Cura", 32, 1500, "white")

    potion = Item("Potion", "potion", "Heals 50 HP", 50)
    hielixer = Item("MegaElixer", "elixer", "Fully restores party's HP/MP", 9999)
    grenade = Item("Grenade", "attack", "Deals 500 damage", 500)

    # Train the agent
    rewards, agent_wins, enemy_wins, moves, success_rate, action_score = train_dqn(episodes=1000, model_save="")
    save_metrics_to_csv("", agent_wins, rewards, moves, action_score)
