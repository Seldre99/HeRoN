import numpy as np
import matplotlib.pyplot as plt
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.environment import BattleEnv
from classes.agent import DQNAgent
import pandas as pd
import os
import action_score as score
import re


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


# Main loop with training
def train_dqn(episodes, batch_size=32, load_model_path=None):
    player_spells = [fire, thunder, blizzard, meteor, cura]
    player_items = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                    {"item": hielixer, "quantity": 1}]
    player1 = Person("Valos", 3260, 132, 300, 34, player_spells, player_items)
    enemy1 = Person("Magus", 4000, 701, 525, 25, [fire, cura], [])

    players = [player1]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    agent = DQNAgent(env.state_size, env.action_size, load_model_path)

    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    action_scores = []

    '''
    # Load existing progress
    rewards_per_episode = load_csv_series("reward_per_episode.csv", "Reward")
    agent_wins = load_csv_series("agent_wins.csv", "Wins")
    enemy_wins = load_csv_series("enemy_wins.csv", "Wins")
    agent_moves_per_episode = load_csv_series("agent_moves.csv", "Moves")
    success_rate = load_csv_series("success_rate.csv", "Rate")
    '''

    total_agent_wins = 0

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0
        moves = 0
        match_score = []
        score.reset_quantity()
        while not done:
            action = agent.act(state, env)

            match = map_action(action)
            total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
            match_score.append(round(total_score.get(match), 2))

            next_state, reward, done, a_win, e_win, enemy_choise = env.step(action)
            score.updage_quantity(match, players[0].get_mp())

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
    print("Media delle ricompense: ", np.mean(rewards_per_episode))
    print("Media delle mosse: ", np.mean(agent_moves_per_episode))
    print("Media score mosse: ", np.mean(action_scores))

    #if (e + 1) % 200 == 0:
    #    save_path = f"model_dqn_episode_{e + 1}"
    #    print(f"Saving model to {save_path}...")
    #    agent.save(save_path)
    agent.save(f"model_dqn_episode_1000")

    #append_csv("reward_per_episode.csv", rewards_per_episode, "Reward")
    #append_csv("agent_wins.csv", agent_wins, "Wins")
    #append_csv("enemy_wins.csv", enemy_wins, "Wins")
    #append_csv("agent_moves.csv", agent_moves_per_episode, "Moves")
    #append_csv("success_rate.csv", success_rate, "Rate")

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores


# Plotting function
def plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, match_score):
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.savefig("Train_reward_DQN.png")

    plt.figure(figsize=(8, 6))
    # Calcoliamo le vittorie cumulative
    cumulative_agent_wins = np.cumsum(agent_wins)
    cumulative_enemy_wins = np.cumsum(enemy_wins)

    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')

    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.savefig("Train_cumulative_Win_DQN.png")

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Train_moves_DQN.png")

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Train_success_rate_DQN.png")

    plt.figure(figsize=(8, 6))
    plt.plot(match_score)
    plt.title('Score mosse per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Score')
    plt.savefig("Score_DQN.png")


def export_success_rate(success_rate):
    df = pd.DataFrame({
        "Episode": list(range(1, len(success_rate) + 1)),
        "Success Rate": success_rate
    })

    df.to_csv('train_success_rate_model_dqn_1000.csv', index=False)


def append_csv(path, data, column_name):
    df = pd.DataFrame({
        "Episode": list(range(1, 1 + len(data))),
        column_name: data
    })
    df.to_csv(path, index=False)


def load_csv_series(filename, column):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return df[column].tolist()
    return []


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
    rewards, agent_wins, enemy_wins, moves, success_rate, match_score = train_dqn(episodes=1000)
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, match_score)
    export_success_rate(success_rate)