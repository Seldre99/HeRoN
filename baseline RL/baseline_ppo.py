import numpy as np
import matplotlib.pyplot as plt
from classes.environment_ppo import BattleEnv
from classes.ppo_agent import PPOAgent
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
import pandas as pd
import os
import action_score as score


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


def train_ppo(episodes=1000, model_save):
    player_spells = [fire, thunder, blizzard, meteor, cura]
    player_items = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                    {"item": hielixer, "quantity": 1}]
    player1 = Person("Valos", 3260, 132, 300, 34, player_spells, player_items)
    enemy1 = Person("Magus", 5000, 701, 525, 25, [fire, cura], [])

    players = [player1]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    agent = PPOAgent(env.state_size, env.action_size)

    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    action_scores = []
    total_agent_wins = 0

    for ep in range(episodes):
        state = env.reset()
        state = state.reshape(1, -1)

        states, actions, rewards_ep, log_probs, dones, values = [], [], [], [], [], []
        total_reward = 0
        done = False
        moves = 0
        match_score = []
        score.reset_quantity()

        while not done:
            mask = env.get_action_mask()
            action, log_prob = agent.act(state, mask)

            match = map_action(action)
            total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
            match_score.append(round(total_score.get(match), 2))

            next_state, reward, done, a_win, e_win, enemy_choise = env.step(action)
            next_state = next_state.reshape(1, -1)

            score.updage_quantity(match, players[0].get_mp())
            total_reward += reward


            states.append(state[0])
            actions.append(action)
            rewards_ep.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            values.append(agent.critic(state)[0, 0])

            state = next_state
            total_reward += reward
            moves += 1

            if done:
                print(f"Episode: {ep}/{episodes}, Score: {total_reward}, Moves: {moves}")
                if a_win:
                    agent_wins.append(1)
                    enemy_wins.append(0)
                    total_agent_wins += 1
                else:
                    agent_wins.append(0)
                    enemy_wins.append(1)

                success_rate.append(total_agent_wins / (ep + 1))
                print("Agent Win: ", agent_wins.count(1), "Enemy Win: ", enemy_wins.count(1))

        rewards_per_episode.append(total_reward)
        agent_moves_per_episode.append(moves)
        action_scores.append(np.mean(match_score))

        advantages = agent.compute_gae(rewards_ep, values, dones)
        returns = advantages + np.array(values)

        agent.train(
            np.array(states),
            actions,
            np.array(log_probs),
            returns,
            advantages,
        )

    print("Average rewards: ", np.mean(rewards_per_episode))
    print("Average moves: ", np.mean(agent_moves_per_episode))
    print("Average move score: ", np.mean(action_scores))

    agent.save(model_save) # save the agent model

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores

# Plotting function
def plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, match_score):
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.savefig("Train_reward_PPO_1.png")

    plt.figure(figsize=(8, 6))
    cumulative_agent_wins = np.cumsum(agent_wins)
    cumulative_enemy_wins = np.cumsum(enemy_wins)

    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')

    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.savefig("Train_cumulative_Win_PPO_1.png")

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Train_moves_PPO_1.png")

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Train_success_rate_PPO_1.png")

    plt.figure(figsize=(8, 6))
    plt.plot(match_score)
    plt.title('Score mosse per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Score')
    plt.savefig("Score_PPO_1.png")


def export_success_rate(success_rate):
    df = pd.DataFrame({
        "Episode": list(range(1, len(success_rate) + 1)),
        "Success Rate": success_rate
    })

    df.to_csv('train_success_rate_model_PPO_1000.csv', index=False)


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
    rewards, agent_wins, enemy_wins, moves, success_rate, match_score = train_ppo(episodes=1000, model_save)
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, match_score)
    export_success_rate(success_rate)
