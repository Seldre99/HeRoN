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
from classes.instructor_agent import InstructorAgent
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

SERVER_API_HOST = "http://127.0.0.1:1234" #host for connecting to LM Studio, if it does not work insert the string localhost:1234

lms.get_default_client(SERVER_API_HOST)

#Reviewer model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer_instruction = AutoTokenizer.from_pretrained("")
model_instruction = T5ForConditionalGeneration.from_pretrained("").to(device)


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


def train_dqn(episodes, model_save, batch_size=32):
    #environment settings
    player_spells = [fire, thunder, blizzard, meteor, cura]
    player_items = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                    {"item": hielixer, "quantity": 1}]
    player1 = Person("Valos", 3260, 132, 300, 34, player_spells, player_items)
    enemy1 = Person("Magus", 5000, 701, 525, 25, [fire, cura], [])

    players = [player1]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    #NPC
    agent = DQNAgent(env.state_size, env.action_size, None)

    #Reviewer
    instructor_agent = InstructorAgent(model_instruction, tokenizer_instruction, device)

    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    action_scores = []
    mean_suggestion = []
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
        threshold = 1.0
        decay = 0.1 #0.2
        suggestion = 0
        while not done:
            p = np.random.rand()
            if p > threshold and e < 600:
                # Description of environment and Helper action #
                suggestion += 1
                game_description = env.describe_game_state(last_enemy_move)

                input_text = "You are a game asstitant for player." \
                             "The battle involves a player and an enemy. Both player and enemy are characterised by:" \
                             "HP (health point), MP (magic point), Attack points, Defence points, Available magic points " \
                             "and Available items. " \
                             "For spells, the use of MP is necessary, while items have limited availability. " \
                             "The player can have a maximum of 3260 HP and 132 MP, while the enemy 5000 HP and 701 MP. " \
                             f"Given the game state '{game_description}', what is the next action to take? " \
                             "Write only the chosen action in square brackets and " \
                             "explain your reasoning briefly, max 50 words. /no_think"

                with lms.Client() as client:
                    model = client.llm.model("yi-34b-chat")  # Helper model
                    llm_response = model.respond(input_text, config={"temperature": 0})
                    llm_response = str(llm_response)
                    llm_response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
                    print("NPC Response: ", llm_response)
                    response = instructor_agent.generate_suggestion(game_description, llm_response) # Reviewer suggestion
                    print("\nReviewer response: ", response)

                    revise = f"Given the game state '{game_description}'. Your initial response was '{llm_response}'. Considering " \
                             f"this suggestion '{response}', rephrase your answer. Write only the chosen action in square brackets and explain your reasoning briefly, max 50 words. /no_think"

                    new_llm_response = model.respond(revise, config={"temperature": 0}) # Final Helper response

                    new_llm_response = str(new_llm_response)
                    new_llm_response = re.sub(r"<think>.*?</think>", "", new_llm_response, flags=re.DOTALL).strip()
                    print("\nNew NPC response: ", new_llm_response)

                action = map_llm_action_to_agent_action(new_llm_response)

                if action != None:
                    match = re.search(r'\[(.*?)\]', new_llm_response)
                    match = match.group(1).strip().lower()
                    if match == "granade":
                        match = "grenade"
                    elif match == "elixer":
                        match = "elixir"
                    total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
                    match_score.append(round(total_score.get(match), 2))
                else:
                    action = map_llm_action_to_agent_action(llm_response)
                    if action != None:
                        match = re.search(r'\[(.*?)\]', llm_response)
                        match = match.group(1).strip().lower()
                        if match == "granade":
                            match = "grenade"
                        elif match == "elixer":
                            match = "elixir"
                    else:
                        action = agent.act(state, env)
                        match = map_action(action)
                    total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
                    match_score.append(round(total_score.get(match), 2))
                    allucination += 1
            else:
                action = agent.act(state, env)
                match = map_action(action)
                total_score = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
                match_score.append(round(total_score.get(match), 2))

            # Execution of RL action #
            next_state, reward, done, a_win, e_win, last_enemy_move = env.step(action)
            score.updage_quantity(match, players[0].get_mp())
            total_reward += reward
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            moves += 1

            threshold = max(0, threshold - decay)

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
        mean_suggestion.append(suggestion)

    agent.save(model_save) # save the agent model
    print("Average rewards: ", np.mean(rewards_per_episode))
    print("Average moves: ", np.mean(agent_moves_per_episode))
    print("Average move score: ", np.mean(action_scores))
    print("Hallucinations: ", allucination)
    print("Number of suggestions per game: ", np.mean(mean_suggestion))

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores


def plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, match_score):
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.savefig("Reward_dqn_Yi_flan_threshold2_1000.png")

    plt.figure(figsize=(8, 6))
    cumulative_agent_wins = np.cumsum(agent_wins)
    cumulative_enemy_wins = np.cumsum(enemy_wins)

    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')
    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.savefig("Cumulative_Win_dqn_Yi_flan_threshold2_1000.png")

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Moves_dqn_Yi_flan_threshold2_1000.png")

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Success_rate_dqn_Yi_flan_threshold2_1000.png")

    plt.figure(figsize=(8, 6))
    plt.plot(match_score)
    plt.title('Score moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Score')
    plt.savefig("Score_dqn_Yi_flan_threshold2_1000.png")


def export_success_rate(success_rate):
    df = pd.DataFrame({
        "Episode": list(range(1, len(success_rate) + 1)),
        "Success Rate": success_rate
    })
    df.to_csv('', index=False)


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
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_score)

    export_success_rate(success_rate)



