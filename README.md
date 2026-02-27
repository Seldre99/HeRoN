# HeRoN: A Mediated RL–LLM Framework for Adaptive NPC Behavior in Interactive Environments 🎮
HeRoN (Helper–Reviewer–NPC) is a mediated Reinforcement Learning framework that integrates Large Language Models (LLMs) into the training loop of adaptive Non-Player Characters (NPCs).
The framework introduces functional separation and critique-based mediation to improve contextual coherence, strategic consistency, and learning efficiency.

## 📌 Motivation
Recent progress in Reinforcement Learning (RL) and Large Language Models (LLMs) has opened new opportunities for improving NPC decision-making, but both face key limitations: 
* RL struggles with training efficiency and generalization;
* LLMs are prone to hallucinations and context drift.
We present a mediated framework that integrates RL and LLMs through functional separation and critique-based mediation, enabling contextually coherent and strategically adaptive NPC behavior across interactive environments. 

## 🧠 HeRoN Architecture
HeRoN combines three components: 
* NPC: an RL-driven agent whose policy is iteratively refined via LLM-generated critiques;
* Helper: an LLM operating in zero-shot reasoning mode to generate diverse, context-aware action strategies;
* Reviewer: a lightweight, fine-tuned LLM that evaluates and refines the Helper’s suggestions, ensuring strategic consistency and alignment with game-specific constraints.

## 🧪 Evaluation Environments
We test HeRoN across two domains:
* 🎮 Custom Turn-Based Battle Environment
* 🔫 FPS Environment: [ViZDoom](https://github.com/rosariopiognazzo/DoomHeron)

## 🚀 Reproducibility Guide
### Requirements
Recommended Python 3.10+
```
pip install -r requirements.txt
```
### Environment
The `classes` folder contains all the files related to the implementation of the NPC (`agent.py`) and the game environment (`environment.py` - `game.py` - `inventory.py` - `magic.py`) for any changes to the settings defined in the article.

### Setup LLMs for Helper
To test LLMs for Helper, you need to install [LM Studio](https://lmstudio.ai/), enter the SERVER_API_HOST string and enter the name of the LLM to be tested in the string present in all training files in the `HeRoN` folder:
```
SERVER_API_HOST = "insert api"
model = client.llm.model("LLM_NAME")
```

### Reviewer
All files for training the Reviewer are located in the `reviewer` folder. To create your own dataset, refer to the `dataset Reviewer` folder. Once the Reviewer has been trained, you can use it in HeRoN files by inserting the tokenizer and model in the following strings
```
AutoTokenizer.from_pretrained(YOUR_MODEL_PATH) 
T5ForConditionalGeneration.from_pretrained(YOUR_MODEL_PATH)
```

### Training NPC
The configurations tested to train the NPC are located in the `HeRoN` folder. Once the LLM has been set up for Helper and the Reviewer model has been entered, change the names of the graphs in the `plot_training` function and the name of the CSV file relating to the success rate in the `export_success_rate` function and training can begin. Specifically, DQNAgent is the NPC and IntructorAgent is the Reviewer. The NPC model will be saved in keras format.

### Testing NPC
To test the trained NPC, use the `testing_model.py` file, enter the model name (i.e. ‘npc_model’), change the names of the graphs in the `plot_training` function, and start testing.

## Purpose
This repo is intended to serve as a foundation with which you can reproduce the results of the experiments detailed in our paper, recreate your own NPCs, and the ability to modify the JRPG environment.

## 📖 Citation
If you find our work helpful, we would appreciate if you cite it:
