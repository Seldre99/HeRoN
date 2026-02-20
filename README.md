# HeRoN: A Mediated RL–LLM Framework for Adaptive NPC Behavior in Interactive Environments 🎮
Non-Player Characters (NPCs) play a central role in modern video games, in fluencing both immersion and narrative depth. However, traditional design approaches, from rule-based systems to utility-driven AI, often fail to produce adaptive and contextually coherent behaviors. Recent progress in Reinforcement Learning (RL) and Large Language Models (LLMs) has opened new opportunities for improving NPC decision-making, but both face key limitations: RL struggles with training efficiency and generalization, while LLMs are prone to hallucinations and context drift. In this work,a mediated framework that integrates RL and LLMs through functional separation and critique-based mediation, enabling contextually coherent and strategically adaptive NPC behavior across interactive environments. HeRoN combines three components: (i) the NPC, an RL-driven agent whose policy is iteratively refined via LLM-generated critiques; (ii) the Helper, an LLM operating in zero-shot reasoning mode to generate diverse, context-aware action strategies; and (iii) the Reviewer, a lightweight, fine-tuned LLM that evaluates and refines the Helper’s suggestions, ensuring strategic consistency and alignment with game-specific constraints. We evaluate HeRoN in a custom turn-based battle environment, in an open-world survival environment ([Crafter](https://github.com/Daaanilo/IA2025_26_progetto/)) and in a first-person-shooting environment ([ViZDoom](https://github.com/rosariopiognazzo/DoomHeron)), demonstrating superior performance over standard RL baselines in strategy refinement, learning efficiency, adaptability, and contextual decision-making.

## Purpose
This repo is intended to serve as a foundation with which you can reproduce the results of the experiments detailed in our paper 

## Running Experiments
### Environment
The `classes` folder contains all the files related to the implementation of the NPC (`agent.py`) and the game environment (`environment.py` - `game.py` - `inventory.py` - `magic.py`) for any changes to the settings defined in the article.

### Reviewer
All files for training the Reviewer are located in the `reviewer` folder. To create your own dataset, refer to the `dataset Reviewer` folder. Once the Reviewer has been trained, you can use it in HeRoN files by inserting the tokeniser in the string `AutoTokenizer.from_pretrained()` and the model in the string `T5ForConditionalGeneration.from_pretrained`.

### Setup LLMs for Helper
To test LLMs for Helper, you need to install [LM Studio](https://lmstudio.ai/), enter the SERVER_API_HOST string and enter the name of the LLM to be tested in the string `model = client.llm.model(‘’)` present in all training files in the `HeRoN` folder.

### Training NPC
The configurations tested to train the NPC are located in the `HeRoN` folder. Once the LLM has been set up for Helper and the Reviewer model has been entered, change the names of the graphs in the `plot_training` function and the name of the CSV file relating to the success rate in the `export_success_rate` function and training can begin. Specifically, DQNAgent is the NPC and IntructorAgent is the Reviewer. The NPC model will be saved in keras format.

### Testing NPC
To test the trained NPC, use the `testing_model.py` file, enter the model name (i.e. ‘npc_model’) in the DQNAgent string, change the names of the graphs in the `plot_training` function, and start testing.

### Citation
If you find our work helpful, we would appreciate if you cite it:
