# HeRoN: A Multi Agent RL-LLM Framework for Adaptive NPC Decision Making
Non-Player Characters (NPCs) play a central role in modern video games, in fluencing both immersion and narrative depth. However, traditional design approaches, from rule-based systems to utility-driven AI, often fail to produce adaptive and contextually coherent behaviors. Recent progress in Reinforcement Learning (RL) and Large Language Models (LLMs) has opened new opportunities for improving NPC decision-making, but both face key limitations: RL struggles with training efficiency and generalization, while LLMs are prone to hallucinations and context drift. In this work, we introduce HeRoN, a multi-agent architecture that integrates RL and LLMs to produce NPCs with more strategic and contextually relevant behaviors. HeRoN combines three components: (i) the NPC, an RL-driven agent whose policy is iteratively refined via LLM-generated critiques; (ii) the Helper, an LLM operating in zero-shot reasoning mode to generate diverse, context-aware action strategies; and (iii) the Reviewer, a lightweight, fine-tuned LLM that evaluates and refines the Helper’s
suggestions, ensuring strategic consistency and alignment with game-specific constraints. We evaluate HeRoN in a custom turn-based battle environment, demonstrating superior performance over standard RL baselines in strategy refinement, learning efficiency, adaptability, and contextual decision-making.

## Purpose
This repo is intended to serve as a foundation with which you can reproduce the results of the experiments detailed in our paper 

## Running Experiments
*Environment*
Nella cartella `classes` sono presenti tutti i file relativi all'implementazione dell'NPC (agent.py), e dell'environment di gioco (environment.py - game.py - inventory.py - magic.py) per eventuali modifiche ai settaggi definiti nell'articolo.

*Reviewer*
Tutti i file per addestrare il Reviewer sono presenti nella cartella `reviewer`, per creare il proprio dataset consultare la cartella `dataset Reviewer`. Una volta addestrato il Reviewer, è possibile utilizzarlo nei file HeRoN inserendo il tokenizer nella stringa `AutoTokenizer.from_pretrained()`.

**Setup LLMs for Helper**
Per testare gli LLM per Helper, è necessario installare ([LM Studio](https://lmstudio.ai/)), inserire la stringa di SERVER_API_HOST ed inserire il nome del LLM da testare nella stringa  `model = client.llm.model("")` presente in tutti i file di training della cartella `HeRoN`.

**Training NPC**
Le configurazioni testate per addestrare l'NPC sono presenti nella cartella `HeRoN`

**Testing NPC**


## Citation
If you find our work helpful, we would appreciate if you cite it:
