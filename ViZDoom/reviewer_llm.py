#!/usr/bin/env python3
"""
HERON Reviewer LLM.

"""

import gc
import json
import os
import re
from collections import deque


# Action name aliases for normalization
ACTION_ALIASES = {
    'SHOOT': 'ATTACK',
    'FIRE': 'ATTACK',
    'LEFT': 'MOVE_LEFT',
    'RIGHT': 'MOVE_RIGHT',
    'FORWARD': 'MOVE_FORWARD',
    'BACKWARD': 'MOVE_BACKWARD'
}


class ReviewerLLM:
    """
    Reviewer LLM with episodic memory.

    Workflow:
    1. Helper suggests plan: ["ATTACK", "TURN_LEFT"]
    2. Reviewer analyzes: current state, helper plan, history
    3. Reviewer outputs: corrected/improved plan
    """

    def __init__(self, model_path=None, quantization="8bit", history_size=5, device=None):
        """
        Initialize the Reviewer LLM.

        Args:
            model_path: Path to fine-tuned model or .gguf file.
                       Falls back to REVIEWER_MODEL_PATH env var if None.
            quantization: "fp16", "8bit", or "4bit" (ignored for GGUF)
            history_size: Number of steps to keep in memory
            device: torch.device or None (auto-detect)
        """
        self.model_path = model_path or os.getenv(
            'REVIEWER_MODEL_PATH',
            './models/reviewer.gguf'
        )
        self.quantization = quantization
        self.history_size = history_size
        self.is_gguf = self._is_gguf_model(self.model_path)

        if device is None and not self.is_gguf:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.is_gguf:
            self.device = "cuda"
        else:
            self.device = device

        self.history = deque(maxlen=history_size)
        self._load_model()

        self.total_reviews = 0
        self.total_corrections = 0
        self.total_agreements = 0

    def _is_gguf_model(self, path):
        """Check if path contains a GGUF file."""
        if os.path.isfile(path) and path.endswith('.gguf'):
            return True
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.gguf'):
                    return True
        return False

    def _load_model(self):
        """Load the model with specified quantization."""
        if self.is_gguf:
            self._load_gguf_model()
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        if self.quantization == "fp16":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        elif self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Invalid quantization: {self.quantization}")

        self.model.eval()

    def _load_gguf_model(self):
        """Load GGUF model with llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        if os.path.isfile(self.model_path):
            gguf_file = self.model_path
        else:
            gguf_files = [f for f in os.listdir(self.model_path) if f.endswith('.gguf')]
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF file found in {self.model_path}")
            gguf_file = os.path.join(self.model_path, gguf_files[0])

        self.model = Llama(
            model_path=gguf_file,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False
        )
        self.tokenizer = None

    def reset_episode(self):
        """Reset history for a new episode."""
        self.history.clear()

    def update_history(self, state_desc, action, reward):
        """
        Add a step to episodic history.

        Args:
            state_desc: Description of game state
            action: Executed action name
            reward: Received reward
        """
        self.history.append({
            'state': state_desc,
            'action': action,
            'reward': reward
        })

    def review_plan(self, state_desc, helper_plan, available_actions, scenario="doom"):
        """
        Review and correct the Helper's suggested plan.

        Args:
            state_desc: Current state description (Oracle)
            helper_plan: Helper's suggested plan, e.g., ["ATTACK", "TURN_LEFT"]
            available_actions: List of valid actions
            scenario: Scenario name for additional context

        Returns:
            Corrected plan (list of actions)
        """
        self.total_reviews += 1

        prompt = self._build_review_prompt(state_desc, helper_plan, available_actions, scenario)
        reviewer_response = self._query_llm(prompt)
        corrected_plan = self._parse_plan(reviewer_response, available_actions)

        if not corrected_plan:
            corrected_plan = helper_plan

        if corrected_plan == helper_plan:
            self.total_agreements += 1
        else:
            self.total_corrections += 1

        return corrected_plan

    def _build_review_prompt(self, state_desc, helper_plan, available_actions, scenario):
        """Build the review prompt including history context."""
        history_text = self._format_history()
        helper_plan_str = json.dumps(helper_plan)
        actions_str = ', '.join(available_actions)

        user_prompt = f"""SCENARIO: {scenario}

CURRENT STATE:
{state_desc}

HELPER SUGGESTION:
{helper_plan_str}

RECENT HISTORY (Last {len(self.history)} steps):
{history_text}

AVAILABLE ACTIONS: {actions_str}

TASK: Review the Helper's plan. Consider:
1. Does it align with successful patterns from history?
2. Are there better alternatives given recent rewards?
3. Should we continue the current strategy or pivot?

OUTPUT: JSON array of corrected actions (max 5).
FORMAT: ["ACTION1", "ACTION2", ...]
"""

        return [
            {
                "role": "system",
                "content": "You are a Tactical Reviewer for a Doom agent. Analyze suggestions and correct them based on episodic memory and game state."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

    def _format_history(self):
        """Format history for the prompt."""
        if not self.history:
            return "No history yet (episode start)"

        lines = []
        for i, step in enumerate(self.history, 1):
            reward_indicator = "+" if step['reward'] > 0 else "-" if step['reward'] < 0 else "0"
            lines.append(f"  Step -{len(self.history)-i+1}: {step['action']} -> Reward: {step['reward']:.1f} [{reward_indicator}]")

        return "\n".join(lines)

    def _query_llm(self, messages):
        """
        Query the local LLM model.

        Args:
            messages: Chat format messages (system, user, assistant)

        Returns:
            Model response string
        """
        if self.is_gguf:
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=100,
                temperature=0.5,
                top_p=0.9,
                stop=["User:", "\n\n"]
            )
            return response['choices'][0]['message']['content']

        import torch

        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    prompt += f"System: {content}\n\n"
                elif role == 'user':
                    prompt += f"User: {content}\n\n"
            prompt += "Assistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            response = full_response

        for marker in ["assistant:", "Assistant:", "ASSISTANT:"]:
            if marker in response:
                parts = response.split(marker)
                response = parts[-1].strip()
                break

        return response

    def _parse_plan(self, response, valid_actions):
        """
        Parse the plan from Reviewer's response.

        Args:
            response: Model output string
            valid_actions: List of valid actions

        Returns:
            List of actions or [] if parsing fails
        """
        try:
            clean = re.sub(r"```json|```", "", response).strip()
            match = re.search(r'\[.*?\]', clean, re.DOTALL)
            if match:
                clean = match.group(0)

            actions = json.loads(clean)
            normalized = []

            for a in actions:
                a_up = str(a).upper().strip()
                a_up = ACTION_ALIASES.get(a_up, a_up)

                if a_up in valid_actions:
                    normalized.append(a_up)

            return normalized[:5]

        except Exception:
            return []

    def get_stats(self):
        """
        Get Reviewer statistics.

        Returns:
            dict with review statistics
        """
        agreement_rate = (self.total_agreements / self.total_reviews) if self.total_reviews > 0 else 0.0

        return {
            'total_reviews': self.total_reviews,
            'total_corrections': self.total_corrections,
            'total_agreements': self.total_agreements,
            'agreement_rate': agreement_rate
        }

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print(f"Reviewer Statistics")
        print(f"{'='*60}")
        print(f"  Total Reviews: {stats['total_reviews']}")
        if stats['total_reviews'] > 0:
            print(f"  Corrections: {stats['total_corrections']} ({stats['total_corrections']/stats['total_reviews']*100:.1f}%)")
            print(f"  Agreements: {stats['total_agreements']} ({stats['agreement_rate']*100:.1f}%)")
        else:
            print(f"  Corrections: 0 (N/A)")
            print(f"  Agreements: 0 (N/A)")
        print(f"{'='*60}\n")

    def clear_cache(self):
        """Free GPU memory."""
        if not self.is_gguf:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            except:
                pass


if __name__ == "__main__":
    """Standalone test for ReviewerLLM."""
    model_path = os.getenv('REVIEWER_MODEL_PATH', './models/reviewer.gguf')

    try:
        reviewer = ReviewerLLM(
            model_path=model_path,
            quantization="8bit",
            history_size=5
        )

        state_desc = """
HP: 75/100
Ammo: 12/50
Enemies: 2 in center (distance: medium)
Crosshair: Locked on enemy
"""

        helper_plan = ["MOVE_FORWARD", "MOVE_FORWARD", "ATTACK"]

        reviewer.update_history("HP: 100, Ammo: 50", "ATTACK", 1.0)
        reviewer.update_history("HP: 85, Ammo: 49", "MOVE_FORWARD", 0.0)
        reviewer.update_history("HP: 75, Ammo: 48", "ATTACK", 1.0)

        corrected_plan = reviewer.review_plan(
            state_desc=state_desc,
            helper_plan=helper_plan,
            available_actions=["ATTACK", "MOVE_FORWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT"],
            scenario="defend_the_center"
        )

        print(f"Helper Plan: {helper_plan}")
        print(f"Reviewer Output: {corrected_plan}")

        reviewer.print_stats()
        reviewer.clear_cache()

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
