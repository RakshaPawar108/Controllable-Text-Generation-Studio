from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Literal, Optional, Dict, Any

Style = Literal["formal", "casual", "enthusiastic", "sarcastic", "poetic", "neutral"]

# Prompt templates for different tones
STYLE_PROMPTS: Dict[Style, str] = {
    "formal": (
        "You are a writing assistant that rewrites sentences in a formal and professional tone "
        "while keeping the original meaning and sentiment.\n\n"
        "Example 1:\n"
        "Input: I'm super happy with how this turned out.\n"
        "Formal: I am very pleased with the outcome of this.\n\n"
        "Example 2:\n"
        "Input: The service was really slow and annoying.\n"
        "Formal: The service was quite slow and somewhat frustrating.\n\n"
        "Task: Rewrite the following sentence in a formal and professional tone. "
        "Keep the same meaning and sentiment. Do not copy it word for word.\n\n"
        "Sentence: {text}\n"
        "Formal rewrite:"
    ),
    "casual": (
        "You are a writing assistant that rewrites sentences in a casual and friendly tone, "
        "like you are talking to a friend, while keeping the same meaning and sentiment.\n\n"
        "Example 1:\n"
        "Input: I am very pleased with the outcome of this.\n"
        "Casual: I'm really happy with how this turned out.\n\n"
        "Example 2:\n"
        "Input: The meeting has been postponed until further notice.\n"
        "Casual: The meeting got pushed back for now.\n\n"
        "Task: Rewrite the following sentence in a casual and friendly tone. "
        "Keep the same meaning and sentiment. Do not copy it word for word.\n\n"
        "Sentence: {text}\n"
        "Casual rewrite:"
    ),
    "enthusiastic": (
        "You are a writing assistant that rewrites sentences in an enthusiastic and energetic tone. "
        "You should keep the overall opinion and sentiment of the original sentence. "
        "If the opinion is negative, the rewrite must still clearly be negative. "
        "If the opinion is positive, the rewrite must still clearly be positive.\n\n"
        "Example 1 (positive):\n"
        'Input: The event went well and I liked it.\n'
        "Enthusiastic: The event went amazingly well and I absolutely loved it.\n\n"
        "Example 2 (negative):\n"
        "Input: The experience was disappointing.\n"
        "Enthusiastic: The whole experience was wildly disappointing and I definitely would not repeat it.\n\n"
        "Task: Rewrite the following sentence in an enthusiastic and energetic tone. "
        "Keep the same overall opinion and sentiment. Do not turn a negative opinion into a positive one "
        "or a positive opinion into a negative one. Do not copy it word for word.\n\n"
        "Sentence: {text}\n"
        "Enthusiastic rewrite:"
    ),
    "sarcastic": (
        "You are a writing assistant that rewrites sentences with clear written sarcasm. "
        "The sarcasm should be understandable from text alone, without voice or facial expression. "
        "You must keep the same overall opinion and sentiment as the original sentence.\n\n"
        "Example 1 (negative):\n"
        "Input: The service was terrible and I would not come back.\n"
        "Sarcastic: The service was such an unforgettable delight that I cannot wait to never come back.\n\n"
        "Example 2 (negative):\n"
        "Input: The software crashed all the time.\n"
        "Sarcastic: The software is so stable that it crashes whenever I even look at it.\n\n"
        "Task: Rewrite the following sentence with clear, negative sarcasm if the opinion is negative, "
        "or clear sarcastic exaggeration if the opinion is positive. "
        "The reader should easily see the sarcasm from the wording alone. "
        "Do not remove or flip the sentiment of the original sentence. "
        "Do not copy it word for word.\n\n"
        "Sentence: {text}\n"
        "Sarcastic rewrite:"
    ),
    "poetic": (
        "You are a writing assistant that rewrites sentences in a poetic and descriptive style, "
        "while keeping the core meaning and sentiment.\n\n"
        "Example 1:\n"
        "Input: The room was very quiet.\n"
        "Poetic: The room rested in a hush so deep that even the air seemed to hold its breath.\n\n"
        "Example 2:\n"
        "Input: The journey was difficult but worth it.\n"
        "Poetic: The road was steep and stubborn, yet every hard won step carried its own quiet reward.\n\n"
        "Task: Rewrite the following sentence in a poetic and descriptive style. "
        "Keep the same core meaning and sentiment. Do not copy it word for word.\n\n"
        "Sentence: {text}\n"
        "Poetic rewrite:"
    ),
    "neutral": (
        "You are a writing assistant that rewrites sentences in a clear, neutral, and balanced tone. "
        "You remove emotional language but keep the factual meaning and overall stance.\n\n"
        "Example 1:\n"
        "Input: This product is absolutely terrible and I hate it.\n"
        "Neutral: This product does not meet my expectations.\n\n"
        "Example 2:\n"
        "Input: This is the best service I have ever used.\n"
        "Neutral: This service works very well for my needs.\n\n"
        "Task: Rewrite the following sentence in a neutral and balanced tone. "
        "Keep the same basic meaning and stance, but avoid strong emotional words. "
        "Do not copy it word for word.\n\n"
        "Sentence: {text}\n"
        "Neutral rewrite:"
    ),
}


class Phi2Generator:
    def __init__(self, max_new_tokens: int = 80, device: Optional[int] = None):
        print("Loading microsoft/phi-2 model… this may take a minute.")

        self.model_name = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device if device is not None else -1,
        )

        self.max_new_tokens = max_new_tokens

    def build_prompt(self, text: str, style: Style) -> str:
        template = STYLE_PROMPTS[style]
        return template.format(text=text)

    def generate(
        self,
        text: str,
        style: Style = "neutral",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a styled rewrite with microsoft/phi-2 and strip extra
        artifacts like 'Input:' / 'Output:' or fake examples,
        while keeping the full rewritten text (even if it spans lines).
        """
        prompt = self.build_prompt(text, style)
        max_tokens = max_new_tokens or self.max_new_tokens

        outputs = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        full = outputs[0]["generated_text"]

        # Remove the prompt portion at the beginning if the model echoed it
        if full.startswith(prompt):
            gen = full[len(prompt):]
        else:
            gen = full

        # Markers that usually mean the model started adding meta stuff
        junk_markers = [
            "\nInput:",
            "\nINPUT:",
            "Input:",
            "INPUT:",
            "\nOutput:",
            "\nOUTPUT:",
            "Output:",
            "OUTPUT:",
            "## INPUT",
            "##INPUT",
            "## OUTPUT",
            "##OUTPUT",
            "Exercise",
            "EXERCISE",
            "Example 3",
            "Example 2",
            "Example 1",
        ]

        # Cut off everything after the earliest junk marker (if any)
        cutoff_index = len(gen)
        for marker in junk_markers:
            idx = gen.find(marker)
            if idx != -1 and idx < cutoff_index:
                cutoff_index = idx

        gen = gen[:cutoff_index].strip()

        return gen


# CLI for testing
if __name__ == "__main__":
    gen = Phi2Generator()

    while True:
        print("\nEnter text (blank to quit):")
        text = input("> ").strip()
        if not text:
            break

        print("Choose style: formal, casual, enthusiastic, sarcastic, poetic, neutral")
        style = input("> ").strip().lower()
        if style not in STYLE_PROMPTS:
            style = "neutral"

        output = gen.generate(text, style)  # type: ignore
        print(f"\n[{style.upper()} VERSION]")
        print(output)
