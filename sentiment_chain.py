"""
IMDB Sentiment Analysis with LangChain + Ollama:
Classifies movie reviews as Positive/Negative.
"""

import random
import re
import time
from collections import Counter
from datetime import datetime
from typing import Literal
import os
import logging

from pydantic import BaseModel, Field
from datasets import load_dataset

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser


# Logging — writes to both the terminal and a timestamped file in logs/

LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"imdb_sentiment_{datetime.now():%Y%m%d_%H%M%S}.log")

logger = logging.getLogger("imdb_sentiment")
logger.setLevel(logging.INFO)

_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
logger.addHandler(_console_handler)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)

logger.propagate = False
logger.info(f"Log file: {LOG_FILE}")


#  Model setup

OLLAMA_MODEL = "llama3.1"

logger.info(f"Configuring model '{OLLAMA_MODEL}'...")
model = init_chat_model(
    model=OLLAMA_MODEL,
    model_provider="ollama",
    temperature=0,
    num_ctx=8192,
    num_predict=300,
)
logger.info("Model ready.")

parser = StrOutputParser()


# Sentiment classification prompts

SENTIMENT_PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", "Classify the sentiment of the following movie review as Positive or Negative."),
    ("user", "{review}"),
])

SENTIMENT_PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert sentiment classifier for movie reviews.\n"
     "Judge the reviewer's OVERALL FINAL verdict on the movie as a whole — not "
     "just which words appear more often. Reviews often mix praise and "
     "criticism, use sarcasm, or spend more words complaining even when the "
     "final verdict is positive (or vice versa). Weigh the conclusion the "
     "reviewer actually reaches.\n\n"
     "Example 1:\n"
     "Review: \"The plot dragged in the middle and the dialogue was clunky in "
     "places, but the lead performance was so magnetic and the ending so "
     "satisfying that I walked out grinning. Worth it.\"\n"
     "Reasoning: Despite criticism of pacing and dialogue, the reviewer's "
     "final verdict is clearly enthusiastic.\n"
     "Label: Positive\n\n"
     "Example 2:\n"
     "Review: \"Gorgeous cinematography, a great score, and a talented cast — "
     "all wasted on a story that goes nowhere and a runtime that overstays "
     "its welcome. I wouldn't recommend it.\"\n"
     "Reasoning: Despite praising the visuals and cast, the reviewer "
     "explicitly does not recommend the movie.\n"
     "Label: Negative\n\n"
     "Now classify the review below. Give 1-2 sentences of reasoning, then "
     "your final label."),
    ("user", "{review}"),
])


class SentimentResult(BaseModel):
    reasoning: str = Field(description="1-2 sentence justification for the label")
    label: Literal["Positive", "Negative"]


def make_text_chain(prompt_template: ChatPromptTemplate):
    return prompt_template | model | parser


def make_structured_chain(prompt_template: ChatPromptTemplate):
    return prompt_template | model.with_structured_output(SentimentResult)


def normalize_label(raw: str) -> str:
    match = re.search(r"label\s*:\s*(positive|negative)", raw, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    cleaned = raw.strip().lower()
    last_pos, last_neg = cleaned.rfind("positive"), cleaned.rfind("negative")
    if last_pos == -1 and last_neg == -1:
        return cleaned.capitalize() or "Unknown"
    return "Positive" if last_pos > last_neg else "Negative"



# Dataset loading

def load_balanced_sample(n_per_class: int = 20, split: str = "train", seed: int = 42):
    logger.info(f"Loading IMDB '{split}' split from Hugging Face...")
    t0 = time.perf_counter()
    ds = load_dataset("stanfordnlp/imdb", split=split).shuffle(seed=seed)
    logger.info(f"Loaded {len(ds)} reviews in {time.perf_counter() - t0:.1f}s. Filtering sample...")

    negatives = ds.filter(lambda ex: ex["label"] == 0).select(range(n_per_class))
    positives = ds.filter(lambda ex: ex["label"] == 1).select(range(n_per_class))
    examples  = [{"text": ex["text"], "label": "Negative"} for ex in negatives]
    examples += [{"text": ex["text"], "label": "Positive"} for ex in positives]
    random.Random(seed).shuffle(examples)

    logger.info(f"Sample ready: {len(examples)} reviews ({n_per_class} pos / {n_per_class} neg)")
    return examples



# Evaluation

def evaluate(predict_fn, examples, name="chain"):
    correct, mistakes = 0, []
    total = len(examples)
    logger.info(f"[{name}] Starting evaluation on {total} examples...")

    for i, ex in enumerate(examples, start=1):
        t0     = time.perf_counter()
        pred   = predict_fn(ex["text"])
        elapsed = time.perf_counter() - t0
        ok     = pred == ex["label"]
        correct += int(ok)

        status = "ok" if ok else "MISS"
        logger.info(
            f"[{name}] {i}/{total}  true={ex['label']:<8s}  pred={pred:<8s}  "
            f"({elapsed:.1f}s)  {status}"
        )
        if not ok:
            mistakes.append({"true": ex["label"], "pred": pred})

    accuracy = correct / total
    logger.info(f"[{name}] Done — accuracy {accuracy:.2%}  ({correct}/{total} correct)")
    return accuracy, mistakes



# Adjective extraction

list_parser = CommaSeparatedListOutputParser()

adjective_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract every adjective in the review below that describes the movie, "
     "the acting, or the story (e.g. bad, perfect, dreamy, boring).\n"
     "{format_instructions}\n"
     "Return ONLY the comma-separated list — no numbering, no extra commentary."),
    ("user", "{review}"),
]).partial(format_instructions=list_parser.get_format_instructions())

adjective_chain = adjective_prompt | model | list_parser


def aggregate_adjectives_by_sentiment(examples):
    counts = {"Positive": Counter(), "Negative": Counter()}
    total  = len(examples)
    logger.info(f"[adjectives] Extracting adjectives from {total} reviews...")

    for i, ex in enumerate(examples, start=1):
        t0   = time.perf_counter()
        adjs = adjective_chain.invoke({"review": ex["text"]})
        adjs = [a.strip().lower() for a in adjs if a.strip()]
        counts[ex["label"]].update(adjs)
        logger.info(
            f"[adjectives] {i}/{total}  ({ex['label']:<8s})  "
            f"{len(adjs)} adjective(s) found  ({time.perf_counter() - t0:.1f}s)"
        )

    logger.info("[adjectives] Done.")
    return counts



# Run everything

if __name__ == "__main__":
    examples = load_balanced_sample(n_per_class=20)

    print("=" * 70)
    print("SENTIMENT CLASSIFICATION ACCURACY (20 pos / 20 neg)")
    print("=" * 70)

    # v1 — basic prompt, plain text output
    logger.info("Building v1 chain...")
    chain_v1 = make_text_chain(SENTIMENT_PROMPT_V1)
    acc_v1, _ = evaluate(
        lambda t: normalize_label(chain_v1.invoke({"review": t})),
        examples,
        name="v1",
    )
    print(f"v1 (basic prompt):                {acc_v1:.2%}")

    # v2 — few-shot + chain-of-thought + structured output
    logger.info("Building v2 chain...")
    chain_v2 = make_structured_chain(SENTIMENT_PROMPT_V2)
    acc_v2, mistakes_v2 = evaluate(
        lambda t: chain_v2.invoke({"review": t}).label,  
        examples,
        name="v2",
    )
    print(f"v2 (few-shot + CoT + structured): {acc_v2:.2%}")

    if mistakes_v2:
        print(f"\n{len(mistakes_v2)} misclassified example(s) with v2:")
        for m in mistakes_v2:
            print(f"  true={m['true']:<8s}  pred={m['pred']}")
    else:
        print("\nv2 reached 100% accuracy on this sample.")

    print()
    print("=" * 70)
    print("ADJECTIVES BY SENTIMENT CLASS")
    print("=" * 70)
    adjective_counts = aggregate_adjectives_by_sentiment(examples)

    for sentiment in ["Positive", "Negative"]:
        print(f"\nTop adjectives in {sentiment} reviews:")
        for adj, count in adjective_counts[sentiment].most_common(15):
            print(f"  {adj:20s} {count}")

    logger.info("All done.")

    



