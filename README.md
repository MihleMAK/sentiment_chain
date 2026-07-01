# Sentiment Chain Moview Review Analysis

This is a sentiment analysis pipeline for movie reviews that:
- runs entirely on a **local, open-weight model** via Ollama instead of
  a paid cloud API,
- classifies reviews from the Stanford IMDB dataset as **Positive** or
  **Negative** using two different system prompts.
- is evaluated for accuracy against the dataset's ground-truth labels
- characterizes *how* each sentiment is expressed, by surfacing the
  adjectives most associated with positive vs. negative reviews.

  The descriptive system prompt with few-shot examples(v2) resulted in a higher accuracy percentage.

  ## Resources used
  - Langchain
  - Ollama using llama3.1 model
  - Hugging face dataset


## Constraints & Solutions

| # | Constraint | Why it's a problem | Solution |
|---|---|---|---|
| 1 | Must run fully locally, no cloud API key | Azure OpenAI is paid and requires credentials
| 2 | Local models are smaller than GPT-4.1 | More prone to adding preambles ("Sure, the answer is..."), hedging ("mostly positive"), or inconsistent casing/punctuation, which breaks exact-match grading | A strict prompt that demands a single-word, no-punctuation answer, plus a small normalization function that maps near-miss outputs (e.g. "Positive." or "POSITIVE") back to a clean label |
| 3 | Ollama's default context window is small (often 2048 tokens) | IMDB reviews can run 500–1000+ words; if a review and system prompt exceeds the context window, it gets silently truncated and the model judges an incomplete review | Explicitly set `num_ctx` (e.g. 8192) on the Ollama model so the full review always fits |
| 4 | Local models struggle more with nuance (sarcasm, mixed-sentiment reviews) | A review can criticize most aspects of a movie but still conclude positively (or vice versa); naive prompts latch onto surface word counts rather than the reviewer's actual verdict | Add a couple of few-shot examples that demonstrate weighing the final verdict, and ask for 1–2 sentences of reasoning before the label (lets the model "think" before committing) |


## Setup

### 1. Install Ollama
Download and install from https://ollama.com/download.

### 2. Pull a model
```bash
ollama pull llama3.1
```
(Swap in any tool-calling/JSON-schema-capable model you prefer — `llama3.1`, `qwen2.5`,
`mistral-nemo`, etc. Larger models generally classify more accurately.)


### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure the script
At the top of `imdb_sentiment_chains.py`, set:
```python
OLLAMA_MODEL = "llama3.1"   # match whatever you pulled in step 2
```
