from openai import OpenAI
import os
import csv
import time
import json



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


data = []
with open("/results/gpt-5-mini_inContext_t4.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))


def evaluate_response(question, gold_answer, model_answer, tolerance="1pct"):
    if tolerance == "1pct":
        prompt = f"""
You are evaluating responses from an AI model on Financial Benchmark questions.

Please classify the response as CORRECT, INCORRECT, or FAILED_TO_ANSWER with ±1% tolerance for numerical answers.

Question: {question}

Gold Answer: {gold_answer}

Model Answer: {model_answer}

CLASSIFICATION RULES (±1% TOLERANCE):
1. CORRECT: 
   - Exact match with gold answer
   - Semantically equivalent (e.g., "2.9" vs "2.9%")
   - For numerical answers: within ±1% of correct value
   - Different formatting but same meaning
   - Correct text spans extracted

2. INCORRECT: 
   - Factually wrong specific answer
   - Wrong numbers outside ±1% tolerance
   - Wrong text spans

3. FAILED_TO_ANSWER:
   - Model refused to answer
   - Said it lacks information/data
   - Non-specific response like "I cannot determine"

Respond with exactly one word: CORRECT, INCORRECT, or FAILED_TO_ANSWER

Your classification:"""
    elif tolerance == "0pct":
        prompt = f"""
You are evaluating responses from an AI model on Financial Benchmark questions.

Please classify the response as CORRECT, INCORRECT, or FAILED_TO_ANSWER with ±0% tolerance (EXACT MATCH only).

Question: {question}

Gold Answer: {gold_answer}

Model Answer: {model_answer}

CLASSIFICATION RULES (±0% EXACT MATCH):
1. CORRECT: 
   - EXACT match with gold answer
   - Same text, same numbers, same format
   - No tolerance for numerical differences

2. INCORRECT: 
   - Any factual difference from gold answer
   - Any numerical difference, even tiny ones
   - Different formatting or wording

3. FAILED_TO_ANSWER:
   - Model refused to answer
   - Said it lacks information/data
   - Non-specific response like "I cannot determine"

Respond with exactly one word: CORRECT, INCORRECT, or FAILED_TO_ANSWER

Your classification:"""
    else:
        raise ValueError("tolerance must be '1pct' or '0pct'")

   
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text.strip()


output_file = "eval_results_mini3.csv"

with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["question", "gold_answer", "model_answer", "score_1pct", "score_0pct"])
    writer.writeheader()

    for item in data:
        q = item["question"]
        gold = item["gold_answer"]
        model_ans = item["model_answer"]

        # ±1% 
        score_1pct = evaluate_response(q, gold, model_ans, tolerance="1pct")
        # ±0% 
        score_0pct = evaluate_response(q, gold, model_ans, tolerance="0pct")

        writer.writerow({
            "question": q,
            "gold_answer": gold,
            "model_answer": model_ans,
            "score_1pct": score_1pct,
            "score_0pct": score_0pct
        })

        
        time.sleep(1)

print(f"saved to {output_file}")
