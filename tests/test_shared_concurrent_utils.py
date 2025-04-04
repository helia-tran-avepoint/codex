import ollama

from shared.concurrent_utils import run_jobs_multiprocessing


def query_ollama(prompt: str):
    response = ollama.chat(
        model="deepseek-llm-32b", messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


prompts = ["Summarize this text"] * 10
jobs = [lambda p=prompt: query_ollama(p) for prompt in prompts]

results = run_jobs_multiprocessing(
    jobs, show_progress=True, workers=4, desc="Running LLM jobs"
)
print(results)
