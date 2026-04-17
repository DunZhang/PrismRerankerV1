import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/rerank",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "cohere/rerank-4-fast",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "London is the capital of England.",
      "Berlin is the capital of Germany."
    ],
    "top_n": 3
  })
)

results = response.json()
for result in results["results"]:
  print(f"Index: {result['index']}, Score: {result['relevance_score']}")
  print(f"  Document: {result['document']['text']}")