import ollama
import time

t0 = time.time()
try:
    res = ollama.generate(model='phi3', prompt='Output valid JSON: {"test": "success"}')
    print("Ollama response in", round(time.time() - t0, 2), "s:")
    print(res['response'])
except Exception as e:
    print("Error calling ollama:", e)
