from langchain_ollama import OllamaLLM
import time

model = OllamaLLM(model='llama3.2')

user_input = 'Quelle est la capitale de Maroc ?'

start = time.time()
response = model.invoke(user_input)
end = time.time()

for char in response:
    print(char, end="", flush=True)
    time.sleep(0.02)

print()
print(f"⏱️ Temps de traitement : {end - start:.2f} secondes")
