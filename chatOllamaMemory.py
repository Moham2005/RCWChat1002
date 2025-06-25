from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time

template = """
Tu es un professeur de sciences au lycée. Sois clair, concis et pédagogue.
Explique les réponses en utilisant des exemples simples et un langage adapté aux élèves.

Voici l’historique de la conversation : {context}

Question : {question}

Réponse :
"""

model = OllamaLLM(model='mistral:latest')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Bonjour, je suis Mohamed. Comment puis-je vous aider ? (Tapez 'quit' pour quitter)")

    while True:
        user_input = input("\nVous : ")
        if user_input.lower() == "quit":
            print("Au revoir ! 👋")
            break

        inputs = {
            "context": context,
            "question": user_input
        }

        start = time.time()
        response = chain.invoke(inputs)
        end = time.time()

        print("Assistant :", end=" ", flush=True)
        for char in response:
            print(char, end="", flush=True)
            time.sleep(0.02)
        print(f"\n⏱️ Temps de traitement : {end - start:.2f} secondes")

        context += f"\nQuestion : {user_input}\nRéponse : {response}"


if __name__ == "__main__":
    handle_conversation()
