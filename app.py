import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(page_title="RCW Chat IA", layout="centered")
st.title("🤖 Streaming Chatbot avec LangChain + OpenAI")

# Fonction pour récupérer la réponse de l'IA
def retrieve_response(user_input, chat_history):
    template = """
    Tu es un professeur de sciences au lycée. Sois clair, concis et pédagogue.
    Explique les réponses en utilisant des exemples simples et un langage adapté aux élèves.

    Voici l’historique de la conversation : {chat_history}
    Question : {user_question}

    Réponse :
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o", max_tokens=2500)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_input
    })

# Initialiser l'historique de conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Bonjour Je suis Mohamed! Pose-moi ta question 🧪")
    ]

# Afficher les messages précédents
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)






# Saisie utilisateur
user_query = st.chat_input("Écris ta question ici...")

if user_query is not None and user_query != "":
    # Ajouter le message utilisateur
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # Affichage réponse IA en streaming
    with st.chat_message("AI"):
        response_stream = retrieve_response(
            user_query,
            [f"{'👤' if isinstance(m, HumanMessage) else '🤖'}: {m.content}" for m in st.session_state.chat_history]
        )

        full_response = ""
        response_area = st.empty()
        for chunk in response_stream:
            full_response += chunk
            response_area.markdown(full_response + "▌")
        response_area.markdown(full_response)

    # Ajouter la réponse de l'IA à l'historique
    st.session_state.chat_history.append(AIMessage(content=full_response))
