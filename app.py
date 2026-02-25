from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ollama

# Structure de la requête
class ChatRequest(BaseModel):
    message: str
    sessionId: str

app = FastAPI()

# Autoriser ton futur frontend à communiquer avec le backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_memories = {}

# --- TON PROFIL PERSONNALISÉ ---
PROFILE_TEXT = """
Identité : Dossou Borice, Data Analyst et Data Scientist en Master Mathématiques appliquées (UBS Vannes).
Contact : boricefiacredossou@gmail.com | 07 45 62 80 50.
Expertise : Analyse de données, Machine Learning (Séries temporelles, NLP), et Data Engineering (ETL).

Expériences clés :
1. EHESP (Rennes) : Conception de dashboards RH sous Power BI et automatisation de la qualité des données avec Talend.
2. CCI Morbihan : Pipeline ETL sous R et modélisation prédictive du risque de liquidation.
3. BB Players : Pilotage d'enquêtes statistiques et analyse de données.

Compétences Techniques : 
- Langages : Python, R, SQL, SAS.
- Outils : Power BI, Tableau, Talend, Azure Data Factory, Dataiku.
- Projets phares : Application R Shiny de prévision de ventes, Agent IA automatisé avec n8n, Pipeline MovieLens avec FastAPI.

Personnalité : Amical, dynamique, autonome et très curieux. Adore le football et s'engage bénévolement à la Croix-Rouge.
"""

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Tu es l'assistant virtuel de Dossou Borice. "
        "Ton ton est amical, dynamique et professionnel. "
        "Réponds aux questions en utilisant les informations suivantes :\n"
        + PROFILE_TEXT +
        "\nSi une question sort de ce cadre, réponds poliment que tu ne sais pas mais propose de contacter Borice directement."
    )
}

def get_ollama_response(message: str, history: list):
    try:
        # On utilise le modèle ministral que tu as déjà testé
        response = ollama.chat(
            model="ministral-3:3b",
            messages=[SYSTEM_PROMPT] + history + [{"role": "user", "content": message}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Désolé, j'ai un petit souci technique : {str(e)}"

@app.post("/chat")
async def chat(request: ChatRequest):
    sid = request.sessionId
    if sid not in session_memories:
        session_memories[sid] = []
    
    user_message = request.message
    bot_reply = get_ollama_response(user_message, session_memories[sid])
    
    # Mise à jour de la mémoire de session
    session_memories[sid].append({"role": "user", "content": user_message})
    session_memories[sid].append({"role": "assistant", "content": bot_reply})
    
    # Limiter la mémoire pour éviter de saturer le contexte
    if len(session_memories[sid]) > 10:
        session_memories[sid] = session_memories[sid][-10:]
        
    return {"reply": bot_reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)