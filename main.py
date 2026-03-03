import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")

# URL forcée en version stable V1
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

class ChatRequest(BaseModel):
    message: str
    sessionId: str

# Lecture sécurisée du fichier profile.txt
PROFILE_PATH = "profile.txt"  # ou chemin absolu si besoin

if os.path.exists(PROFILE_PATH):
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        PROFILE_TEXT = f.read()
else:
    PROFILE_TEXT = "Profil indisponible. Vérifiez le fichier profile.txt"


@app.post("/chat")
async def chat(request: ChatRequest):
    # Prompt avec instructions de ciblage
    prompt_final = (
        f"Tu es l'expert virtuel de Borice Dossou. Ton objectif est de convaincre les recruteurs en étant précis.\n\n"
        f"CONTEXTE DU PROFIL :\n{PROFILE_TEXT}\n\n"
        f"DIRECTIVES DE RÉPONSE :\n"
        f"1. Si la question porte sur les COMPÉTENCES : Liste les outils techniques (Python, SQL, Power BI, etc.) mentionnés dans le profil.\n"
        f"2. Si la question porte sur la RECHERCHE D'EMPLOI : Précise que Borice recherche des opportunités en Data Science/Analytics et souligne sa valeur ajoutée.\n"
        f"3. Si la question est VAGUE : Réponds par une question ouverte pour guider l'utilisateur vers ses projets ou son expérience.\n"
        f"4. INTERDICTION : Ne commence JAMAIS par 'Je suis l'assistant ...' si l'utilisateur a déjà posé une question précise.\n\n"
        f"MESSAGE DE L'UTILISATEUR : {request.message}"
    )
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt_final}]
        }]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(URL, json=payload, timeout=30.0)
            data = response.json()
            
            # Gestion des erreurs renvoyées par Google
            if response.status_code != 200:
                error_msg = data.get('error', {}).get('message', 'Erreur inconnue')
                return {"reply": f"Erreur Google ({response.status_code}) : {error_msg}"}

            # Extraction de la réponse
            if "candidates" in data and len(data["candidates"]) > 0:
                reply = data["candidates"][0]["content"]["parts"][0]["text"]
                return {"reply": reply}
            else:
                return {"reply": "L'IA n'a pas pu générer de réponse. Réessaie."}
                
        except Exception as e:
            return {"reply": f"Erreur de connexion : {str(e)}"}

if __name__ == "__main__":
    import os
    import uvicorn
    # Render définit une variable d'environnement PORT, sinon on utilise 8000 par défaut
    port = int(os.environ.get("PORT", 8000))
    # On force l'hôte à 0.0.0.0 pour être accessible sur le web
    uvicorn.run(app, host="0.0.0.0", port=port)