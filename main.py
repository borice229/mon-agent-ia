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


API_KEY = os.getenv("GOOGLE_API_KEY")

# URL forcée en version stable V1
URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={API_KEY}"

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
    # Prompt consolidé pour éviter les erreurs de structure JSON
    prompt_final = (
        f"Tu es l'assistant virtuel de Borice Dossou. Réponds de manière professionnelle et amicale. "
        f"Ne fais pas de longs discours. Sois direct et pro. "
        f"Voici ses informations : {PROFILE_TEXT}\n\n"
        f"Question de l'utilisateur : {request.message}"
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