from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"DEBUG: Active GEMINI_API_KEY: {api_key[:6]}...{api_key[-4:]}")
else:
    print("DEBUG: Active GEMINI_API_KEY: MISSING")
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import timedelta
import json
import io
from PIL import Image
from typing import Optional

from db import Base, engine, get_db, User, History, SavedOutfit, Chat
from auth import (
    verify_password, get_password_hash, create_access_token, 
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from pydantic import BaseModel
import app3  # Leveraging existing search logic

# Create tables if not exist
Base.metadata.create_all(bind=engine)

# App state to hold models
class AppState:
    clip_model = None
    fashion_bert = None
    faiss_index = None
    fashion_items = None
    embeddings = None

app = FastAPI(title="AI Fashion Styling Assistant")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_models():
    print("Loading AI Models and FAISS Index (this may take a bit)...")
    try:
        from fashion_models import get_fashion_models
        import faiss
        import pickle
        AppState.clip_model, AppState.fashion_bert = get_fashion_models()
        AppState.faiss_index = faiss.read_index("fashion_index.faiss")
        with open("fashion_items.pkl", "rb") as f:
            AppState.fashion_items = pickle.load(f)
        with open("fashion_embeddings.pkl", "rb") as f:
            AppState.embeddings = pickle.load(f)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models. Have you run build_faiss_index.py? Error: {e}")

# Frontend static files routing
try:
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
except Exception:
    pass

# schemas
class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    preferences: dict

class UserLogin(BaseModel):
    email: str
    password: str

class ChatMessage(BaseModel):
    message: str

import re
def normalize_styling_response(text: str) -> str:
    if not text: return ""
    
    # 1. Remove Markdown Headings (##, ###, etc.)
    text = re.sub(r'(?m)^#+\s*', '', text)
    
    # 2. Remove Bold and Italics (**, *)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # 3. Normalize Bullets (convert *, • to -)
    text = re.sub(r'^[ \t]*[*•][ \t]*', '- ', text, flags=re.M)
    
    # 4. Remove Numbered Sections (1., 2., etc.)
    text = re.sub(r'(?m)^\d+\.\s*', '', text)
    
    # 5. Fix stray characters & inline artifacts
    text = text.replace('•', '-')
    
    # 6. Normalize Spacing (Trim lines and reduce multiple newlines)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = get_password_hash(user.password)
    new_user = User(
        name=user.name,
        email=user.email,
        password_hash=hashed_pw,
        preferences=json.dumps(user.preferences)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user_id": new_user.id}

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "preferences": db_user.get_preferences()}

@app.post("/analyze-style")
async def analyze_style(
    image: Optional[UploadFile] = File(None),
    text_query: Optional[str] = Form(""),
    occasion: Optional[str] = Form(""),
    season: Optional[str] = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not image and not text_query:
        raise HTTPException(status_code=400, detail="Must provide image or text_query")
    
    try:
        if image:
            img_content = await image.read()
            pil_image = Image.open(io.BytesIO(img_content)).convert("RGB")
            query_embedding = AppState.clip_model.encode_image(pil_image)
        else:
            query_embedding = AppState.clip_model.encode_text(text_query)
        
        # Search FAISS
        # Re-using the logic from app3
        results = app3.search_similar_items(
            query_embedding, 
            AppState.faiss_index, 
            AppState.fashion_items, 
            top_k=20
        )
        
        # Filter context
        if occasion or season:
            results = app3.filter_by_context(results, occasion, season)
        
        # Personalization (Indian vs Western based on preferences)
        prefs = current_user.get_preferences()
        preferred_style = prefs.get('style_type', '')
        # Simple boost for matching dataset origins or categories (to process later in unified dataset)
        # Indian matching logic can be refined later when dataset is merged
        if preferred_style == 'Indian':
            # Boost ethnic items in results if available
            pass 
        
        # Strict Category Post-Filtering to prevent irrelevant vectors (e.g., ties for dresses)
        if text_query:
            query_lower = text_query.lower()
            clothing_keywords = ["dress", "shirt", "skirt", "tie", "pant", "jeans", "top", "suit", "jacket", "coat", "sweater", "saree", "kurta", "lehenga"]
            # Find which strict keywords the user asked for
            active_keywords = [kw for kw in clothing_keywords if kw in query_lower]
            
            if active_keywords:
                filtered_results = []
                for res in results:
                    text_blob = f"{res.get('category','')} {res.get('name','')} {res.get('description','')}".lower()
                    # Keep if ANY active keyword is found in the item's text blob
                    if any(kw in text_blob for kw in active_keywords):
                        filtered_results.append(res)
                
                # Only apply strict filter if we still have results left, otherwise fallback to original
                if len(filtered_results) > 0:
                    results = filtered_results

        # Generate Styling Advice (Now returns a dict)
        advice_data = app3.generate_styling_advice(results, occasion, season, text_query)
        
        # Record History correctly (Store JSON string of advice for future context)
        hist_prompt = text_query if text_query.strip() else "Visual Search (Image Upload)"
        hist = History(
            user_id=current_user.id,
            input_type="image" if image else "text",
            user_prompt=hist_prompt,
            response=json.dumps({"results_count": len(results), "advice": advice_data})
        )
        db.add(hist)
        db.commit()
        
        # Clean results (remove non-serializable objects like memory references if any)
        clean_results = []
        for r in results[:10]:
            cln = r.copy()
            if 'image' in cln: cln.pop('image') # We don't send raw PIL back
            if 'embedding' in cln: cln.pop('embedding', None)
            clean_results.append(cln)

        return {
            "results": clean_results, 
            "advice": advice_data # This contains suggested_look and outfit_matches
        }
        
    except Exception as e:
        print(f"CRITICAL ERROR in analyze_style: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-outfit")
def save_outfit(outfit_data: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    so = SavedOutfit(
        user_id=current_user.id,
        outfit_data=json.dumps(outfit_data),
        liked=True
    )
    db.add(so)
    db.commit()
    return {"status": "success", "id": so.id}

@app.get("/saved-outfits")
def get_saved_outfits(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    outfits = db.query(SavedOutfit).filter(SavedOutfit.user_id == current_user.id).order_by(SavedOutfit.created_at.desc()).all()
    return [{"id": o.id, "data": json.loads(o.outfit_data), "liked": o.liked, "time": o.created_at} for o in outfits]

@app.get("/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    hist = db.query(History).filter(History.user_id == current_user.id).order_by(History.timestamp.desc()).all()
    return [{"id": h.id, "type": h.input_type, "prompt": h.user_prompt, "time": h.timestamp} for h in hist]

@app.get("/user-stats")
def get_user_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    saved_count = db.query(SavedOutfit).filter(SavedOutfit.user_id == current_user.id).count()
    recommendation_count = db.query(History).filter(History.user_id == current_user.id).count()
    return {
        "name": current_user.name,
        "saved_count": saved_count,
        "recommendation_count": recommendation_count
    }
@app.post("/chat")
def chat_agent(chat: ChatMessage, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from google import genai
    import os

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key missing")

    try:
        client = genai.Client(api_key=api_key)

        # Clean history (ONLY user messages, no JSON)
        recent_chats = db.query(Chat).filter(Chat.user_id == current_user.id)\
            .order_by(Chat.timestamp.desc()).limit(3).all()

        history_context = ""
        for rc in reversed(recent_chats):
            history_context += f"User: {rc.message}\n"

        prompt = f"""You are Genie — a warm, sharp, luxury fashion stylist having a real conversation with a user.

        Your goal is to understand the user's needs through the conversation itself — 
        NOT from stored data. Pay close attention to everything they say: 
        what occasion they mention, what they like or dislike, 
        how they describe their body, their lifestyle, their budget signals, 
        colour mentions, cultural context (Indian or Western wear), 
        and any past outfit references they make.

        ---

        CONVERSATION HISTORY (read carefully to extract what you already know):
        {history_context}

        ---

        HOW TO BEHAVE:

        PHASE 1 — DISCOVERY (first 1-2 replies):
        If the user's message is vague or missing key details, ask ONE focused follow-up question.
        Pick the single most important unknown. Examples:
        - "What's the occasion — a casual day out or something more formal?"
        - "Are you thinking Indian ethnic, Western, or a fusion look?"
        - "What's your colour comfort zone — do you lean towards neutrals, bold tones, or pastels?"
        - "Any silhouettes you love or want to avoid?"
        Never ask more than one question per reply. Never bullet your questions.

        PHASE 2 — RECOMMENDATION (once you have: occasion + style preference + at least one personal detail):
        Give a complete, specific styled look in 4-6 flowing sentences. No bullet points. No headers.
        Cover ALL of the following naturally within those sentences:
        → Outfit: specific garments, fabric, fit, how it's worn (tucked, layered, draped)
        → Colour palette: 2-3 colours that work together, with a reason why
        → Footwear: exact style and colour (e.g. "ivory block-heel mules", "tan kolhapuris")
        → Bag or accessory: specific type and tone
        → Jewellery: be precise (e.g. "slim gold bangles and a delicate mangalsutra", "oxidised jhumkas", "pearl drop earrings")
        → Optional: a one-line finishing tip (hair, lipstick shade, how to carry the look)

        PHASE 3 — FOLLOW-UP (after giving a look):
        End with one gentle question to refine further, like:
        - "Would you like a more casual version of this, or something with a bit more drama?"
        - "Want me to swap this for a saree or lehenga option instead?"
        Never repeat a look already given in this conversation.

        ---

        TONE RULES:
        - Write like a stylist at a boutique — confident, warm, never robotic
        - No hashtags, no markdown, no emojis
        - No generic phrases like "Of course!", "Great choice!", "Certainly!" or "Darling,"
        - Start replies directly — with the question or the look, not a preamble
        - Keep it conversational but polished, like texting a stylish friend who knows fashion

        ---

        User's latest message: {chat.message}

        Genie's reply:"""
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'max_output_tokens': 1000,
                'temperature': 0.7
            }
        )

        final_text = response.text.strip()

        # Save chat
        chat_rec = Chat(
            user_id=current_user.id,
            message=chat.message,
            response=final_text
        )
        db.add(chat_rec)
        db.commit()

        return {"response": final_text}

    except Exception as e:
        print(f"ERROR in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/image/{item_id}")
def get_image(item_id: int):
    # Locate item
    if not AppState.fashion_items:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
        
    for item in AppState.fashion_items:
        if item['id'] == item_id:
            img = item.get('image')
            if not img:
                raise HTTPException(status_code=404, detail="Image missing for this item")
            
            # Convert PIL image to byte stream
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/jpeg")
            
    raise HTTPException(status_code=404, detail="Item not found")
