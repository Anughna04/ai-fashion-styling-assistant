import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./fashion_assistant.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    preferences = Column(Text, default="{}")  # Stored as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    history = relationship("History", back_populates="user")
    saved_outfits = relationship("SavedOutfit", back_populates="user")
    chats = relationship("Chat", back_populates="user")

    def get_preferences(self):
        return json.loads(self.preferences)
    
    def set_preferences(self, prefs_dict):
        self.preferences = json.dumps(prefs_dict)


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    input_type = Column(String) # "text", "image"
    user_prompt = Column(Text)
    response = Column(Text) # JSON string of result IDs or summary
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="history")


class SavedOutfit(Base):
    __tablename__ = "saved_outfits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    outfit_data = Column(Text) # JSON string of the outfit details
    liked = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="saved_outfits")


class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chats")

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
