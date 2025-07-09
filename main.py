from fastapi import FastAPI, HTTPException, Query, Depends, APIRouter
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, UTC
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy import Integer, String, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, Session


# db config
DATABASE_URL = "sqlite:///./reviews.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# keywords
positive_keywords = ["хорош", "люблю", "отличн", "прекрасн", "нравит"]
negative_keywords = ["плох", "ненавиж", "ужас", "не работа"]


# Models in db fo positive and negative reviews
class Base(DeclarativeBase):
    pass


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    sentiment: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now(UTC))


# Pydantic models
class ReviewRequest(BaseModel):
    text: str


class ReviewResponse(BaseModel):
    id: int
    text: str
    sentiment: str
    created_at: datetime

    class Config:
        from_attributes = True


# db session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def detect_sentiment(text: str) -> str:
    """
    Detects if the review is positive, negative or neutral.
    """
    lower_text = text.lower()
    if any(word in lower_text for word in positive_keywords):
        return "positive"
    elif any(word in lower_text for word in negative_keywords):
        return "negative"
    return "neutral"


app = FastAPI()
Base.metadata.create_all(bind=engine)


@app.post("/reviews", response_model=ReviewResponse)
def create_review(review: ReviewRequest, db: Session = Depends(get_db)):
    """
    Receives review from client. Writes reviews to database.
    """
    sentiment = detect_sentiment(review.text)

    db_review = Review(text=review.text, sentiment=sentiment)
    db.add(db_review)
    db.commit()
    db.refresh(db_review)

    return db_review


@app.get("/reviews", response_model=List[ReviewResponse])
def get_reviews(sentiment: Optional[str] = Query(None), db: Session = Depends(get_db)):
    """
    Retrieve a list of reviews from the database.
    """
    query = db.query(Review)
    if sentiment:
        query = query.filter(Review.sentiment == sentiment)
    return query.all()


