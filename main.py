from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, UTC
from enum import Enum
from sqlalchemy import String, Integer, DateTime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.future import select
from contextlib import asynccontextmanager


# db config
DATABASE_URL = "sqlite+aiosqlite:///./reviews.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# keywords enum
class SentimentEnum(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


# Base class for ORM
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


# Dependency for DB session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


class SentimentAnalyzer:
    def __init__(self):
        self.positive_keywords = ["хорош", "люблю", "отличн", "прекрасн", "нравит"]
        self.negative_keywords = ["плох", "ненавиж", "ужас", "не работа"]

    def analyze(self, text: str) -> SentimentEnum:
        lower_text = text.lower()
        match lower_text:
            case _ if any(word in lower_text for word in self.positive_keywords):
                return SentimentEnum.positive
            case _ if any(word in lower_text for word in self.negative_keywords):
                return SentimentEnum.negative
            case _:
                return SentimentEnum.neutral


# Dependency
def get_sentiment_analyzer():
    return SentimentAnalyzer()


# Repository
class ReviewRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_review(self, text: str, sentiment: str) -> Review:
        review = Review(text=text, sentiment=sentiment)
        self.db.add(review)
        await self.db.commit()
        await self.db.refresh(review)
        return review

    async def get_reviews(self, sentiment: Optional[str] = None) -> List[Review]:
        stmt = select(Review)
        if sentiment:
            stmt = stmt.where(Review.sentiment == sentiment)
        result = await self.db.execute(stmt)
        return result.scalars().all()



@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


# FastAPI app
app = FastAPI(lifespan=lifespan)


@app.post("/reviews", response_model=ReviewResponse)
async def create_review(
    review: ReviewRequest,
    db: AsyncSession = Depends(get_db),
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """
    Receives review from client. Writes reviews to database.
    """
    sentiment = analyzer.analyze(review.text).value
    repository = ReviewRepository(db)
    saved_review = await repository.create_review(review.text, sentiment)
    return saved_review


@app.get("/reviews", response_model=List[ReviewResponse])
async def get_reviews(
    sentiment: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve a list of reviews from the database.
    """
    repository = ReviewRepository(db)
    return await repository.get_reviews(sentiment)
