from typing import TypeVar, Generic, TYPE_CHECKING, Any
from uuid import UUID
from collections.abc import Sequence
import uvicorn
from starlette.status import HTTP_200_OK, HTTP_201_CREATED
from fastapi import FastAPI, HTTPException, Query, Depends, APIRouter
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, UTC
from enum import Enum
from sqlalchemy import String, Integer, DateTime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from contextlib import asynccontextmanager
from sqlalchemy import delete, insert, select

if TYPE_CHECKING:
    from sqlalchemy.engine import Result

DATABASE_URL = "sqlite+aiosqlite:///./reviews.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class SentimentEnum(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class Base(DeclarativeBase):
    pass


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    sentiment: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now(UTC))


class ReviewRequest(BaseModel):
    text: str


class ReviewResponse(BaseModel):
    id: int
    text: str
    sentiment: str
    created_at: datetime

    class Config:
        from_attributes = True


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


class SentimentAnalyzer:
    _positive_keywords: set[str] = {"хорош", "люблю", "отличн", "прекрасн", "нравит"}
    _negative_keywords: set[str] = {"плох", "ненавиж", "ужас", "не работа"}

    @classmethod
    def analyze(cls, text: str) -> SentimentEnum:
        lower_text = text.lower()
        match lower_text:
            case _ if any(word in lower_text for word in cls._positive_keywords):
                return SentimentEnum.positive
            case _ if any(word in lower_text for word in cls._negative_keywords):
                return SentimentEnum.negative
            case _:
                return SentimentEnum.neutral


def get_sentiment_analyzer():
    return SentimentAnalyzer()


M = TypeVar('M', bound=BaseModel)


class SqlRepository(Generic[M]):
    _model: type[M]

    def __init__(self, session: AsyncSession = Depends(get_db)):
        self._session = session

    async def add_one(self, **kwargs: Any) -> None:
        query = insert(self._model).values(**kwargs)
        await self._session.execute(query)

    async def add_one_and_get_id(self, **kwargs: Any) -> int | str | UUID:
        query = insert(self._model).values(**kwargs).returning(self._model.id)
        obj_id: Result = await self._session.execute(query)
        return obj_id.scalar_one()

    async def add_one_and_get_obj(self, **kwargs: Any) -> M:
        query = insert(self._model).values(**kwargs).returning(self._model)
        obj: Result = await self._session.execute(query)
        return obj.scalar_one()

    async def get_by_filter_one_or_none(self, **kwargs: Any) -> M | None:
        query = select(self._model).filter_by(**kwargs)
        res: Result = await self._session.execute(query)
        return res.unique().scalar_one_or_none()

    async def get_by_filter_all(self, **kwargs: Any) -> Sequence[M]:
        query = select(self._model).filter_by(**kwargs)
        res: Result = await self._session.execute(query)
        return res.scalars().all()

    async def commit(self) -> None:
        await self._session.commit()


class ReviewRepository(SqlRepository[Review]):
    _model = Review


router = APIRouter()


class ReviewService:
    def __init__(
        self,
        analyzer: SentimentAnalyzer = Depends(),
        review_repository: ReviewRepository = Depends(),
    ):
        self._analyzer = analyzer
        self._review_repository = review_repository

    async def create_review(self, review: ReviewRequest) -> Review:
        sentiment = self._analyzer.analyze(review.text).value
        saved_review = await self._review_repository.add_one_and_get_obj(text=review.text, sentiment=sentiment)
        await self._review_repository.commit()
        return saved_review

    async def get_reviews(self, sentiment: str) -> List[Review]:
        reviews = await self._review_repository.get_by_filter_all(sentiment=sentiment)
        return reviews


@router.post(
    path="/reviews",
    response_model=ReviewResponse,
    status_code=HTTP_201_CREATED,
)
async def create_review(
        review: ReviewRequest,
        service: ReviewService = Depends(),
) -> ReviewResponse:
    """
    Receives review from client. Writes reviews to database.
    """
    saved_review: Review = await service.create_review(review)
    return ReviewResponse(
        id=saved_review.id,
        text=saved_review.text,
        sentiment=saved_review.sentiment,
        created_at=saved_review.created_at,
        )


@router.get(
    path="/reviews",
    response_model=List[ReviewResponse],
    status_code=HTTP_200_OK,
)
async def get_reviews(
        sentiment: Optional[str] = Query(None),
        service: ReviewService = Depends(),
):
    """
    Retrieve a list of reviews from the database.
    """
    reviews: List[Review] = await service.get_reviews(sentiment)
    return reviews


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
