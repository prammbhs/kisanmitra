"""SQLite Database Engine for User & Profile data on EBS volume."""
import os
from sqlmodel import Session, SQLModel, create_engine
from app.config import APP_DB_PATH

# Ensure parent directory on EBS exists
db_dir = os.path.dirname(APP_DB_PATH)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

sqlite_url = f"sqlite:///{APP_DB_PATH}"

# check_same_thread=False is safe for FastAPI multi-thread worker
engine = create_engine(
    sqlite_url,
    connect_args={"check_same_thread": False},
    echo=False
)


def init_db() -> None:
    """Create tables if they do not exist."""
    SQLModel.metadata.create_all(engine)
    print(f"[DB INIT] SQLite App Database ready at '{APP_DB_PATH}'")


def get_session():
    """Dependency for acquiring a db session."""
    with Session(engine) as session:
        yield session
