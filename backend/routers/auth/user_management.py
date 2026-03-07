"""
user_management.py

Business logic for user management.
Handles password hashing, JWT token creation, and CRUD operations on users.
All database operations go through the Database context manager.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel

from database import Database


from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer


# ---------------------------------------------------------------------------
# JWT Configuration
# ---------------------------------------------------------------------------

# Read from env in prod — fallback for local dev only
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

# bcrypt is the recommended hashing scheme — slow by design to resist brute force
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UnsecureUser(BaseModel):
    """User input — carries plain password, never persisted as-is."""
    email:    str
    password: str


class User(BaseModel):
    """Safe user model — only hashed password is stored and exposed."""
    email:           str
    hashed_password: str

    @classmethod
    def from_unsecure_user(cls, user: UnsecureUser) -> "User":
        """Hash the plain password and return a safe User instance."""
        return cls(
            email=user.email,
            hashed_password=hash_password(user.password),
        )


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Encode user data into a signed JWT token with an expiry.
       Details: 
          - The token is returned to the client after login.
          - For all future queries, the client sends ithis token in the 'Authorization: Bearer' header
          - get_current_user() decodes it to retrieve the email and identify the user.
    """
    # Copy user dict and add token expiration date
    to_encode = data.copy()
    expire    = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    # Retrun a signed JWT token 
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def user_exists(db_url: str, email: str) -> bool:
    """
    Checks if an email already exists in the database.
    Used during signup to prevent duplicate accounts.
    """
    with Database(db_url) as db:
        row = db.fetchone(
            "SELECT email FROM users WHERE email = ?", (email,)
        )
    return row is not None


def create_user(db_url: str, user: User) -> None:
    """
    Inserts a new user with a pre-hashed password into the database.
    """
    with Database(db_url) as db:
        db.execute(
            "INSERT INTO users (email, hashed_password) VALUES (?, ?)",
            (user.email, user.hashed_password),
        )


def get_user(db_url: str, email: str) -> Optional[User]:
    """
    Fetch user by email.
    """

    with Database(db_url) as db:
        row = db.fetchone(
            "SELECT email, hashed_password FROM users WHERE email = ?", (email,)
        )
    if row is None:
        return None
    return User(email=row[0], hashed_password=row[1])


def delete_user(db_url: str, email: str) -> None:
    with Database(db_url) as db:
        db.execute(
            "DELETE FROM users WHERE email = ?", (email,)
        )


def authenticate_user(db_url: str, email: str, password: str) -> Optional[User]:
    """Fetch user by email and verify password. Returns None if invalid.
       Used at login to check credentials in the database."""
    user = get_user(db_url, email)
    if user is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------


_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login")


async def get_current_user(token: str = Depends(_oauth2_scheme)) -> User:
    """
    FastAPI dependency — validates the JWT token on every protected request.

    Reads the Bearer token from the Authorization header, decodes it and
    verifies its signature against SECRET_KEY. No database query is performed —
    the signature is sufficient to guarantee authenticity.

    Returns a minimal User (email only) — hashed_password is not stored in the token.

    Inject into any route or add_routes() call to protect it:
        @app.get("/protected")
        async def route(user: User = Depends(get_current_user)): ...

        add_routes(app, chain, dependencies=[Depends(get_current_user)])

    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("email")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # db_url is not available here — caller must pass it via closure or config
    # get_current_user only validates the token, user existence is checked in routes
    return User(email=email, hashed_password="")