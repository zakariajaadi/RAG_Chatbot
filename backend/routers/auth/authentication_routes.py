"""
authentication_routes.py

FastAPI routes for user authentication.
Handles signup, login, and user management via JWT tokens.

Returns a reusable Depends(get_current_user) that can be injected
into other routers (e.g. session_routes) to protect their endpoints.

Usage in server.py:
    from routers.auth.authentication_routes import authentication_routes
    authentication = authentication_routes(app, config)
"""

from pathlib import Path
from typing import List, Optional, Sequence

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

import os

from config import RAGConfig

# If False, signup is disabled — accounts must be created by an admin
ADMIN_MODE = os.getenv("ADMIN_MODE", "false").lower() == "true"
from database import Database

from routers.auth.user_management import (
    User,
    UnsecureUser,
    authenticate_user,
    create_access_token,
    create_user,
    delete_user,
    get_current_user,
    get_user,
    user_exists,
)


def authentication_routes(
    app: FastAPI | APIRouter,
    config: RAGConfig,
) -> Depends:
    """
    Register authentication routes on the given FastAPI app or router.
    Creates the users table on startup if it doesn't exist.
    Returns a Depends(get_current_user) to inject into protected routes.

    Args:
        app:    FastAPI app or APIRouter to register routes on
        config: RAGConfig instance (provides db_url)

    Returns:
        Depends: reusable dependency that resolves to the authenticated User
    """
    # Get db_url
    db_url = config.database.db_url

    # Create users table on startup if it doesn't exist
    with Database(db_url) as db:
        db.run_script(Path(__file__).parent / "users_tables.sql")


    # ------------------------------------------------------------------ #
    # Routes
    # ------------------------------------------------------------------ #

    @app.post("/user/signup", include_in_schema=ADMIN_MODE)
    async def signup(user: UnsecureUser) -> dict:
        """
        Create a new user account. Email must be unique.
        Only available when ADMIN_MODE=true — disabled in production by default.
        """
        if not ADMIN_MODE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Signup is disabled — contact your administrator",
            )
        if user_exists(db_url, user.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User {user.email} already registered",
            )
        create_user(db_url, User.from_unsecure_user(user))
        return {"email": user.email}

    @app.post("/user/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> dict:
        """
        Authenticate with email + password.
        Returns a JWT access token to use in subsequent requests.
        """
        user = authenticate_user(db_url, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Build token payload — exclude hashed_password
        user_data = {"email": user.email}
        access_token = create_access_token(data=user_data)
        return {"access_token": access_token, "token_type": "bearer"}

    @app.get("/user/me")
    async def user_me(current_user: User = Depends(get_current_user)) -> User:
        """Return the currently authenticated user."""
        return current_user

    @app.delete("/user/")
    async def del_user(current_user: User = Depends(get_current_user)) -> dict:
        """Delete the currently authenticated user account."""
        user = get_user(db_url, current_user.email)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {current_user.email} not found",
            )
        delete_user(db_url, current_user.email)
        return {"detail": f"User {current_user.email} deleted"}

    # Return get_current_user as a reusable Depends for other routers
    return Depends(get_current_user)