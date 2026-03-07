-- Creates the users table on startup if it doesn't exist.
CREATE TABLE IF NOT EXISTS "users" (
    "email" VARCHAR(255) PRIMARY KEY,
    "hashed_password" TEXT
);