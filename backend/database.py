"""
Handles database operations with connection pooling and multi-dialect support.
Supports SQLite, PostgreSQL, MySQL, and SQL Server.

Usage:
    from database import Database

    with Database(db_url) as conn:
        conn.execute("INSERT INTO users VALUES (?, ?)", (email, hashed_password))

    with Database(db_url) as conn:
        row = conn.fetchone("SELECT * FROM users WHERE email = ?", (email,))
"""

import sqlite3
import logging
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import sqlglot
from dbutils.pooled_db import PooledDB
from sqlalchemy.engine.url import make_url


# Connection pool singleton — shared across all Database instances
_POOL: Optional[PooledDB] = None


class Database:
    """
    Manages database connections with pooling and multi-dialect support.

    Uses a context manager to ensure connections are properly opened and closed.
    Automatically transpiles SQL scripts written in SQLite dialect to the
    target database dialect (PostgreSQL, MySQL, etc.).

    Args:
        db_url: SQLAlchemy-style connection string (e.g. "sqlite:///path/to/db.sqlite")
        logger: Optional logger instance
    """

    # Each dialect has its own placeholder syntax for query parameters
    DIALECT_PLACEHOLDERS = {
        "sqlite":     "?",  # "SELECT * FROM users WHERE email = ?"
        "postgresql": "%s", # "SELECT * FROM users WHERE email = %s"
        "mysql":      "%s", # "SELECT * FROM users WHERE email = %s"

    }

    def __init__(self, db_url: str, logger: Optional[Logger] = None):
        
        
        # Store db url
        self.db_url = db_url
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Parse URL into a structured object, exposes drivername, host, port, database as attributes 
        # e.g. self.url.drivername, self.url.host
        self.url    = make_url(db_url)
        
        # Initialize the shared pool (module-level singleton : created once, reused by all instances)
        global _POOL
        _POOL    = _POOL or self._create_pool()

        # Each instance maintains a reference to the shared connection pool.
        self.pool = _POOL
        
        # Actual connection is not taken here — only inside __enter__
        self.conn = None

    # ------------------------------------------------------------------ #
    # Context Manger  
    # ------------------------------------------------------------------ #
    
    def __enter__(self) -> "Database":
        """
        Called automatically when entering a "with" block.
        Borrows a connection from the pool """
        self.conn = self.pool.connection()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],     # None if no exception, otherwise the exception class
        exc_value: Optional[BaseException],  # The exception instance
        traceback: Optional[Any],            # The stack trace
    ) -> None:
        """
        Called automatically when exiting a "with" block — always runs,
        even if an exception occurred inside the block.
        Commits on success, rolls back on error, always returns the connection to the pool. 
        """
        if self.conn:
            if exc_type:
                self.logger.error(
                    "Transaction failed", exc_info=(exc_type, exc_value, traceback)
                )
                self.conn.rollback()  # error → undo all operations in the block
            else:
                self.conn.commit()    # success → persist all operations atomically
            self.conn.close()         # return the connection to the pool
        self.conn = None              # prevent accidental reuse after exit

    # ------------------------------------------------------------------ #
    # Query execution
    # ------------------------------------------------------------------ #

    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        Execute a query and return the raw cursor.
        """
        # Create cursor from the active connexion
        cursor = self.conn.cursor()
        try:
            # Replace "?" by the dialect-specific placeholder (e.g. "%s" for PostgreSQL)
            placeholder = self.DIALECT_PLACEHOLDERS.get(self.url.drivername, "?")
            query = query.replace("?", placeholder)
            # Execute query   (params or () : most drivers don't accept None as params)
            cursor.execute(query, params or ())
            return cursor
        except Exception as e:
            cursor.close()
            self.logger.exception("Query execution failed", exc_info=e)
            raise

    def fetchone(self, query: str, params: Optional[tuple] = None) -> Optional[tuple]:
        """
        Wrapper around execute, to fetch a single line.
        """
        cursor = self.execute(query, params)
        try:
            return cursor.fetchone()
        finally:
            cursor.close()

    def fetchall(self, query: str, params: Optional[tuple] = None) -> list:
        """
        Wrapper around execute, to fetch all lines.
        """
        cursor = self.execute(query, params)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()

    # ------------------------------------------------------------------ #
    # Schema management
    # ------------------------------------------------------------------ #

    def run_script(self, path: Path) -> None:
        """
        Execute a SQL script file written in SQLite dialect.
        Automatically transpiles to the target dialect (PostgreSQL, MySQL, etc.)
        Typically called at startup to create tables (CREATE TABLE IF NOT EXISTS).
        """
        try:
            # Transpile from SQLite to the target dialect
            sql_script = path.read_text()
            transpiled = sqlglot.transpile(
                sql_script,
                read="sqlite",
                write=self.url.drivername.replace("postgresql", "postgres"),
            )
            
            # Execute each statement individually (sqlglot splits on ";")
            for statement in transpiled:
                self.execute(statement)
            self.logger.info(f"Script {path.name} executed successfully.")
        except Exception as e:
            self.logger.exception(f"Failed to execute script {path}", exc_info=e)
            raise

    # ------------------------------------------------------------------ #
    # Connection pool
    # ------------------------------------------------------------------ #

    def _create_pool(self) -> PooledDB:
        """
        Create the connection pool for the target database.
        Called once at first instantiation — result is cached as a module singleton.
        Drivers are imported lazily so you only need to install what you actually use.
        maxconnections=5 limits concurrent connections to avoid overwhelming the db.
        """
        # SQLite
        if self.db_url.startswith("sqlite:///"):
            db_path = Path(self.db_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return PooledDB(
                creator=sqlite3,
                database=str(db_path),
                maxconnections=5,
            )
        # Postgres
        elif self.db_url.startswith("postgresql://"):
            import psycopg2
            return PooledDB(creator=psycopg2, dsn=self.db_url, maxconnections=5)
        
        # Mysql
        elif self.db_url.startswith(("mysql://", "mysql+pymysql://")):
            import mysql.connector
            return PooledDB(
                creator=mysql.connector,
                user=self.url.username,
                password=self.url.password,
                host=self.url.host,
                port=self.url.port,
                database=self.url.database,
                maxconnections=5,
            )

        # SQLServer
        elif self.db_url.startswith("sqlserver://"):
            import pyodbc
            return PooledDB(
                creator=pyodbc,
                dsn=self.db_url.replace("sqlserver://", ""),
                maxconnections=5,
            )

        else:
            raise ValueError(f"Unsupported database dialect: {self.url.drivername}")