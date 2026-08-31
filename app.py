"""
Root entry point for Render / WSGI servers.
Imports the Flask app instance from backend/server.py.
"""
from backend.server import app

if __name__ == "__main__":
    app.run()
