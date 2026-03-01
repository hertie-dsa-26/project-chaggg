"""
Run the Chicago Crime Analysis Flask application.

From project root, run:
    python run_app.py

Or with Flask CLI:
    flask --app src.flask_app run
"""
import sys
from pathlib import Path

# Ensure project root is on path (works before pip install -e .)
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.flask_app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
