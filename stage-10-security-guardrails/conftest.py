import sys
from pathlib import Path

# Make this stage's src/ importable without installation
sys.path.insert(0, str(Path(__file__).parent / "src"))
