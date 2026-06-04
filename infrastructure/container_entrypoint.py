"""Point d'entrée du conteneur SageMaker Processing Job.

Installe les dépendances runtime, puis lance htmir-collect.
Exécuté comme : python3 /opt/ml/processing/input/code/infrastructure/container_entrypoint.py
"""

import subprocess
import sys
from pathlib import Path

CODE_DIR = "/opt/ml/processing/input/code"
CONFIG_PATH = "/opt/ml/processing/input/config/collection.yaml"

# 1. Installer le package sans ses dépendances lourdes (torch, kraken…)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", CODE_DIR],
    check=True,
)

# 2. Installer uniquement les dépendances nécessaires à la collecte,
#    en pinant numpy<2.0 pour compatibilité avec le cv2 du conteneur
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet",
        "numpy>=1.26.4,<2.0",
        "requests>=2.32.0",
        "boto3>=1.34.0",
        "Pillow>=10.3.0",
        "PyYAML>=6.0",
        "datasets>=2.19.1",
        "huggingface-hub>=0.23.2",
        "jsonschema>=4.22.0",
    ],
    check=True,
)

# 3. Ajouter src/ au path et lancer la collecte HuggingFace
sys.path.insert(0, str(Path(CODE_DIR) / "src"))
sys.argv = ["htmir-collect", "--config", CONFIG_PATH, "--source", "huggingface"]

from htmir.cli.collect import main  # noqa: E402
main()
