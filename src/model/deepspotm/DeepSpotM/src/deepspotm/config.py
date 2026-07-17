from pathlib import Path
import importlib.resources



class Config:
    """Configuration class for DeepSpotM model."""

    # Gene vocabulary
    try:
        with importlib.resources.path("deepspotm.assets", "tokens.csv") as p:
            ALPHABET_PATH = Path(p)
    except Exception:
        ALPHABET_PATH = Path(__file__).resolve().parent / "assets" / "tokens.csv"


config = Config()
