import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_mythos import (
    mythos_1b,
    OpenMythos,
)

cfg = mythos_1b()
model = OpenMythos(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")
