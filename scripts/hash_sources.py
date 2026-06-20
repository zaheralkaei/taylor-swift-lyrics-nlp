"""
Recompute the combined source hash for reproduce-the-analysis verification.
Used by docs/SOURCE_HASH.txt. Run after any change to analyze/*.py or data/*.
"""
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

files = []
for d in ["analyze", "data"]:
    for ext in ["*.py", "*.json", "*.tsv"]:
        for f in (REPO_ROOT/d).rglob(ext):
            if "__pycache__" in str(f): continue
            files.append(f)

hashes = {}
for f in sorted(files):
    h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    hashes[str(f.relative_to(REPO_ROOT))] = h

combined = "\n".join(f"{h} {k}" for k, h in hashes.items())
root_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]

print(f"Combined SHA256: {root_hash}\n")
print("Individual file hashes (first 16 chars of SHA256):\n")
for k, h in hashes.items():
    print(f"  {h}  {k}")
