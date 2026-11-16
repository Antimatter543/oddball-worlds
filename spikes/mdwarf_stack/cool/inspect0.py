"""Quick inspection of graded tables (scratch)."""
import json
import sys

sys.path.insert(0, "/home/antiraedus/Projects/oddball-worlds/.claude/worktrees/agent-aca92afd38bce9bc4/spikes/mdwarf_stack")
from common import OUT

rows = [json.loads(line) for line in open(OUT / "survey100" / "vet" / "graded.jsonl")]
r = [x for x in rows if x["grade"]["status"].get("catalogue") == "fail"][0]
print(json.dumps(r["stats"]["star"], indent=1)[:4000])
print(json.dumps(r["grade"], indent=1)[:3000])
print(json.dumps(r["stats"]["v2"], indent=1)[:1500])
