#!/usr/bin/env python3
"""Stage deterministic static UI fixtures and compare two RmlUi layout probe outputs.

This intentionally does not claim runtime coverage: data bindings, generated theme
media and custom element internals require the full application verification.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def stage(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    resources = root / "src/visualizer/gui/rmlui/resources"
    inventory = []
    for source in sorted(resources.iterdir()):
        if source.suffix not in (".rml", ".rcss"):
            continue
        text = source.read_text(encoding="utf-8")
        if source.suffix == ".rml":
            inventory.append({"file": source.name,
                              "bindings": sorted(set(re.findall(r'\b(data-[\w-]+)', text))),
                              "custom_tags": sorted(set(re.findall(r'<([\w]+-[\w-]+)', text)))})
            # All conditional branches remain present; dynamic loops have one representative row.
            text = re.sub(r'\sdata-[\w-]+\s*=\s*(?:"[^"]*"|\x27[^\x27]*\x27)', '', text)
            text = re.sub(r'\{\{.*?\}\}', 'Sample', text, flags=re.S)
            text = re.sub(r'@tr:([\w.]+)', lambda m: m[1].split('.')[-1].replace('_', ' '), text)
        (destination / source.name).write_text(text, encoding="utf-8")
    (destination / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Staged {len(inventory)} RML resources (including templates).")


def compare(before, after, output):
    old, new = (json.loads(p.read_text(encoding="utf-8")) for p in (before, after))
    changes = []
    for case in sorted(old["cases"].keys() | new["cases"].keys()):
        a, b = old["cases"].get(case, {}), new["cases"].get(case, {})
        for state in sorted(a.keys() | b.keys()):
            if not isinstance(a.get(state), dict) or not isinstance(b.get(state), dict):
                if a.get(state) != b.get(state):
                    changes.append({"case": case, "state": state, "before": a.get(state), "after": b.get(state)})
                continue
            for path in sorted(a[state].keys() | b[state].keys()):
                first, second = a[state].get(path), b[state].get(path)
                if first is None or second is None:
                    changes.append({"case":case,"state":state,"path":path,"before":first,"after":second})
                    continue
                fields = {}
                for key in first.keys() | second.keys():
                    x, y = first.get(key), second.get(key)
                    if key == "class" and isinstance(x, str) and isinstance(y, str):
                        # RmlUi 6.3 normalizes whitespace in class lists. CSS
                        # matches tokens, so spacing and token order are immaterial.
                        equal = set(x.split()) == set(y.split())
                    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
                        equal = all(abs(i-j) <= 0.1 for i,j in zip(x,y))
                    else:
                        equal = x == y
                    if not equal:
                        fields[key] = {"before": x, "after": y}
                if fields:
                    changes.append({"case":case,"state":state,"path":path,"tag":first["tag"],"id":first["id"],"class":first["class"],"fields":fields})
    report = {"cases_before":len(old["cases"]),"cases_after":len(new["cases"]),
              "load_failures_before":old["load_failures"],"load_failures_after":new["load_failures"],
              "changed_documents":sorted({c["case"].split('@')[0] for c in changes}),
              "changed_elements":len(changes),"changes":changes,
              "changed_elements_by_state":dict(Counter(c["state"] for c in changes)),
              "messages_before":old["messages"],"messages_after":new["messages"]}
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("changes","messages_before","messages_after")},indent=2))
    return bool(changes or old["load_failures"] or new["load_failures"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("stage")
    p.add_argument("root", type=Path)
    p.add_argument("destination", type=Path)
    p = commands.add_parser("compare")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--report-only", action="store_true", help="Do not fail when layouts differ")
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.root, args.destination)
    else:
        changed = compare(args.before, args.after, args.output)
        raise SystemExit(1 if changed and not args.report_only else 0)
