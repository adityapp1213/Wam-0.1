from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path("/home/aditya/robotics/wam-0.1")
FINAL = PROJECT / "data" / "final"
ROBOCASA = Path("/home/aditya/robotics/robocasa-src")
REPORT = PROJECT / "reports" / "metadata_path_cleanup.json"


def metadata_files():
    found = set()
    for level in ("level_0", "level_1"):
        root = FINAL / level
        if not root.is_dir():
            continue
        for mode in root.iterdir():
            if not mode.is_dir() or mode.name == "quarantine":
                continue
            for pattern in ("*/*/metadata.json", "*/*/*/metadata.json", "*/*/*/*/metadata.json"):
                found.update(mode.glob(pattern))
    return sorted(found)


def portable_path(value: str) -> str:
    path = Path(value)
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        pass
    try:
        return "external://robocasa/" + path.relative_to(ROBOCASA).as_posix()
    except ValueError:
        return "external://source/" + path.name


def main() -> None:
    changed = []
    fields = {"source_dataset": 0, "source_level_0": 0}
    for path in metadata_files():
        meta = json.loads(path.read_text())
        original = json.dumps(meta, sort_keys=True)
        for key in fields:
            value = meta.get(key)
            if isinstance(value, str) and value.startswith("/"):
                meta[key] = portable_path(value)
                fields[key] += 1
        if json.dumps(meta, sort_keys=True) != original:
            path.write_text(json.dumps(meta, indent=2) + "\n")
            changed.append(path.relative_to(PROJECT).as_posix())

    result = {
        "changed_files": len(changed),
        "changed_fields": fields,
        "files": changed,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"changed_files={len(changed)} source_dataset={fields['source_dataset']} source_level_0={fields['source_level_0']}")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
