"""Projects for the Streamlit build — plain JSON files in data/projects/.

No database on Streamlit Cloud: each project is one {id}.json file holding
the script + voice + preset + settings (never the audio — audio is
regenerated on demand, and the .json file is the permanent copy the user
downloads).

HONEST LIMIT: Streamlit Cloud has no persistent disk. Files written here
live only while the app container lives; when the app sleeps/restarts they
are gone. The UI says this plainly and offers Download/Upload of the
.json file as the permanent copy.
"""
import json
import time
import uuid
from pathlib import Path

from .emotion import ALL_TAGS

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = BASE_DIR / "data" / "projects"

REQUIRED_STR_FIELDS = ("name", "script")
OPTIONAL_FIELDS = {
    "voice_id": str,
    "preset_id": str,
    "speed": (int, float),
    "pause_sentence_sec": (int, float),
    "pause_chunk_sec": (int, float),
    "pause_paragraph_sec": (int, float),
    "voice_gain": (int, float),
    "auto_emotion": bool,
    "emo_overrides": dict,
}


def projects_dir() -> Path:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


def new_project_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def validate_project_dict(d: object) -> tuple[dict | None, list[str]]:
    """Validate an uploaded/loaded project dict. Returns (cleaned, errors)."""
    if not isinstance(d, dict):
        return None, ["প্রজেক্ট ফাইলটি সঠিক JSON অবজেক্ট নয়।"]
    errors: list[str] = []
    for f in REQUIRED_STR_FIELDS:
        if not isinstance(d.get(f), str) or not d.get(f, "").strip():
            errors.append(f"'{f}' থাকতে হবে (খালি নয়)।")
    for f, types in OPTIONAL_FIELDS.items():
        if f in d and d[f] is not None and not isinstance(d[f], types):
            errors.append(f"'{f}'-এর ধরন ভুল।")
    if isinstance(d.get("speed"), (int, float)) and not 0.5 <= d["speed"] <= 1.5:
        errors.append("গতি ০.৫–১.৫ এর মধ্যে হতে হবে।")
    for f in ("pause_sentence_sec", "pause_chunk_sec", "pause_paragraph_sec"):
        if isinstance(d.get(f), (int, float)) and not 0.0 <= d[f] <= 3.0:
            errors.append(f"'{f}' ০–৩ সেকেন্ডের মধ্যে হতে হবে।")
    if isinstance(d.get("voice_gain"), (int, float)) and not 0.5 <= d["voice_gain"] <= 1.5:
        errors.append("ভয়েস গেইন ০.৫–১.৫ এর মধ্যে হতে হবে।")
    if isinstance(d.get("emo_overrides"), dict):
        for k, v in d["emo_overrides"].items():
            if not isinstance(v, list) or any(t not in ALL_TAGS for t in v):
                errors.append(f"emo_overrides['{k}']-তে অচেনা ট্যাগ আছে।")
                break
    if errors:
        return None, errors
    cleaned = {
        "id": d.get("id") if isinstance(d.get("id"), str) and d["id"] else new_project_id(),
        "name": d["name"].strip(),
        "script": d["script"],
        "voice_id": d.get("voice_id", "tiny-female"),
        "preset_id": d.get("preset_id", "none"),
        "speed": float(d.get("speed", 1.0)),
        "pause_sentence_sec": float(d.get("pause_sentence_sec", 0.3)),
        "pause_chunk_sec": float(d.get("pause_chunk_sec", 0.35)),
        "pause_paragraph_sec": float(d.get("pause_paragraph_sec", 0.5)),
        "voice_gain": float(d.get("voice_gain", 1.0)),
        "auto_emotion": bool(d.get("auto_emotion", True)),
        "emo_overrides": d.get("emo_overrides", {}),
        "created_at": d.get("created_at", _now()),
        "updated_at": _now(),
    }
    return cleaned, []


def save_project(data: dict) -> tuple[bool, str]:
    """Validate + write a project JSON file. Returns (ok, message)."""
    cleaned, errors = validate_project_dict(data)
    if errors:
        return False, " ".join(errors)
    try:
        d = projects_dir()
        path = d / f"{cleaned['id']}.json"
        if not path.exists():
            cleaned["created_at"] = _now()
        cleaned["updated_at"] = _now()
        path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return True, f"প্রজেক্ট '{cleaned['name']}' সংরক্ষিত হয়েছে।"
    except OSError as e:
        return False, f"ডিস্কে লেখা যায়নি: {e}"


def list_projects() -> tuple[list[dict], int]:
    """Return (projects sorted by updated_at desc, corrupt_file_count)."""
    try:
        d = projects_dir()
    except OSError:
        return [], 0
    out: list[dict] = []
    corrupt = 0
    for path in sorted(d.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cleaned, errors = validate_project_dict(data)
            if errors:
                corrupt += 1
                continue
            out.append(cleaned)
        except (OSError, ValueError, UnicodeDecodeError):
            corrupt += 1
    out.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
    return out, corrupt


def load_project(pid: str) -> dict | None:
    try:
        path = projects_dir() / f"{pid}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        cleaned, errors = validate_project_dict(data)
        return cleaned if not errors else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def delete_project(pid: str) -> bool:
    try:
        (projects_dir() / f"{pid}.json").unlink(missing_ok=True)
        return True
    except OSError:
        return False


def rename_project(pid: str, new_name: str) -> tuple[bool, str]:
    if not new_name or not new_name.strip():
        return False, "নাম খালি রাখা যাবে না।"
    data = load_project(pid)
    if data is None:
        return False, "প্রজেক্ট পাওয়া যায়নি।"
    data["name"] = new_name.strip()
    return save_project(data)


def project_to_bytes(data: dict) -> bytes:
    """Serialize a project for download (validated first)."""
    cleaned, errors = validate_project_dict(data)
    if errors:
        raise ValueError(" ".join(errors))
    return json.dumps(cleaned, ensure_ascii=False, indent=2).encode("utf-8")


def project_from_bytes(raw: bytes) -> tuple[dict | None, list[str]]:
    """Parse + validate an uploaded .json file. Returns (project, errors)."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, ["ফাইলটি সঠিক JSON নয়।"]
    return validate_project_dict(data)


def projects_disk_usage() -> tuple[int, int]:
    """Return (file_count, total_bytes) of stored project files."""
    try:
        files = list(projects_dir().glob("*.json"))
        return len(files), sum(f.stat().st_size for f in files)
    except OSError:
        return 0, 0
