import json
import os
from typing import Dict, Any
from client.config import CHECKPOINT_FILE

class CheckpointManager:
    def __init__(self, filepath: str = CHECKPOINT_FILE):
        self.filepath = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint file ({e}). Starting fresh.")
                return {}
        return {}

    def get_completed_lines(self, rel_file_path: str) -> int:
        clean_key = rel_file_path.replace("\\", "/")
        file_info = self.data.get(clean_key, {})
        return file_info.get("completed_lines", 0)

    def is_file_completed(self, rel_file_path: str, total_lines: int) -> bool:
        completed = self.get_completed_lines(rel_file_path)
        return completed >= total_lines and total_lines > 0

    def update_checkpoint(self, rel_file_path: str, completed_lines: int):
        clean_key = rel_file_path.replace("\\", "/")
        if clean_key not in self.data:
            self.data[clean_key] = {}

        self.data[clean_key]["completed_lines"] = completed_lines
        self.save()

    def save(self):
        tmp_file = self.filepath + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, self.filepath)
