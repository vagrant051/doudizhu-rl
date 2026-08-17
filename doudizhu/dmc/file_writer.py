"""精简版训练日志：写 CSV，不依赖 GitPython。"""

import csv
import json
import logging
import os
import time
from typing import Dict


class FileWriter:
    """把训练 stats 追加写到 checkpoints/<xpid>/logs.csv，并保存 meta.json。"""

    def __init__(self, xpid: str, xp_args: dict = None, rootdir: str = "checkpoints"):
        self.xpid = xpid
        self._tick = 0
        self.metadata = {"xpid": xpid, "args": dict(xp_args or {})}

        self._logger = logging.getLogger("doudizhu/out")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

        self.basepath = os.path.join(os.path.expanduser(rootdir), self.xpid)
        os.makedirs(self.basepath, exist_ok=True)
        self.paths = {
            "logs": os.path.join(self.basepath, "logs.csv"),
            "meta": os.path.join(self.basepath, "meta.json"),
        }
        with open(self.paths["meta"], "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)
        self.fieldnames = ["_tick", "_time"]

    def log(self, to_log: Dict) -> None:
        to_log = dict(to_log)
        to_log["_tick"] = self._tick
        to_log["_time"] = time.time()
        self._tick += 1
        for k in to_log:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        write_header = not os.path.exists(self.paths["logs"])
        with open(self.paths["logs"], "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(to_log)

    def close(self, successful: bool = True) -> None:
        self.metadata["successful"] = successful
        with open(self.paths["meta"], "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)
