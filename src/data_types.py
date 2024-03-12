from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


@dataclass
class Source:
    dir_path: Path
    path: Path
    time: timedelta

    @property
    def relative_path(self):
        return self.path.relative_to(self.dir_path)

    def __lt__(self, other):
        return (self.path, self.time) < (other.path, other.time)

@dataclass
class InputImage:
    dir_path: Path
    path: Path
    source: Source

    @property
    def relative_path(self):
        return self.path.relative_to(self.dir_path)
