"""Build a standalone engine JAR using the JDK, without requiring Maven."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def build() -> Path:
    source = ROOT / "src" / "main"
    if not source.is_dir():
        raise FileNotFoundError("Run this helper from an editable repository installation")
    java_home = os.environ.get("JAVA_HOME")
    tools = {}
    for name in ("javac", "jar"):
        candidate = Path(java_home) / "bin" / name if java_home else None
        executable = shutil.which(str(candidate)) if candidate else shutil.which(name)
        if executable is None:
            raise RuntimeError(f"{name} not found; install JDK 17+ and set JAVA_HOME or PATH")
        tools[name] = executable
    target = ROOT / "target"
    target.mkdir(exist_ok=True)
    output = target / "scrabble-gym.jar"
    with tempfile.TemporaryDirectory(prefix="scrabble-build-") as directory:
        classes = Path(directory) / "classes"
        classes.mkdir()
        arguments = Path(directory) / "sources.txt"
        arguments.write_text(
            "\n".join(
                '"' + str(p.relative_to(source)).replace("\\", "\\\\") + '"'
                for p in sorted(source.rglob("*.java"))
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [tools["javac"], "--release", "17", "-encoding", "UTF-8",
             "-Xlint:all", "-Werror", "-d", str(classes), f"@{arguments}"],
            check=True, cwd=source,
        )
        subprocess.run(
            [tools["jar"], "--create", "--file", str(output),
             "-C", str(classes), ".", "-C", str(source / "resources"), "."],
            check=True,
        )
    return output


if __name__ == "__main__":
    print(build())
