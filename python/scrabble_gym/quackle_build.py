"""Build Quackle's C++ core and a local GPL-licensed static-policy bridge."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .build import ROOT


TARGET = ROOT / "target" / "quackle"


def build_lexicon(source: Path, output: Path) -> int:
    """Write Quackle's supported v0 seven-byte DAWG format using a prefix trie."""
    words = sorted({
        w for w in source.read_text(encoding="utf-8").split()
        if w.isascii() and w.isalpha() and w.isupper() and 2 <= len(w) <= 15
    })
    if not words:
        raise ValueError("No supported dictionary words")
    root: dict = {}
    for word in words:
        node = root
        for letter in word:
            node = node.setdefault(letter, {})
        node[""] = {}
    records = [bytearray(7)]

    def emit(node: dict) -> int:
        children = [(letter, child) for letter, child in sorted(node.items()) if letter]
        if not children:
            return 0
        start = len(records)
        records.extend(bytearray(7) for _ in children)
        for offset, (letter, child) in enumerate(children):
            pointer = emit(child)
            flags = ord(letter) - ord("A") | 128
            if "" in child:
                flags |= 32
            if offset == len(children) - 1:
                flags |= 64
            records[start + offset][:] = pointer.to_bytes(3, "big") + bytes([flags, 0, 0, 1])
        return start

    emit(root)
    output.write_bytes(b"".join(records))
    return len(words)


def build() -> Path:
    upstream = ROOT / "third_party" / "quackle"
    if not (upstream / "CMakeLists.txt").is_file():
        raise FileNotFoundError("Clone official Quackle into third_party\\quackle first")
    if shutil.which("cmake") is None:
        raise RuntimeError("CMake 3.16+ and a C++17 compiler are required")
    TARGET.mkdir(parents=True, exist_ok=True)
    build_dir = TARGET / ("msvc" if os.name == "nt" else "build")
    configure = ["cmake", "-S", str(ROOT / "native" / "quackle"), "-B", str(build_dir)]
    if os.name == "nt":
        configure.extend(["-G", "Visual Studio 17 2022", "-A", "x64"])
    else:
        configure.append("-DCMAKE_BUILD_TYPE=Release")
        if shutil.which("g++") and shutil.which("ninja"):
            configure.extend(["-G", "Ninja", f"-DCMAKE_CXX_COMPILER={shutil.which('g++')}"])
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", "2"],
                   check=True)
    executable = (build_dir / "Release" / "quackle_bridge.exe" if os.name == "nt"
                  else build_dir / "quackle_bridge")
    if not executable.is_file():
        raise FileNotFoundError("Bridge executable not produced")
    dictionary = ROOT / "docs" / "words.txt"
    count = build_lexicon(dictionary, TARGET / "demo.dawg")
    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    (TARGET / "manifest.json").write_text(json.dumps({
        "revision": revision, "words": count,
        "dictionary_sha256": hashlib.sha256(dictionary.read_bytes()).hexdigest(),
        "executable": str(executable.relative_to(ROOT)),
    }, indent=2), encoding="utf-8")
    return executable


if __name__ == "__main__":
    print(build())
