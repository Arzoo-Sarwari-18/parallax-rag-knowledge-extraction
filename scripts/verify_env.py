"""
verify_env.py
-------------
Sanity-checks the local development environment before any pipeline work
begins. Confirms that:
  1. Core third-party libraries import successfully.
  2. PyTorch can see a CUDA-capable GPU (warns, does not hard-fail, if not).
  3. spaCy has at least one language model available.

Run:
    python scripts/verify_env.py

Exit codes:
    0  -> all required imports succeeded (GPU absence is a warning only)
    1  -> one or more required imports failed
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name, ok, detail))

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.results)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("ENVIRONMENT VERIFICATION SUMMARY")
        print("=" * 60)
        for r in self.results:
            status = "PASS" if r.ok else "FAIL"
            line = f"[{status}] {r.name}"
            if r.detail:
                line += f" — {r.detail}"
            print(line)
        print("=" * 60)
        overall = "ALL CHECKS PASSED" if self.all_ok else "SOME CHECKS FAILED"
        print(overall)
        print("=" * 60 + "\n")


REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "sentence_transformers",
    "chromadb",
    "spacy",
    "bs4",  # beautifulsoup4
    "langdetect",
    "dotenv",  # python-dotenv
]


def check_imports(report: Report) -> None:
    for mod_name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "unknown")
            report.add(f"import {mod_name}", True, f"version={version}")
        except ImportError as exc:
            report.add(f"import {mod_name}", False, str(exc))


def check_gpu(report: Report) -> None:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            report.add("CUDA / GPU", True, f"detected: {device_name}")
        else:
            # Not a hard failure: many contributors will prototype on CPU
            # (e.g. Apple Silicon, cloud CPU instances) and switch to a GPU
            # box for full-corpus embedding runs later.
            report.add(
                "CUDA / GPU",
                True,
                "no CUDA device found — CPU mode OK for now, "
                "but embedding 5k+ docs will be slow without a GPU",
            )
    except ImportError as exc:
        report.add("CUDA / GPU", False, f"torch not importable: {exc}")


def check_spacy_model(report: Report) -> None:
    try:
        import spacy

        try:
            spacy.load("en_core_web_sm")
            report.add("spaCy model (en_core_web_sm)", True, "loaded")
        except OSError:
            report.add(
                "spaCy model (en_core_web_sm)",
                False,
                "not installed — run: python -m spacy download en_core_web_sm",
            )
    except ImportError as exc:
        report.add("spaCy model (en_core_web_sm)", False, f"spacy not importable: {exc}")


def main() -> int:
    report = Report()
    check_imports(report)
    check_gpu(report)
    check_spacy_model(report)
    report.print_summary()

    # GPU absence is a warning, not a failure. Only missing required
    # imports fail the script (spaCy model download is a fixable warning).
    hard_failures = [
        r for r in report.results
        if not r.ok and not r.name.startswith("spaCy model")
    ]
    return 0 if not hard_failures else 1


if __name__ == "__main__":
    sys.exit(main())
