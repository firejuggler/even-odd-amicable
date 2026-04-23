#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import ast
import sys


FILES = [
    "even_odd_amicable_v2.py",
    "even_odd_amicable_gmp.py",
    "micro_benchmark.py",
    "bench.txt",
]

MARKERS = ("<<<<<<<", "=======", ">>>>>>>")
BAD_VERIFY = '"""Retourne (amicale?, hard?).  hard = m non factorise avec la borne."""'
BAD_SIGMA_LINE = "    sigma_sq[k] = sigma(k^2) pour k dans [0, n]."


def _repair_source(src: str) -> tuple[str, bool]:
    repaired = False
    if any(line.startswith(MARKERS) for line in src.splitlines()):
        src = "\n".join(
            line for line in src.splitlines()
            if not line.startswith(MARKERS)
        ) + "\n"
        repaired = True

    if BAD_VERIFY in src:
        src = src.replace(
            BAD_VERIFY,
            "# Retourne (amicale?, hard?) ; hard = m non factorise avec la borne.",
        )
        repaired = True

    if BAD_SIGMA_LINE in src:
        src = src.replace(
            BAD_SIGMA_LINE,
            "    # sigma_sq[k] = sigma(k^2) pour k dans [0, n].",
        )
        repaired = True
    return src, repaired


def check_file(path: Path) -> int:
    errors = 0
    src = path.read_text(encoding="utf-8")

    if any(line.startswith(MARKERS) for line in src.splitlines()):
        print(f"[CONFLICT] {path}: marqueurs de merge detectes.")
        errors += 1

    try:
        ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        fixed_src, repaired = _repair_source(src)
        if repaired:
            try:
                ast.parse(fixed_src, filename=str(path))
                path.write_text(fixed_src, encoding="utf-8")
                print(f"[AUTO-FIX] {path}: syntaxe reparée automatiquement.")
                src = fixed_src
            except SyntaxError:
                print(f"[SYNTAX]   {path}:{exc.lineno} -> {exc.msg}")
                errors += 1
                return errors
        else:
            print(f"[SYNTAX]   {path}:{exc.lineno} -> {exc.msg}")
            errors += 1
            return errors

    tree = ast.parse(src, filename=str(path))
    seen: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen:
                print(
                    f"[DUPLICATE] {path}: '{node.name}' "
                    f"ligne {node.lineno} (deja ligne {seen[node.name]})"
                )
                errors += 1
            else:
                seen[node.name] = node.lineno

    return errors


def main() -> int:
    total_errors = 0
    for rel in FILES:
        p = Path(rel)
        if not p.exists():
            print(f"[MISSING]  {p}")
            total_errors += 1
            continue
        total_errors += check_file(p)

    if total_errors == 0:
        print("OK: aucun conflit, aucune duplication top-level, aucune erreur de syntaxe.")
        return 0
    print(f"ECHEC: {total_errors} probleme(s) detecte(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
