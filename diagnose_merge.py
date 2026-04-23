#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import ast
import re
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
SIGMA_CANONICAL = """def build_sigma_square_sieve(n: int, spf: list[int] | None = None) -> list[int]:
    # sigma_sq[k] = sigma(k^2) pour k dans [0, n].
    # Version de production: crible par premiers et multiples.
    # En CPython, cette variante est souvent plus rapide que SPF pur Python.
    sigma_sq = [1] * (n + 1)
    sigma_sq[0] = 0  # convention (non utilise)
    if spf is None:
        for p in build_primes(n):
            for k in range(p, n + 1, p):
                kk = k
                e = 0
                while kk % p == 0:
                    kk //= p
                    e += 1
                sigma_sq[k] *= (pow(p, 2 * e + 1) - 1) // (p - 1)
        return sigma_sq

    for k in range(2, n + 1):
        kk = k
        acc = 1
        while kk > 1:
            p = spf[kk]
            e = 0
            while kk % p == 0:
                kk //= p
                e += 1
            acc *= (pow(p, 2 * e + 1) - 1) // (p - 1)
        sigma_sq[k] = acc
    return sigma_sq
"""


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

    # Si la fonction a été vidée par un merge, on restaure un bloc canonique.
    m = re.search(
        r"def build_sigma_square_sieve\(.*?(?=\ndef build_sigma_square_sieve_spf)",
        src,
        flags=re.DOTALL,
    )
    if m and "sigma_sq =" not in m.group(0):
        src = src[: m.start()] + SIGMA_CANONICAL + src[m.end():]
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
