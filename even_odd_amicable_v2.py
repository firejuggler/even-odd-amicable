#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# even_odd_amicable_v2.py
# -----------------------
# Recherche de paires amicales pair-impair. Revision 2 du prototype.

from __future__ import annotations

import argparse
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Iterator

# ------------------------------------------------------------------ #
#  1. Cribles                                                        #
# ------------------------------------------------------------------ #

def build_primes(n: int) -> list[int]:
    """Eratosthene classique jusqu'a n."""
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0] = sieve[1] = 0
    limit = math.isqrt(n)
    for i in range(2, limit + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(n + 1) if sieve[i]]


def build_spf(n: int) -> list[int]:
    """spf[k] = plus petit facteur premier de k (k >= 2)."""
    spf = [0] * (n + 1)
    if n >= 1:
        spf[1] = 1
    primes: list[int] = []
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            v = i * p
            if v > n or p > spf[i]:
                break
            spf[v] = p
    return spf


def build_sigma_square_sieve(n: int, spf: list[int] | None = None) -> list[int]:
    """sigma_sq[k] = sigma(k^2) pour k dans [0, n].

    Version de production: crible par premiers et multiples.
    En CPython, cette variante est souvent plus rapide que SPF pur Python.
    """
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


def build_sigma_square_sieve_spf(n: int, spf: list[int] | None = None) -> list[int]:
    """Version sigma(k^2) basee SPF (utile pour benchmark/experimentation)."""
    if spf is None:
        spf = build_spf(n)
    return build_sigma_square_sieve(n, spf)


def build_omega(n: int, spf: list[int] | None = None) -> list[int]:
    """omega[k] = nombre de facteurs premiers distincts de k.
    Utilise pour appliquer la contrainte de Kanold (omega(s) >= 2).

    Note perf:
      En CPython, cette version "par premiers et multiples" reste en
      pratique plus rapide que la version SPF pure-Python.
    """
    omega = [0] * (n + 1)
    if spf is None:
        for p in build_primes(n):
            for k in range(p, n + 1, p):
                omega[k] += 1
        return omega

    for k in range(2, n + 1):
        p = spf[k]
        q = k // p
        omega[k] = omega[q] if (q % p == 0) else (omega[q] + 1)
    return omega


def build_omega_spf(n: int, spf: list[int] | None = None) -> list[int]:
    """Version omega basee SPF (utile pour benchmark/experimentation)."""
    if spf is None:
        spf = build_spf(n)
    return build_omega(n, spf)


# ------------------------------------------------------------------ #
#  2. Test "est-ce un carre ?"                                       #
# ------------------------------------------------------------------ #

_QR_MODULI = (9, 5, 7, 11, 13, 17, 19, 23, 29, 31)   # mod 8 teste a part
_QR_SETS = tuple({(x * x) % m for x in range(m)} for m in _QR_MODULI)


def is_square_fast(n: int) -> bool:
    """True ssi n est un carre parfait.  Suppose n >= 0."""
    for m, qr in zip(_QR_MODULI, _QR_SETS):
        if (n % m) not in qr:
            return False
    r = math.isqrt(n)
    return r * r == n


# ------------------------------------------------------------------ #
#  3. Factorisation legere et Miller-Rabin                           #
# ------------------------------------------------------------------ #

def trial_factor(n: int, bound: int) -> tuple[dict[int, int], int]:
    """Division d'essai jusqu'a `bound`.  Retourne (facteurs, reste).
    Le reste peut valoir 1 (completement factorise), un premier, ou un
    composite a cofacteurs > bound."""
    fact: dict[int, int] = {}
    e = 0
    while n % 2 == 0:
        n //= 2
        e += 1
    if e:
        fact[2] = e
    p = 3
    while p <= bound and p * p <= n:
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            fact[p] = e
        p += 2
    return fact, n


def is_probable_prime(n: int) -> bool:
    """Miller-Rabin deterministe pour n < 3.3 * 10^24."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def sigma_from_factorization(fact: dict[int, int]) -> int:
    s = 1
    for p, e in fact.items():
        s *= (pow(p, e + 1) - 1) // (p - 1)
    return s


def sigma_if_easy(m: int, bound: int = 10_000_000) -> int | None:
    """Renvoie sigma(m) si m se factorise facilement, sinon None."""
    fact, rest = trial_factor(m, bound)
    if rest == 1:
        return sigma_from_factorization(fact)
    if is_probable_prime(rest):
        fact[rest] = fact.get(rest, 0) + 1
        return sigma_from_factorization(fact)
    return None  # cofacteur composite non factorise


# ------------------------------------------------------------------ #
#  4. Persistance SQLite                                             #
# ------------------------------------------------------------------ #

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    s         INTEGER PRIMARY KEY,
    n         TEXT NOT NULL,
    m         TEXT NOT NULL,
    sigma_n   TEXT NOT NULL,
    status    TEXT NOT NULL            -- 'survived', 'amicable', 'hard'
);
CREATE TABLE IF NOT EXISTS checkpoint (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    s_last_done   INTEGER NOT NULL
);
"""

def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def get_checkpoint(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT s_last_done FROM checkpoint WHERE id = 1").fetchone()
    return row[0] if row else None


def set_checkpoint(conn: sqlite3.Connection, s: int) -> None:
    with conn:
        conn.execute(
            "INSERT INTO checkpoint(id, s_last_done) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET s_last_done = excluded.s_last_done",
            (s,),
        )


def save_candidate(conn: sqlite3.Connection, c: "Candidate", status: str) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO candidates(s, n, m, sigma_n, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (c.s, str(c.n), str(c.m), str(c.sigma_n), status),
        )


# ------------------------------------------------------------------ #
#  5. Scan                                                           #
# ------------------------------------------------------------------ #

@dataclass
class Candidate:
    s: int
    n: int
    m: int
    sigma_n: int


@dataclass
class Stats:
    scanned: int = 0
    kept_fast: int = 0       # m > 0 pair, m != n
    kept_mod8: int = 0       # partie impaire ≡ 1 (mod 8)
    kept_square: int = 0     # is_square_fast OK
    hits: int = 0            # paires amicales trouvees
    hard: int = 0            # m non factorise

    def summary(self) -> str:
        return (f"scanned={self.scanned}  kept_fast={self.kept_fast}  "
                f"kept_mod8={self.kept_mod8}  kept_square={self.kept_square}  "
                f"hard={self.hard}  hits={self.hits}")


def scan(s_min: int, s_max: int,
         sigma_sq: list[int], omega: list[int],
         m_max: int | None,
         conn: sqlite3.Connection,
         verbose_every: int,
         checkpoint_every: int,
         stats: Stats) -> Iterator[Candidate]:
    """Itere sur s impairs et yield les candidats survivant aux filtres."""
    if s_min % 2 == 0:
        s_min += 1

    t0 = time.time()
    last_print_s = s_min

    for s in range(s_min, s_max + 1, 2):
        stats.scanned += 1

        # Kanold : n = s^2 doit avoir au moins 2 premiers distincts
        # => equivalent a omega(s) >= 2.
        if omega[s] < 2:
            continue

        sig_n = sigma_sq[s]
        n = s * s
        m = sig_n - n

        if m > 0 and (m & 1) == 0 and m != n and (m_max is None or m <= m_max):
            stats.kept_fast += 1

            v2 = (m & -m).bit_length() - 1
            odd_part = m >> v2

            if (odd_part & 7) == 1:     # carre impair ≡ 1 (mod 8)
                stats.kept_mod8 += 1

                if is_square_fast(odd_part):
                    stats.kept_square += 1
                    cand = Candidate(s=s, n=n, m=m, sigma_n=sig_n)
                    save_candidate(conn, cand, "survived")
                    yield cand

        # progression (hors chemin candidat)
        if verbose_every and s - last_print_s >= verbose_every:
            elapsed = time.time() - t0
            rate = (s - s_min) / elapsed if elapsed > 0 else 0
            print(f"[s={s:>12}]  {stats.summary()}  ({rate:,.0f} s/s)")
            last_print_s = s

        # checkpoint
        if checkpoint_every and s % checkpoint_every == 1:
            set_checkpoint(conn, s)

    set_checkpoint(conn, s_max)


def verify(c: Candidate, trial_bound: int = 10_000_000) -> tuple[bool, bool]:
    # Retourne (amicale?, hard?) ; hard = m non factorise avec la borne.
    sigma_m = sigma_if_easy(c.m, trial_bound)
    if sigma_m is None:
        return False, True
    return sigma_m == c.sigma_n, False


# ------------------------------------------------------------------ #
#  6. Main                                                           #
# ------------------------------------------------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--s-min", type=int, default=3)
    ap.add_argument("--s-max", type=int, default=1_000_000)
    ap.add_argument("--m-max", type=int, default=None,
                    help="borne sup. sur le candidat pair m (defaut : aucune)")
    ap.add_argument("--db", default="even_odd_amicable.sqlite")
    ap.add_argument("--verbose-every", type=int, default=500_000,
                    help="frequence d'affichage par s parcouru")
    ap.add_argument("--checkpoint-every", type=int, default=1_000_000)
    ap.add_argument("--trial-bound", type=int, default=10_000_000,
                    help="borne de la division d'essai pour sigma(m)")
    ap.add_argument("--resume", action="store_true",
                    help="reprendre a partir du checkpoint en base")
    args = ap.parse_args()

    conn = open_db(args.db)
    s_min = args.s_min
    if args.resume:
        cp = get_checkpoint(conn)
        if cp is not None and cp + 1 > s_min:
            s_min = cp + 1
            print(f"# Reprise a partir de s = {s_min}")

    if s_min > args.s_max:
        print("# Rien a faire.")
        return

    print(f"# Cribles jusqu'a s = {args.s_max} ...")
    t0 = time.time()
    sigma_sq = build_sigma_square_sieve(args.s_max)
    omega = build_omega(args.s_max)
    print(f"# Cribles OK en {time.time() - t0:.2f}s\n")

    print(f"# Scan s = [{s_min}, {args.s_max}]\n")
    stats = Stats()
    t0 = time.time()

    for cand in scan(s_min, args.s_max, sigma_sq, omega,
                     args.m_max, conn, args.verbose_every,
                     args.checkpoint_every, stats):
        print(f"-- survivant : s={cand.s}  m={cand.m}  n={cand.n}")
        ok, hard = verify(cand, args.trial_bound)
        if hard:
            stats.hard += 1
            save_candidate(conn, cand, "hard")
            print(f"   [HARD] m non factorise avec bound={args.trial_bound}")
        elif ok:
            stats.hits += 1
            save_candidate(conn, cand, "amicable")
            print(f"   *** PAIRE AMICALE PAIR-IMPAIR : ({cand.m}, {cand.n}) ***")

    elapsed = time.time() - t0
    print(f"\n# Fini en {elapsed:.2f}s")
    print(f"# Stats : {stats.summary()}")


if __name__ == "__main__":
    main()
