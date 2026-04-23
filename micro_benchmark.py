#!/usr/bin/env python3
# Micro-benchmark local pour comparer:
#   - version SPF (actuelle) de sigma(k^2) / omega(k)
#   - version "legacy" (crible par premiers et multiples)
"""
Micro-benchmark local pour comparer:
  - version SPF (actuelle) de sigma(k^2) / omega(k)
  - version "legacy" (crible par premiers et multiples)
"""

from __future__ import annotations

import argparse
import math
import time

from even_odd_amicable_v2 import (
    build_omega,
    build_omega_spf,
    build_sigma_square_sieve,
    build_sigma_square_sieve_spf,
    build_spf,
)


def build_primes_legacy(n: int) -> list[int]:
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, math.isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(n + 1) if sieve[i]]


def build_sigma_square_sieve_legacy(n: int) -> list[int]:
    sigma_sq = [1] * (n + 1)
    sigma_sq[0] = 0
    for p in build_primes_legacy(n):
        for k in range(p, n + 1, p):
            kk = k
            e = 0
            while kk % p == 0:
                kk //= p
                e += 1
            sigma_sq[k] *= (pow(p, 2 * e + 1) - 1) // (p - 1)
    return sigma_sq


def build_omega_legacy(n: int) -> list[int]:
    omega = [0] * (n + 1)
    for p in build_primes_legacy(n):
        for k in range(p, n + 1, p):
            omega[k] += 1
    return omega


def timed(label: str, fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    dt = time.perf_counter() - t0
    print(f"{label:<30} {dt:>9.4f}s")
    return out, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="borne max des cribles")
    ap.add_argument("--check-n", type=int, default=20_000, help="taille de verif d'egalite")
    args = ap.parse_args()

    print(f"# Micro-benchmark sur n={args.n:,}")
    print(f"# Verification d'egalite sur n={args.check_n:,}\n")

    sig_old, t_sig_old = timed("legacy sigma(k^2)", build_sigma_square_sieve_legacy, args.n)
    sig_new, t_sig_new = timed("spf sigma(k^2)", build_sigma_square_sieve_spf, args.n)
    print(f"speedup sigma: x{(t_sig_old / t_sig_new):.2f}\n")

    om_old, t_om_old = timed("legacy omega(k)", build_omega_legacy, args.n)
    spf, t_spf = timed("build_spf(n)", build_spf, args.n)
    om_new, t_om_new = timed("spf omega(k)", build_omega_spf, args.n, spf)
    print(f"speedup omega (hors SPF): x{(t_om_old / t_om_new):.2f}")
    print(f"speedup omega (avec SPF): x{(t_om_old / (t_om_new + t_spf)):.2f}\n")

    t_legacy_total = t_sig_old + t_om_old
    t_spf_total = t_spf + t_sig_new + t_om_new
    print(f"legacy total (sigma+omega):      {t_legacy_total:>9.4f}s")
    print(f"spf total (spf+sigma+omega):     {t_spf_total:>9.4f}s")
    print(f"speedup total precompute: x{(t_legacy_total / t_spf_total):.2f}\n")

    sig_prod, t_sig_prod = timed("prod sigma(k^2) [current]", build_sigma_square_sieve, args.n)
    om_prod, t_om_prod = timed("prod omega(k) [current]", build_omega, args.n)
    t_prod_total = t_sig_prod + t_om_prod
    print(f"prod total (sigma+omega):        {t_prod_total:>9.4f}s")
    print(f"speedup total (prod vs legacy): x{(t_legacy_total / t_prod_total):.2f}\n")

    c = args.check_n
    assert sig_old[: c + 1] == sig_new[: c + 1], "Mismatch sigma"
    assert sig_old[: c + 1] == sig_prod[: c + 1], "Mismatch sigma_prod"
    assert om_old[: c + 1] == om_new[: c + 1], "Mismatch omega_spf"
    assert om_old[: c + 1] == om_prod[: c + 1], "Mismatch omega_prod"
    print("OK: egalite verifiee sur prefixe.")


if __name__ == "__main__":
    main()
