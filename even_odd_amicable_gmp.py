#!/usr/bin/env python3
# even_odd_amicable_gmp.py — driver GMP segmente (s_max illimite).
from __future__ import annotations
import argparse
import time

import even_odd_core_gmp
from even_odd_amicable_v2 import (
    open_db, get_checkpoint, set_checkpoint, save_candidate,
    sigma_if_easy, Stats, Candidate,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s-min", type=int, default=3)
    ap.add_argument("--s-max", type=int, default=1_000_000)
    ap.add_argument("--m-max", type=int, default=0)
    ap.add_argument("--db", default="even_odd_amicable.sqlite")
    ap.add_argument("--block-size", type=int, default=100_000,
                    help="taille segment (GMP est plus couteux, ~100k recommande)")
    ap.add_argument("--trial-bound", type=int, default=10_000_000)
    ap.add_argument("--resume", action="store_true")
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

    m_max_arg = args.m_max if args.m_max and args.m_max > 0 else None
    stats = Stats()
    t0 = time.time()

    def on_segment(survivors, stats_delta, segment_end):
        stats.scanned     += stats_delta["scanned"]
        stats.kept_fast   += stats_delta["kept_fast"]
        stats.kept_mod8   += stats_delta["kept_mod8"]
        stats.kept_square += stats_delta["kept_square"]

        for s, n, m, sig_n in survivors:
            cand = Candidate(s=s, n=n, m=m, sigma_n=sig_n)
            save_candidate(conn, cand, "survived")
            sig_m = sigma_if_easy(m, args.trial_bound)
            if sig_m is None:
                stats.hard += 1
                save_candidate(conn, cand, "hard")
                print(f"[HARD]  s={s}  m_bits={m.bit_length()}")
            elif sig_m == sig_n:
                stats.hits += 1
                save_candidate(conn, cand, "amicable")
                print(f"*** AMICALE PAIR-IMPAIR : s={s}  m={m}  n={n} ***")

        set_checkpoint(conn, segment_end)
        elapsed = time.time() - t0
        rate = stats.scanned / elapsed if elapsed > 0 else 0
        print(f"[s<={segment_end:>14}] {stats.summary()}  ({rate:,.0f} s/s)")
        return True

    print(f"# Scan GMP segmente [s_min={s_min}, s_max={args.s_max}] "
          f"block_size={args.block_size}")
    even_odd_core_gmp.scan_segmented_gmp(
        s_min, args.s_max, m_max_arg, args.block_size, on_segment)

    print(f"\n# Fini en {time.time() - t0:.2f}s")
    print(f"# Stats : {stats.summary()}")


if __name__ == "__main__":
    main()
