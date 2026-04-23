# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# distutils: language = c
"""
even_odd_core_gmp.pyx — crible segmente GMP/mpz (illimite en taille).

Prerequis Windows :
    vcpkg install gmp:x64-windows
    puis configurer INCLUDE/LIB ou utiliser setup_gmp.py.
Linux : apt-get install libgmp-dev.

API :
    scan_segmented_gmp(s_min, s_max, m_max_py, block_size, on_segment)
        s_min, s_max : Python int (illimite)
        m_max_py : None ou Python int
        on_segment(survivors, stats_delta, segment_end_int) -> bool
"""

from libc.stdint cimport int64_t, int32_t, uint64_t, uint8_t
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt as c_sqrt

import numpy as np
cimport numpy as cnp
cnp.import_array()


cdef extern from "gmp.h":
    ctypedef struct __mpz_struct:
        pass
    ctypedef __mpz_struct mpz_t[1]
    ctypedef unsigned long mp_bitcnt_t

    void mpz_init(mpz_t x)
    void mpz_clear(mpz_t x)
    void mpz_set(mpz_t rop, const mpz_t op)
    void mpz_set_ui(mpz_t rop, unsigned long val)
    unsigned long mpz_get_ui(const mpz_t op)
    int mpz_cmp(const mpz_t op1, const mpz_t op2)
    int mpz_cmp_ui(const mpz_t op1, unsigned long op2)
    int mpz_sgn(const mpz_t op)
    void mpz_add(mpz_t rop, const mpz_t op1, const mpz_t op2)
    void mpz_add_ui(mpz_t rop, const mpz_t op1, unsigned long op2)
    void mpz_sub(mpz_t rop, const mpz_t op1, const mpz_t op2)
    void mpz_mul(mpz_t rop, const mpz_t op1, const mpz_t op2)
    void mpz_mul_ui(mpz_t rop, const mpz_t op1, unsigned long op2)
    void mpz_fdiv_q_2exp(mpz_t rop, const mpz_t op1, mp_bitcnt_t op2)
    unsigned long mpz_fdiv_ui(const mpz_t op1, unsigned long op2)
    void mpz_divexact_ui(mpz_t rop, const mpz_t op1, unsigned long op2)
    int mpz_perfect_square_p(const mpz_t op)
    void mpz_sqrt(mpz_t rop, const mpz_t op)
    mp_bitcnt_t mpz_scan1(const mpz_t op, mp_bitcnt_t starting_bit)
    size_t mpz_sizeinbase(const mpz_t op, int base)
    char* mpz_get_str(char *str, int base, const mpz_t op)
    int mpz_set_str(mpz_t rop, const char *str, int base)


cdef object _mpz_to_pyint(mpz_t val):
    cdef size_t sz = mpz_sizeinbase(val, 10) + 2
    cdef char *buf = <char*> malloc(sz)
    if buf == NULL:
        raise MemoryError()
    try:
        mpz_get_str(buf, 10, val)
        return int(buf.decode('ascii'))
    finally:
        free(buf)


cdef void _pyint_to_mpz(mpz_t dst, object py_int):
    s = str(int(py_int)).encode('ascii')
    if mpz_set_str(dst, s, 10) != 0:
        raise ValueError(f"mpz_set_str failed on {py_int}")


cdef uint64_t QR_MOD8=0, QR_MOD9=0, QR_MOD5=0, QR_MOD7=0, QR_MOD11=0
cdef uint64_t QR_MOD13=0, QR_MOD17=0, QR_MOD19=0, QR_MOD23=0, QR_MOD29=0, QR_MOD31=0


cdef uint64_t _build_qr_mask(int mod):
    cdef uint64_t mask = 0
    cdef int x
    for x in range(mod):
        mask |= (<uint64_t>1) << ((x * x) % mod)
    return mask


cdef void _init_qr_masks():
    global QR_MOD8, QR_MOD9, QR_MOD5, QR_MOD7, QR_MOD11, QR_MOD13
    global QR_MOD17, QR_MOD19, QR_MOD23, QR_MOD29, QR_MOD31
    QR_MOD8  = _build_qr_mask(8)
    QR_MOD9  = _build_qr_mask(9)
    QR_MOD5  = _build_qr_mask(5)
    QR_MOD7  = _build_qr_mask(7)
    QR_MOD11 = _build_qr_mask(11)
    QR_MOD13 = _build_qr_mask(13)
    QR_MOD17 = _build_qr_mask(17)
    QR_MOD19 = _build_qr_mask(19)
    QR_MOD23 = _build_qr_mask(23)
    QR_MOD29 = _build_qr_mask(29)
    QR_MOD31 = _build_qr_mask(31)

_init_qr_masks()


cdef inline bint _is_square_mpz(mpz_t n):
    if not ((QR_MOD8  >> mpz_fdiv_ui(n, 8))  & 1):  return 0
    if not ((QR_MOD9  >> mpz_fdiv_ui(n, 9))  & 1):  return 0
    if not ((QR_MOD5  >> mpz_fdiv_ui(n, 5))  & 1):  return 0
    if not ((QR_MOD7  >> mpz_fdiv_ui(n, 7))  & 1):  return 0
    if not ((QR_MOD11 >> mpz_fdiv_ui(n, 11)) & 1):  return 0
    if not ((QR_MOD13 >> mpz_fdiv_ui(n, 13)) & 1):  return 0
    if not ((QR_MOD17 >> mpz_fdiv_ui(n, 17)) & 1):  return 0
    if not ((QR_MOD19 >> mpz_fdiv_ui(n, 19)) & 1):  return 0
    if not ((QR_MOD23 >> mpz_fdiv_ui(n, 23)) & 1):  return 0
    if not ((QR_MOD29 >> mpz_fdiv_ui(n, 29)) & 1):  return 0
    if not ((QR_MOD31 >> mpz_fdiv_ui(n, 31)) & 1):  return 0
    return mpz_perfect_square_p(n) != 0


cdef _build_small_primes(int64_t limit):
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] sieve = np.ones(limit + 1, dtype=np.uint8)
    cdef uint8_t[::1] sv = sieve
    cdef int64_t i, j
    sv[0] = 0
    sv[1] = 0
    for i in range(2, <int64_t>c_sqrt(<double>limit) + 1):
        if sv[i]:
            for j in range(i * i, limit + 1, i):
                sv[j] = 0
    return np.flatnonzero(sieve).astype(np.int64)


def scan_segmented_gmp(object s_min_py, object s_max_py,
                        object m_max_py,
                        int64_t block_size,
                        object on_segment):
    cdef mpz_t s_min_z, s_max_z, L, R, m_max_z, tmp
    cdef mpz_t sig_n, n_sq, m_z, odd_part, ppow_sum, pw_z
    cdef int64_t i, seg_len, k, e, j, pi, p
    cdef int64_t i_start, scanned, kept_fast, kept_mod8, kept_square
    cdef int64_t total_scanned = 0, total_fast = 0, total_mod8 = 0, total_square = 0
    cdef int64_t prime_limit
    cdef unsigned long r_ul
    cdef mp_bitcnt_t v2_
    cdef bint has_m_max = (m_max_py is not None) and (int(m_max_py) > 0)
    cdef bint stop = 0

    mpz_init(s_min_z); mpz_init(s_max_z); mpz_init(L); mpz_init(R)
    mpz_init(m_max_z); mpz_init(tmp)
    mpz_init(sig_n); mpz_init(n_sq); mpz_init(m_z)
    mpz_init(odd_part); mpz_init(ppow_sum); mpz_init(pw_z)

    _pyint_to_mpz(s_min_z, s_min_py)
    _pyint_to_mpz(s_max_z, s_max_py)
    if has_m_max:
        _pyint_to_mpz(m_max_z, m_max_py)

    # s_min >= 3, impair
    if mpz_cmp_ui(s_min_z, 3) < 0:
        mpz_set_ui(s_min_z, 3)
    if mpz_fdiv_ui(s_min_z, 2) == 0:
        mpz_add_ui(s_min_z, s_min_z, 1)

    # prime_limit = floor(sqrt(s_max)) + 1
    mpz_sqrt(tmp, s_max_z)
    mpz_add_ui(tmp, tmp, 1)
    # tmp doit tenir en ulong : s_max < 2^126 suffit
    prime_limit = <int64_t> mpz_get_ui(tmp)
    small_primes_arr = _build_small_primes(prime_limit)
    cdef int64_t[::1] small_primes = small_primes_arr
    cdef int64_t n_small = small_primes_arr.shape[0]

    # buffers segment
    cdef mpz_t *sig_arr  = <mpz_t*> malloc(block_size * sizeof(mpz_t))
    cdef mpz_t *rest_arr = <mpz_t*> malloc(block_size * sizeof(mpz_t))
    cdef int32_t *om_arr = <int32_t*> malloc(block_size * sizeof(int32_t))
    if sig_arr == NULL or rest_arr == NULL or om_arr == NULL:
        if sig_arr != NULL:  free(sig_arr)
        if rest_arr != NULL: free(rest_arr)
        if om_arr != NULL:   free(om_arr)
        raise MemoryError()
    for i in range(block_size):
        mpz_init(sig_arr[i])
        mpz_init(rest_arr[i])

    mpz_set(L, s_min_z)

    try:
        while mpz_cmp(L, s_max_z) <= 0:
            # R = min(L + block_size, s_max + 1)
            mpz_add_ui(R, L, <unsigned long>block_size)
            mpz_add_ui(tmp, s_max_z, 1)
            if mpz_cmp(R, tmp) > 0:
                mpz_set(R, tmp)
            mpz_sub(tmp, R, L)
            seg_len = <int64_t> mpz_get_ui(tmp)

            # init
            for i in range(seg_len):
                mpz_set_ui(sig_arr[i], 1)
                mpz_add_ui(rest_arr[i], L, <unsigned long>i)
                om_arr[i] = 0

            # crible
            for pi in range(n_small):
                p = small_primes[pi]
                if p > prime_limit:
                    break
                r_ul = mpz_fdiv_ui(L, <unsigned long>p)
                if r_ul == 0:
                    k = 0
                else:
                    k = p - <int64_t>r_ul
                while k < seg_len:
                    e = 0
                    while mpz_fdiv_ui(rest_arr[k], <unsigned long>p) == 0:
                        mpz_divexact_ui(rest_arr[k], rest_arr[k], <unsigned long>p)
                        e += 1
                    if e > 0:
                        # ppow_sum = 1 + p + p^2 + ... + p^(2e)
                        mpz_set_ui(ppow_sum, 0)
                        mpz_set_ui(pw_z, 1)
                        for j in range(2 * e + 1):
                            mpz_add(ppow_sum, ppow_sum, pw_z)
                            if j < 2 * e:
                                mpz_mul_ui(pw_z, pw_z, <unsigned long>p)
                        mpz_mul(sig_arr[k], sig_arr[k], ppow_sum)
                        om_arr[k] += 1
                    k += p

            # finalisation : rest > 1 => premier grand
            for i in range(seg_len):
                if mpz_cmp_ui(rest_arr[i], 1) > 0:
                    mpz_set_ui(ppow_sum, 1)
                    mpz_add(ppow_sum, ppow_sum, rest_arr[i])
                    mpz_mul(tmp, rest_arr[i], rest_arr[i])
                    mpz_add(ppow_sum, ppow_sum, tmp)
                    mpz_mul(sig_arr[i], sig_arr[i], ppow_sum)
                    om_arr[i] += 1

            # scan
            scanned = 0; kept_fast = 0; kept_mod8 = 0; kept_square = 0
            survivors = []

            if mpz_fdiv_ui(L, 2) == 1:
                i_start = 0
            else:
                i_start = 1

            for i in range(i_start, seg_len, 2):
                scanned += 1
                if om_arr[i] < 2:
                    continue

                mpz_set(sig_n, sig_arr[i])
                mpz_add_ui(tmp, L, <unsigned long>i)
                mpz_mul(n_sq, tmp, tmp)

                if mpz_cmp(sig_n, n_sq) <= 0:
                    continue
                mpz_sub(m_z, sig_n, n_sq)

                if mpz_fdiv_ui(m_z, 2) != 0:
                    continue
                if mpz_cmp(m_z, n_sq) == 0:
                    continue
                if has_m_max and mpz_cmp(m_z, m_max_z) > 0:
                    continue
                kept_fast += 1

                v2_ = mpz_scan1(m_z, 0)
                mpz_fdiv_q_2exp(odd_part, m_z, v2_)
                if mpz_fdiv_ui(odd_part, 8) != 1:
                    continue
                kept_mod8 += 1

                if not _is_square_mpz(odd_part):
                    continue
                kept_square += 1

                s_py = _mpz_to_pyint(L) + i
                survivors.append((
                    s_py,
                    _mpz_to_pyint(n_sq),
                    _mpz_to_pyint(m_z),
                    _mpz_to_pyint(sig_n),
                ))

            stats_delta = {
                "scanned": scanned,
                "kept_fast": kept_fast,
                "kept_mod8": kept_mod8,
                "kept_square": kept_square,
            }
            total_scanned += scanned
            total_fast    += kept_fast
            total_mod8    += kept_mod8
            total_square  += kept_square

            if on_segment is not None:
                end_py = _mpz_to_pyint(R) - 1
                cont = on_segment(survivors, stats_delta, end_py)
                if cont is False:
                    stop = 1

            if stop:
                break
            mpz_set(L, R)
    finally:
        for i in range(block_size):
            mpz_clear(sig_arr[i])
            mpz_clear(rest_arr[i])
        free(sig_arr); free(rest_arr); free(om_arr)
        mpz_clear(s_min_z); mpz_clear(s_max_z); mpz_clear(L); mpz_clear(R)
        mpz_clear(m_max_z); mpz_clear(tmp)
        mpz_clear(sig_n); mpz_clear(n_sq); mpz_clear(m_z)
        mpz_clear(odd_part); mpz_clear(ppow_sum); mpz_clear(pw_z)

    return {
        "scanned": total_scanned,
        "kept_fast": total_fast,
        "kept_mod8": total_mod8,
        "kept_square": total_square,
    }
