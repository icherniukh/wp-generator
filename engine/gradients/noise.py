"""Vectorized 2D simplex and Perlin noise, ported from the `noise` package.

This is a numpy port of ``snoise2`` and ``pnoise2`` from noise 1.2.2
(``_simplex.c``, ``_perlin.c`` and ``_noise.h``). It uses the same permutation
table, gradient table, octave loops and normalization, so the same coordinates
give the same values and existing seeds render the same images.

Precision: the C code computes in 32-bit ``float``. This port does every step
in float32 in the same order as the C source, so it follows a build that
evaluates each operation separately. Compilers that contract ``a*b + c`` into
fused multiply-add (Apple clang on arm64 does) round some steps once instead
of twice, which moves the result by a few float32 ulp. Against such a build the
largest absolute difference measured is 3.6e-7 (see ``tests/test_noise.py``).

Only the parts the generator uses are ported: ``snoise2`` without tiling
(``repeatx``/``repeaty`` are not supported) and ``pnoise2`` with its repeat
and base arguments. ``pnoise2`` accepts ``base`` 0 or 1 only. Larger values
index past the 512-entry permutation table, which is undefined behaviour in
the C code and an error here.

Original code: Copyright (c) 2008 Casey Duncan, MIT License:

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["snoise2", "pnoise2"]

_F32 = np.float32

# Ken Perlin's permutation table, stored twice (512 entries) as in _noise.h.
# int32, not uint8, so sums such as I + PERM[J] do not wrap at 256.
_PERM_256 = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103,
    30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197,
    62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20,
    125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231,
    83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102,
    143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200,
    196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226,
    250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47,
    16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79,
    113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210,
    144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
    181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236,
    205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
]  # fmt: skip
_PERM = np.array(_PERM_256 * 2, dtype=np.int32)

# GRAD3 from _noise.h; only the x and y components are used in 2D.
_GRAD3 = np.array(
    [
        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
        (1, 0, -1), (-1, 0, -1), (0, -1, 1), (0, 1, 1),
    ],
    dtype=np.float32,
)  # fmt: skip
_GX = _GRAD3[:, 0]
_GY = _GRAD3[:, 1]

# 2D simplex skew factors from _simplex.c, as C float literals.
_F2 = _F32(0.3660254037844386)  # 0.5 * (sqrt(3.0) - 1.0)
_G2 = _F32(0.21132486540518713)  # (3.0 - sqrt(3.0)) / 6.0
_G2X2 = _G2 * _F32(2.0)


def _as_f32_pair(x: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    # PyArg "f" rounds each double argument to float; do the same up front.
    x = np.asarray(x, dtype=np.float64).astype(np.float32)
    y = np.asarray(y, dtype=np.float64).astype(np.float32)
    return np.broadcast_arrays(x, y)


def _simplex_corner(xx: np.ndarray, yy: np.ndarray, g: np.ndarray) -> np.ndarray:
    f = _F32(0.5) - xx * xx - yy * yy
    contrib = f * f * f * f * (_GX[g] * xx + _GY[g] * yy)
    return np.where(f > 0, contrib, _F32(0.0))


def _simplex2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Single-octave 2D simplex noise: noise2() in _simplex.c."""
    s = (x + y) * _F2
    i = np.floor(x + s)
    j = np.floor(y + s)
    t = (i + j) * _G2

    xx0 = x - (i - t)
    yy0 = y - (j - t)

    i1 = (xx0 > yy0).astype(np.int32)
    j1 = (xx0 <= yy0).astype(np.int32)

    xx2 = xx0 + _G2X2 - _F32(1.0)
    yy2 = yy0 + _G2X2 - _F32(1.0)
    xx1 = xx0 - i1.astype(np.float32) + _G2
    yy1 = yy0 - j1.astype(np.float32) + _G2

    ii = i.astype(np.int32) & 255
    jj = j.astype(np.int32) & 255
    g0 = _PERM[ii + _PERM[jj]] % 12
    g1 = _PERM[ii + i1 + _PERM[jj + j1]] % 12
    g2 = _PERM[ii + 1 + _PERM[jj + 1]] % 12

    n0 = _simplex_corner(xx0, yy0, g0)
    n1 = _simplex_corner(xx1, yy1, g1)
    n2 = _simplex_corner(xx2, yy2, g2)
    return (n0 + n1 + n2) * _F32(70.0)


def snoise2(
    x: ArrayLike,
    y: ArrayLike,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    base: float = 0.0,
) -> np.ndarray:
    """Fractal 2D simplex noise over arrays; matches ``noise.snoise2`` without tiling.

    ``base`` is the float offset that ``noise.snoise2`` adds to both coordinates.
    Returns a float32 array of the broadcast shape of ``x`` and ``y``.
    """
    if octaves <= 0:
        raise ValueError("Expected octaves value > 0")
    x, y = _as_f32_pair(x, y)
    z = _F32(base)
    persistence = _F32(persistence)
    lacunarity = _F32(lacunarity)

    freq = _F32(1.0)
    amp = _F32(1.0)
    total_max = _F32(1.0)
    total = _simplex2(x + z, y + z)
    for _ in range(1, octaves):
        freq = freq * lacunarity
        amp = amp * persistence
        total_max = total_max + amp
        total = total + _simplex2(x * freq + z, y * freq + z) * amp
    return total / total_max


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * _F32(6.0) - _F32(15.0)) + _F32(10.0))


def _lerp(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _grad2(hash_: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = hash_ & 15
    return x * _GX[h] + y * _GY[h]


def _perlin2(
    x: np.ndarray, y: np.ndarray, repeatx: np.float32, repeaty: np.float32, base: int
) -> np.ndarray:
    """Single-octave improved Perlin noise: noise2() in _perlin.c."""
    i = np.floor(np.fmod(x, repeatx)).astype(np.int32)
    j = np.floor(np.fmod(y, repeaty)).astype(np.int32)
    ii = np.fmod((i + 1).astype(np.float32), repeatx).astype(np.int32)
    jj = np.fmod((j + 1).astype(np.float32), repeaty).astype(np.int32)
    i = (i & 255) + base
    j = (j & 255) + base
    ii = (ii & 255) + base
    jj = (jj & 255) + base

    x = x - np.floor(x)
    y = y - np.floor(y)
    fx = _fade(x)
    fy = _fade(y)

    a = _PERM[i]
    aa = _PERM[a + j]
    ab = _PERM[a + jj]
    b = _PERM[ii]
    ba = _PERM[b + j]
    bb = _PERM[b + jj]

    one = _F32(1.0)
    return _lerp(
        fy,
        _lerp(fx, _grad2(_PERM[aa], x, y), _grad2(_PERM[ba], x - one, y)),
        _lerp(fx, _grad2(_PERM[ab], x, y - one), _grad2(_PERM[bb], x - one, y - one)),
    )


def pnoise2(
    x: ArrayLike,
    y: ArrayLike,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    repeatx: float = 1024,
    repeaty: float = 1024,
    base: int = 0,
) -> np.ndarray:
    """Fractal 2D improved Perlin noise over arrays; matches ``noise.pnoise2``.

    ``base`` must be 0 or 1 (see the module docstring). Returns a float32 array
    of the broadcast shape of ``x`` and ``y``.
    """
    if octaves <= 0:
        raise ValueError("Expected octaves value > 0")
    base = int(base)
    if base not in (0, 1):
        raise ValueError("base must be 0 or 1; larger values read past the permutation table")
    x, y = _as_f32_pair(x, y)
    persistence = _F32(persistence)
    lacunarity = _F32(lacunarity)
    repeatx = _F32(repeatx)
    repeaty = _F32(repeaty)

    if octaves == 1:
        return _perlin2(x, y, repeatx, repeaty, base)

    freq = _F32(1.0)
    amp = _F32(1.0)
    total_max = _F32(0.0)
    total = np.zeros_like(x)
    for _ in range(octaves):
        total = total + _perlin2(x * freq, y * freq, repeatx * freq, repeaty * freq, base) * amp
        total_max = total_max + amp
        freq = freq * lacunarity
        amp = amp * persistence
    return total / total_max
