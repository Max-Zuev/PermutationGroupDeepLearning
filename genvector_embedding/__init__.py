# coding=utf-8
"""The analytic matrix embedding: transposition token (i, j) -> its d x d
permutation matrix, flattened.

Products of permutation matrices compose exactly as the underlying
permutations, so every generating-vector relation M(gamma_1)...M(gamma_r) = I
holds exactly by construction for elements moving at most d points -- the
closed-form optimum of the relation objective. See ``analytic.py``.

(A gradient-trained variant of this embedding existed earlier in the project
and was superseded by the closed form; see the git history.)
"""
