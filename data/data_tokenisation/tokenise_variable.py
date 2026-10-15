# coding=utf-8
"""Tokenise one group's orbit file into a CSV of token sequences (run under SageMath).

Input: ``{order}_{index}.sobj`` from the orbit stage, holding
``[[signature, classes], ...]``; ``classes`` is a list of topological
equivalence classes, each a list of generating vectors (gamma_1, ..., gamma_r)
of Sage permutation-group elements, stored in signature order.

Output: one CSV row per kept generating vector, with columns

    id    class id '{signature}#{k}': k is the class's position (from 1) among
          that signature's classes in the input file. The pair builder later
          appends '#{group}'.
    sig   the signature c_1_..._c_r
    r     the number of elements
    text  comma-separated tokens: r blocks of part_length tokens, block k being
          [order token of c_k, word tokens of gamma_k, -1 padding]

The word of an element: Sage's ``cycle_tuples()`` (fixed points omitted, each
cycle starting at its smallest point, cycles ordered by smallest point), with
each cycle (a_1, ..., a_k) contributing the pairs (a_1,a_2), (a_2,a_3), ...,
(a_{k-1},a_k). Composed right to left, as functions, these transpositions give
the element; under Sage/GAP's left-to-right product they give its inverse. The
word has the minimal length n - (number of cycles, counting fixed points).

Tokens, with n = largest_degree:

    pair (i, j)  ->  (i-1) + n*(j-1), in 0 .. n**2 - 1. Applied to the ordered
                     pair as it appears in the word, so (i, j) and (j, i) are
                     different tokens.
    order c      ->  n**2 + c (see order_token).
    -1           ->  padding, not a token: the model masks it.

part_length must match the model's data loader
(twin_model/src/utils/loader.py), which assumes blocks of 128 tokens.
What gets skipped or dropped is described in tokenise_group_variable.
"""

import csv

from sage.all import load


def cycle_to_transpositions(cycle):
    """The pairs (a1,a2), (a2,a3), ..., (a_{k-1},a_k) of a cycle (a1, ..., ak)."""
    if len(cycle) < 2:
        return []
    return [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]


def element_tokens(element, largest_degree):
    """Word tokens of a Sage permutation (no order token, no padding).

    Pairs come from cycle_to_transpositions over Sage's cycle_tuples(); each
    ordered pair (i, j) becomes (i-1) + n*(j-1) with n = largest_degree.
    """
    n = largest_degree
    tokens = []
    for cycle in element.cycle_tuples():
        for (i, j) in cycle_to_transpositions(list(cycle)):
            tokens.append((i - 1) + n * (j - 1))
    return tokens


def order_token(order, largest_degree):
    """Token for an element's order: placed just past the transposition tokens.

    Transposition tokens lie in 0 .. largest_degree**2 - 1, so order c becomes
    largest_degree**2 + c (250000 + c for degree 500) and never collides with them.
    """
    return largest_degree ** 2 + order


COUNTS = ("rows", "sigs_over_max_r", "sigs_over_degree",
          "vectors_over_class_cap", "vectors_word_too_long")


def tokenise_group_variable(filepath, output_filepath, *, max_r, largest_degree,
                            part_length, max_vectors_per_class):
    """Tokenise one group's orbit .sobj into a CSV; return a dict of COUNTS.

    Skipping is per signature, never the whole group: a signature is skipped if
    it has more than max_r elements, or if its permutation degree exceeds
    largest_degree (checked on the first element; all elements of a signature
    share one parent group). Of each class only the first max_vectors_per_class
    stored vectors are used, in the order the orbit stage stored them (not a
    random sample). A vector is dropped if any element's word is longer than
    part_length - 1 tokens, the room left in a block after the order token.

    rows                    generating vectors written
    sigs_over_max_r         signatures skipped for having more than max_r elements
    sigs_over_degree        signatures skipped for permutation degree > largest_degree
    vectors_over_class_cap  stored vectors beyond the first max_vectors_per_class of a class
    vectors_word_too_long   vectors dropped because an element's word exceeds part_length - 1
    """
    data = load(filepath)
    counts = dict.fromkeys(COUNTS, 0)
    with open(output_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "sig", "r", "text"])

        for cls_collection in data:
            signature = cls_collection[0]
            r = len(signature)
            if r > max_r:
                counts["sigs_over_max_r"] += 1
                continue
            if cls_collection[1] and cls_collection[1][0] and cls_collection[1][0][0]:
                degree = cls_collection[1][0][0][0].parent().degree()
                if degree > largest_degree:
                    counts["sigs_over_degree"] += 1
                    continue

            sig_text = "_".join(str(int(c)) for c in signature)
            for cls_index, cls in enumerate(cls_collection[1], start=1):
                cls_id = f"{sig_text}#{cls_index}"
                vectors = cls[:max_vectors_per_class]
                counts["vectors_over_class_cap"] += len(cls) - len(vectors)
                for gen_vect in vectors:
                    parts = []
                    too_long = False
                    for k in range(r):
                        word = element_tokens(gen_vect[k], largest_degree)
                        if len(word) > part_length - 1:
                            too_long = True
                            break
                        parts.extend([order_token(int(signature[k]), largest_degree)] + word
                                     + [-1] * (part_length - 1 - len(word)))
                    if too_long:
                        counts["vectors_word_too_long"] += 1
                        continue
                    writer.writerow([cls_id, sig_text, r,
                                     ",".join(str(t) for t in parts)])
                    counts["rows"] += 1
    return counts
