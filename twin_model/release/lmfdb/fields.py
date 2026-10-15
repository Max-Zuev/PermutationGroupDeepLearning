# coding=utf-8
"""The LMFDB gps_groups fields kept in cache.json, one record per group
label 'order.index'. Both fetchers write exactly these; build_census_release.py
reads a subset of them."""

PROPS = ["abelian", "cyclic", "nilpotent", "solvable", "supersolvable", "metabelian",
         "metacyclic", "monomial", "rational", "Agroup", "Zgroup", "simple", "perfect"]
NUMERIC = ["order", "exponent", "elementary", "hyperelementary", "aut_order",
           "number_conjugacy_classes", "derived_length"]
