# Compute the cache.json record of SmallGroup(o, i) in GAP, for groups the
# LMFDB does not hold (orders 512 and 1536 in this dataset) and to check the
# cached LMFDB values independently.  Every field follows the LMFDB
# definition of the gps_groups column of the same name (fields.py).
#
#   gap -q -b -c 'labels_file := "groups.txt"; out_file := "gap_records.json";' gap_records.g
#
# labels_file: one group per line, 'order_index' or 'order.index' (default groups.txt,
#              which is cheap to compute in full apart from AutomorphismGroup of the largest groups);
# out_file:    JSON {label: record} in exactly the cache.json format (default gap_records.json).
# Then: python merge_gap_records.py gap_records.json --check   (compare with the cache)
#       python merge_gap_records.py gap_records.json           (fill missing records)
# Needs the SmallGroups library (orders 512 and 1536 need its 'extra' part).

if not IsBound(labels_file) then labels_file := "groups.txt"; fi;
if not IsBound(out_file) then out_file := "gap_records.json"; fi;

# metacyclic: a cyclic normal subgroup with cyclic quotient
IsMetacyclicGroup := function(G)
    local N;
    if IsCyclic(G) then return true; fi;
    for N in NormalSubgroups(G) do
        if IsCyclic(N) and IsCyclic(G / N) then return true; fi;
    od;
    return false;
end;

# rational: every irreducible character takes rational values
IsRationalGroup := function(G)
    return ForAll(Irr(CharacterTable(G)), chi -> ForAll(chi, x -> IsRat(x)));
end;

# A-group: every Sylow subgroup abelian; Z-group: every Sylow subgroup cyclic
SylowsAre := function(G, test)
    return ForAll(PrimeDivisors(Size(G)), p -> test(SylowSubgroup(G, p)));
end;

# LMFDB's 'elementary' and 'hyperelementary' columns are the products of the
# primes p for which G is p-elementary / p-hyperelementary:
#   p-hyperelementary: G = C : P with C a normal cyclic Hall p'-subgroup, P a Sylow p-subgroup
#   p-elementary:      moreover G = C x P
ElementaryNumbers := function(G)
    local el, hyp, p, P, C, cands;
    el := 1; hyp := 1;
    for p in PrimeDivisors(Size(G)) do
        P := SylowSubgroup(G, p);
        cands := Filtered(NormalSubgroups(G),
                          N -> IsCyclic(N) and Size(N) * Size(P) = Size(G) and Gcd(Size(N), p) = 1);
        if Length(cands) > 0 then
            hyp := hyp * p;
            C := cands[1];
            if IsNormal(G, P) and IsTrivial(Intersection(C, P))
               and ForAll(GeneratorsOfGroup(C), c -> ForAll(GeneratorsOfGroup(P), x -> c * x = x * c)) then
                el := el * p;
            fi;
        fi;
    od;
    return [el, hyp];
end;

# the 20 fields of fields.py (PROPS then NUMERIC), in cache.json order
GroupRecord := function(o, i)
    local G, e;
    G := SmallGroup(o, i);
    e := ElementaryNumbers(G);
    return [ ["abelian", IsAbelian(G)], ["cyclic", IsCyclic(G)],
             ["nilpotent", IsNilpotentGroup(G)], ["solvable", IsSolvableGroup(G)],
             ["supersolvable", IsSupersolvableGroup(G)],
             ["metabelian", IsAbelian(DerivedSubgroup(G))],
             ["metacyclic", IsMetacyclicGroup(G)], ["monomial", IsMonomialGroup(G)],
             ["rational", IsRationalGroup(G)], ["Agroup", SylowsAre(G, IsAbelian)],
             ["Zgroup", SylowsAre(G, IsCyclic)], ["simple", IsSimpleGroup(G)],
             ["perfect", IsPerfectGroup(G)],
             ["order", o], ["exponent", Exponent(G)],
             ["elementary", e[1]], ["hyperelementary", e[2]],
             ["aut_order", Size(AutomorphismGroup(G))],
             ["number_conjugacy_classes", NrConjugacyClasses(G)],
             ["derived_length", DerivedLength(G)] ];
end;

labels := Filtered(SplitString(StringFile(labels_file), "\n"), l -> l <> "");
SetPrintFormattingStatus("*stdout*", false);
out := OutputTextFile(out_file, false);
SetPrintFormattingStatus(out, false);
AppendTo(out, "{");
n := 0;
for lab in labels do
    parts := SplitString(lab, "_.");
    rec_ := GroupRecord(Int(parts[1]), Int(parts[2]));
    if n > 0 then AppendTo(out, ","); fi;
    AppendTo(out, "\n\"", parts[1], ".", parts[2], "\": {");
    AppendTo(out, JoinStringsWithSeparator(List(rec_, kv -> Concatenation("\"", kv[1], "\": ", String(kv[2]))), ", "));
    AppendTo(out, "}");
    n := n + 1;
    if n mod 100 = 0 then Print(n, " of ", Length(labels), " done\n"); fi;
od;
AppendTo(out, "\n}\n");
CloseStream(out);
Print("wrote ", n, " records to ", out_file, "\n");
QUIT;
