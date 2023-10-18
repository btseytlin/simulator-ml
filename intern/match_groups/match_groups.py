from typing import List, Tuple


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    pairs_dict = {}
    for a, b in pairs:
        if not a in pairs_dict:
            pairs_dict[a] = set()
        if not b in pairs_dict:
            pairs_dict[b] = set()

        pairs_dict[a].add(b)
        pairs_dict[b].add(a)

    for a, matches in pairs_dict.items():
        for b in matches:
            pairs_dict[a] = pairs_dict[a].union(pairs_dict[b])

    complete_matches = set()
    for a, matches in pairs_dict.items():
        for b in matches:
            if a == b:
                continue
            sorted_match = tuple(sorted((a, b)))
            complete_matches.add(sorted_match)
    complete_matches = list(complete_matches)
    complete_matches = sorted(complete_matches)
    return complete_matches


assert extend_matches([(1, 2), (7, 2)]) == [(1, 2), (1, 7), (2, 7)]
