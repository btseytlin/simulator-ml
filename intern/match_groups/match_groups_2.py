from typing import List


def extend_matches(groups: List[tuple]) -> List[tuple]:
    groups_dict = {}
    for group in groups:
        for item in group:
            if not item in groups_dict:
                groups_dict[item] = set()
            groups_dict[item] = groups_dict[item].union(set(group))

    for item, group in groups_dict.items():
        for other_item in group:
            groups_dict[item] = groups_dict[item].union(
                groups_dict[other_item]
            )

    for item, group in groups_dict.items():
        for other_item in group:
            groups_dict[item] = groups_dict[item].union(
                groups_dict[other_item]
            )

    match_groups = set()
    for item, group in groups_dict.items():
        match_groups.add(tuple(sorted(group)))
    match_groups = sorted(list(match_groups))
    return match_groups


assert extend_matches([(5, 3, 4, 8), (1, 2), (7, 2)]) == [
    (1, 2, 7),
    (3, 4, 5, 8),
]

assert extend_matches([(1, 2), (2, 3), (3, 4)]) == [(1, 2, 3, 4)]
