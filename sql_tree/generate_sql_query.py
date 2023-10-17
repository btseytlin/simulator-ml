import json

from sklearn.tree import DecisionTreeClassifier


def is_leaf(tree: DecisionTreeClassifier, node_index: int) -> bool:
    return (
        tree.tree_.children_left[node_index] == -1
        and tree.tree_.children_right[node_index] == -1
    )


def recurse(tree: DecisionTreeClassifier, node_index: int) -> dict:
    leaf = is_leaf(tree, node_index=node_index)

    if leaf:
        class_label = int(tree.tree_.value[node_index].argmax())
        return {"class": class_label}

    feature_index = int(tree.tree_.feature[node_index])
    threshold = round(float(tree.tree_.threshold[node_index]), 4)

    left_child_index = tree.tree_.children_left[node_index]
    right_child_index = tree.tree_.children_right[node_index]

    left = recurse(tree, left_child_index)
    right = recurse(tree, right_child_index)

    return {
        "feature_index": feature_index,
        "threshold": threshold,
        "left": left,
        "right": right,
    }


def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    tree_as_dict = recurse(tree, node_index=0)

    tree_as_json = json.dumps(tree_as_dict, indent=2)

    return tree_as_json


def recurse_sql(tree_dict: dict, features: list, depth=0) -> str:
    if "class" in tree_dict:
        class_label = tree_dict["class"]
        return str(class_label)

    feature = features[tree_dict["feature_index"]]
    threshold = tree_dict["threshold"]
    right_query = recurse_sql(tree_dict["left"], features, depth=depth + 1)
    left_query = recurse_sql(tree_dict["right"], features, depth=depth + 1)
    tabs = "\t" * depth
    query = (
        f"\n{tabs}CASE WHEN {feature} > {threshold} THEN"
        f" {left_query}\n{tabs}ELSE {right_query}\n{tabs}END"
    )
    return query


def generate_sql_query(tree_as_json: str, features: list) -> str:
    tree_as_dict = json.loads(tree_as_json)
    sql_query_case = recurse_sql(tree_as_dict, features)
    sql_query = f"SELECT {sql_query_case} as class_label"
    return sql_query
