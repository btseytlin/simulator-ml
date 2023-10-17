import json

from sklearn.tree import DecisionTreeClassifier


def is_leaf(tree, node_index):
    return (
        tree.tree_.children_left[node_index] == -1
        and tree.tree_.children_right[node_index] == -1
    )


def recurse(tree, node_index, parent_index=None, side=None):
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
