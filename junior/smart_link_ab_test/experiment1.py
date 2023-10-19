from typing import List, Tuple


class Experiment:
    """Experiment class. Contains the logic for assigning users to groups."""

    def __init__(
        self,
        experiment_id: int,
        groups: Tuple[str] = ("A", "B"),
        group_weights: List[float] = None,
    ):
        self.experiment_id = experiment_id
        self.groups = groups
        self.group_weights = group_weights or [0.5, 0.5]

        # Define the salt for experiment_id.
        # The salt should be deterministic and unique for each experiment_id.
        self.salt = f"experiment_{experiment_id}"

        # Define the group weights if they are not provided equaly distributed
        # Check input group weights. They must be non-negative and sum to 1.

    def group(self, click_id: int) -> Tuple[int, str]:
        """Assigns a click to a group.

        Parameters
        ----------
        click_id: int :
            id of the click

        Returns
        -------
        Tuple[int, str] :
            group id and group name
        """

        # Assign the click to a group randomly based on the group weights
        # Return the group id and group name
        click_hash = hash(str(click_id) + self.salt)

        mod = click_hash % 100

        group_weights_int = [int(i * 100) for i in self.group_weights]
        group_id = 0
        running_sum = 0
        for i, weight in enumerate(group_weights_int):
            running_sum += weight
            if running_sum >= mod:
                group_id = i
                break
        return group_id, self.groups[group_id]
