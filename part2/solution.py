import search
import time
import os

MOORING_TIME = 0
BERTH_SECTION = 1
VESSEL_INDEX = 2


class State:
    """
    A class to represent a state in the BAProblem class.

    Attributes:
        time (int): The current time of the state.
        vessels (tuple): A tuple of tuples representing vessels' states.
        cost (int): The accumulated cost up to this state.
    """

    __slots__ = ["time", "vessels", "cost", "_hash"]

    def __init__(self, time: int, vessels: tuple, cost=0):
        self.time = time
        self.vessels = vessels  # Vessels is a tuple of tuples
        self.cost = cost
        self._hash = hash((self.time, self.vessels))

    @property
    def vessels_position(self):
        # Returns a list of [mooring_time, berth_section] for each vessel
        return [vessel[:2] for vessel in self.vessels]

    def __str__(self) -> str:
        return f"State({self.time}, {self.vessels})"

    def __hash__(self) -> int:
        # Efficient hash using immutable vessels representation
        return self._hash

    def __eq__(self, other) -> bool:
        # Efficient equality check
        return self.time == other.time and self.vessels == other.vessels

    def __lt__(self, other):
        # For priority queue comparison
        return self.cost < other.cost


class BAProblem(search.Problem):
    __slots__ = ["initial", "vessels", "S", "N"]

    def __init__(self):
        self.initial = None
        self.vessels = []
        self.S = 0  # Total berth size
        self.N = 0  # Number of vessels

    def cost(self, sol):
        """
        Compute the total weighted flow time cost of a given solution.

        Parameters:
        - sol (list): A list of tuples representing the solution, where each tuple contains
                      the starting mooring time (ui) and the starting berth section (vi)
                      for each vessel.

        Returns:
        - int: The total weighted flow time cost.
        """
        total_cost = 0
        for i, (ui, vi) in enumerate(sol):
            vessel = self.vessels[i]
            ci = ui + vessel["p"]
            fi = ci - vessel["a"]
            total_cost += vessel["w"] * fi

        return total_cost

    def load(self, fh):
        """
        Load a BAP problem from the file object.

        Parameters:
        - fh: A file object representing the input file.
        """
        lines = fh.readlines()
        data_lines = [
            line.strip() for line in lines if line.strip() and not line.startswith("#")
        ]

        self.S, self.N = [int(x) for x in data_lines[0].split()]
        self.vessels = []
        for i in range(1, self.N + 1):
            # arrival time, processing time, section size, weight
            ai, pi, si, wi = [int(x) for x in data_lines[i].split()]
            self.vessels.append({"a": ai, "p": pi, "s": si, "w": wi})

        # Initialize vessels state: (mooring_time, berth_section, vessel_index)
        vessels_state_list = [(-1, -1, idx) for idx in range(self.N)]
        self.initial = State(0, tuple(vessels_state_list), 0)

    def result(self, state: State, action: tuple):
        """
        Returns the state that results from executing
        the given action in the given state.
        """
        new_time, berth_space, vessel_idx = action

        if vessel_idx != -1:
            # Action is to moor a vessel
            new_vessels = list(state.vessels)  # Convert to list for modification
            vessel = list(new_vessels[vessel_idx])  # Convert to list to modify
            vessel[MOORING_TIME] = new_time
            vessel[BERTH_SECTION] = berth_space
            new_vessels[vessel_idx] = tuple(vessel)  # Convert back to tuple
            return State(
                new_time,
                tuple(new_vessels),
                state.cost + self.compute_cost(state, action),
            )
        else:
            # Action is to wait until new_time
            return State(
                new_time,
                state.vessels,
                state.cost + self.compute_cost(state, action),
            )

    def actions(self, state: State) -> list:
        """
        Returns the list of actions that can be executed in
        the given state.
        """
        actions = []

        boats_not_scheduled = [
            vessel for vessel in state.vessels if vessel[MOORING_TIME] == -1
        ]
        boats_scheduled = [
            vessel for vessel in state.vessels if vessel[MOORING_TIME] != -1
        ]

        # Build berth occupancy
        # Berth is represented as a list of availability at each berth section
        berth = [1] * self.S  # 1 means available, 0 means occupied

        # Build a list of currently scheduled vessels and their processing end times
        scheduled_vessels_end_times = []

        # Fill the berth array with the vessels that are scheduled
        for vessel in boats_scheduled:
            vessel_idx = vessel[VESSEL_INDEX]
            vessel_info = self.vessels[vessel_idx]
            mooring_time = vessel[MOORING_TIME]
            berth_section = vessel[BERTH_SECTION]
            processing_end_time = mooring_time + vessel_info["p"]

            if state.time >= mooring_time and state.time < processing_end_time:
                # Vessel is currently in berth
                s = berth_section
                e = berth_section + vessel_info["s"]
                for i in range(s, e):
                    berth[i] = 0

            scheduled_vessels_end_times.append(processing_end_time)

        # Create cumulative sum array for berth availability
        berth_cumsum = [0] * (self.S + 1)
        for i in range(self.S):
            berth_cumsum[i + 1] = berth_cumsum[i] + berth[i]

        # Get vessels that have arrived and are not yet scheduled
        boats_arrived = [
            vessel
            for vessel in boats_not_scheduled
            if self.vessels[vessel[VESSEL_INDEX]]["a"] <= state.time
        ]

        # If there are arrived boats, try to schedule them
        for vessel in boats_arrived:
            vessel_idx = vessel[VESSEL_INDEX]
            vessel_info = self.vessels[vessel_idx]
            vessel_size = vessel_info["s"]

            # Find positions where berth[i:i+vessel_size] are all 1s (available)
            for i in range(self.S - vessel_size + 1):
                if berth_cumsum[i + vessel_size] - berth_cumsum[i] == vessel_size:
                    actions.append((state.time, i, vessel_idx))

        # Determine if there are boats that haven't arrived yet
        arrival_times = [
            self.vessels[vessel[VESSEL_INDEX]]["a"]
            for vessel in boats_not_scheduled
            if self.vessels[vessel[VESSEL_INDEX]]["a"] > state.time
        ]

        if arrival_times:
            # There are vessels that haven't arrived yet
            # Compute the next event times
            completion_times = [
                end_time
                for end_time in scheduled_vessels_end_times
                if end_time > state.time
            ]

            next_times = arrival_times + completion_times

            if next_times:
                next_time = min(next_times)
                # Add an action to wait until next_time
                actions.append((next_time, -1, -1))
        else:
            # All vessels have arrived
            # Only consider waiting if there are no scheduling actions
            if not actions:
                # No possible scheduling actions, need to wait for berth space to become available
                completion_times = [
                    end_time
                    for end_time in scheduled_vessels_end_times
                    if end_time > state.time
                ]

                if completion_times:
                    next_time = min(completion_times)
                    # Add an action to wait until next_time
                    actions.append((next_time, -1, -1))

        return actions

    def goal_test(self, state):
        """
        Returns True if the state is a goal.
        """
        return all(vessel[MOORING_TIME] != -1 for vessel in state.vessels)

    def path_cost(self, c, state1, action, state2):
        """
        Returns the cost of a solution path that arrives at state2 from state1 via action.
        """
        return c + self.compute_cost(state1, action)

    def solve(self):
        """
        Calls the uniform cost search algorithm.
        Returns a solution using the specified format.
        """
        solution = search.uniform_cost_search(self)
        if solution is not None:
            return solution.state.vessels_position

    def compute_cost(self, state, action):
        """
        Compute the cost of an action.
        The cost is the sum of the waiting costs for all unscheduled vessels over the time interval from state.time to new_time.
        """
        new_time, berth_space, vessel_idx = action
        total_cost = 0

        # Time interval over which to calculate waiting costs
        time_interval = new_time - state.time

        per_unit_waiting_cost = sum(
            self.vessels[vessel[VESSEL_INDEX]]["w"]
            for vessel in state.vessels
            if vessel[MOORING_TIME] == -1
            and self.vessels[vessel[VESSEL_INDEX]]["a"] <= state.time
        )
        total_cost += per_unit_waiting_cost * time_interval

        return total_cost


if __name__ == "__main__":

    def get_test_files(test_dir="Tests"):
        if test_dir in os.listdir():
            return [
                os.path.join(test_dir, file)
                for file in os.listdir(test_dir)
                if file.endswith(".dat")
            ]
        return []

    test_files = get_test_files()

    # Optionally filter test files
    # test_files = [test for test in test_files if "108" in test]

    for test in test_files:
        with open(test) as fh:
            problem = BAProblem()
            problem.load(fh)
            print(f"Test {test}")

            time_start = time.time()
            solution = problem.solve()
            print(f"Solution is {solution}")
            print(f"Time elapsed: {time.time() - time_start}")
            print("\n")
