import search

MOORING_TIME = 0  # Index for the mooring time of the vessel
BERTH_SECTION = 1  # Index for the berth section of the vessel
VESSEL_INDEX = 2  # Index for the vessel identifier


class State:
    """
    A class to represent a state in the Berth Allocation Problem.

    Attributes:
        time (int): The current time of the state.
        vessels (tuple): A tuple of tuples representing vessels' states.
        cost (int): The accumulated cost up to this state.
    """

    __slots__ = [
        "time",
        "vessels",
        "cost",
        "_hash",
    ]  # Define fixed slots for memory efficiency

    def __init__(self, time: int, vessels: tuple, cost=0):
        """
        Initializes the state with the given time, vessel states, and cost.

        Parameters:
        - time (int): The current time of the state.
        - vessels (tuple): A tuple of tuples representing the vessels' states.
        - cost (int): The accumulated cost (default is 0).
        """
        self.time = time  # Set the current time
        self.vessels = vessels  # Set the vessels' states
        self.cost = cost  # Set the accumulated cost
        self._hash = hash(
            (self.time, self.vessels)
        )  # Create a unique hash for the state

    @property
    def vessels_position(self):
        """
        Returns a list of [mooring_time, berth_section] for each vessel.
        This is useful for obtaining a simplified representation of the vessel positions.
        """
        return [
            vessel[:2] for vessel in self.vessels
        ]  # Extract mooring time and berth section for each vessel

    def __str__(self) -> str:
        """
        Returns a string representation of the state.
        """
        return f"State({self.time}, {self.vessels})"  # Format the state as a string

    def __hash__(self) -> int:
        """
        Returns the hash of the state for efficient storage in data structures like sets and dictionaries.
        """
        return self._hash  # Return the pre-computed hash

    def __eq__(self, other) -> bool:
        """
        Checks if two states are equal based on their hash values.
        """
        return self._hash == other._hash  # Compare the hash values for equality

    def __lt__(self, other):
        """
        For priority queue comparison, determines if this state has a lower cost than another state.
        """
        return self.cost < other.cost  # Compare based on cost


class BAProblem(search.Problem):
    """
    A class to represent the Berth Allocation Problem.

    Attributes:
        initial (State): The initial state of the problem.
        vessels (list): A list containing details about each vessel.
        S (int): Total berth size.
        N (int): Number of vessels.
    """

    __slots__ = [
        "initial",
        "vessels",
        "S",
        "N",
    ]  # Define fixed slots for memory efficiency

    def __init__(self):
        """
        Initializes the Berth Allocation Problem with no initial state and empty vessels.
        """
        self.initial = None  # Initial state
        self.vessels = []  # List of vessels
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
        total_cost = 0  # Initialize total cost to zero
        for i, (ui, vi) in enumerate(sol):
            vessel = self.vessels[i]  # Get the vessel details
            ci = ui + vessel["p"]  # Compute the completion time
            fi = ci - vessel["a"]  # Calculate flow time
            total_cost += vessel["w"] * fi  # Accumulate the weighted flow time cost

        return total_cost  # Return the total cost

    def load(self, fh):
        """
        Load a Berth Allocation Problem from a file object.

        Parameters:
        - fh: A file object representing the input file.
        """
        lines = fh.readlines()  # Read all lines from the file
        data_lines = [
            line.strip() for line in lines if line.strip() and not line.startswith("#")
        ]  # Filter out comments and empty lines

        self.S, self.N = [
            int(x) for x in data_lines[0].split()
        ]  # Parse total berth size and number of vessels
        self.vessels = []  # Initialize vessel list
        for i in range(1, self.N + 1):
            # arrival time, processing time, section size, weight
            ai, pi, si, wi = [
                int(x) for x in data_lines[i].split()
            ]  # Parse vessel details
            self.vessels.append(
                {"a": ai, "p": pi, "s": si, "w": wi}
            )  # Store vessel information in a dictionary

        # Initialize vessels state: (mooring_time, berth_section, vessel_index)
        vessels_state_list = [
            (-1, -1, idx) for idx in range(self.N)
        ]  # Set initial state for each vessel
        self.initial = State(0, tuple(vessels_state_list), 0)  # Set the initial state

    def result(self, state: State, action: tuple):
        """
        Returns the state that results from executing the given action in the given state.

        Parameters:
        - state (State): The current state.
        - action (tuple): The action to be performed, containing the new time, berth space, and vessel index.

        Returns:
        - State: The new state after executing the action.
        """
        new_time, berth_space, vessel_idx = action  # Unpack the action parameters

        if vessel_idx != -1:
            # Action is to moor a vessel
            new_vessels = list(state.vessels)  # Convert to list for modification
            vessel = list(
                new_vessels[vessel_idx]
            )  # Convert to list to modify the specific vessel
            vessel[MOORING_TIME] = new_time  # Update mooring time
            vessel[BERTH_SECTION] = berth_space  # Update berth section
            new_vessels[vessel_idx] = tuple(
                vessel
            )  # Convert back to tuple after modification
            return State(
                new_time,
                tuple(new_vessels),
                state.cost + self.compute_cost(state, action),  # Compute new cost
            )
        else:
            # Action is to wait until new_time
            return State(
                new_time,
                state.vessels,
                state.cost
                + self.compute_cost(state, action),  # Compute cost for waiting
            )

    def actions(self, state: State) -> list:
        """
        Returns the list of actions that can be executed in the given state.

        Parameters:
        - state (State): The current state.

        Returns:
        - list: A list of possible actions that can be performed.
        """
        actions = []  # Initialize actions list

        boats_not_scheduled = [
            vessel
            for vessel in state.vessels
            if vessel[MOORING_TIME] == -1  # Identify vessels that are not scheduled
        ]
        boats_scheduled = [
            vessel
            for vessel in state.vessels
            if vessel[MOORING_TIME] != -1  # Identify vessels that are scheduled
        ]

        # Build berth occupancy
        berth = [
            1
        ] * self.S  # Initialize berth availability, 1 means available, 0 means occupied

        # Build a list of currently scheduled vessels and their processing end times
        scheduled_vessels_end_times = []  # Store end times of scheduled vessels

        for vessel in boats_scheduled:
            vessel_idx = vessel[VESSEL_INDEX]
            vessel_info = self.vessels[vessel_idx]
            mooring_time = vessel[MOORING_TIME]
            berth_section = vessel[BERTH_SECTION]
            processing_end_time = mooring_time + vessel_info["p"]

            if state.time >= mooring_time and state.time < processing_end_time:
                s = berth_section
                e = berth_section + vessel_info["s"]
                for i in range(s, e):
                    berth[i] = 0  # Mark berth as occupied

            scheduled_vessels_end_times.append(processing_end_time)

        berth_cumsum = [0] * (self.S + 1)
        for i in range(self.S):
            berth_cumsum[i + 1] = berth_cumsum[i] + berth[i]

        boats_arrived = [
            vessel
            for vessel in boats_not_scheduled
            if self.vessels[vessel[VESSEL_INDEX]]["a"] <= state.time
        ]

        for vessel in boats_arrived:
            vessel_idx = vessel[VESSEL_INDEX]
            vessel_info = self.vessels[vessel_idx]
            vessel_size = vessel_info["s"]

            for i in range(self.S - vessel_size + 1):
                if berth_cumsum[i + vessel_size] - berth_cumsum[i] == vessel_size:
                    actions.append((state.time, i, vessel_idx))

        arrival_times = [
            self.vessels[vessel[VESSEL_INDEX]]["a"]
            for vessel in boats_not_scheduled
            if self.vessels[vessel[VESSEL_INDEX]]["a"] > state.time
        ]

        if arrival_times:
            completion_times = [
                end_time
                for end_time in scheduled_vessels_end_times
                if end_time > state.time
            ]
            next_times = arrival_times + completion_times
            if next_times:
                next_time = min(next_times)
                actions.append((next_time, -1, -1))
        else:
            if not actions:
                completion_times = [
                    end_time
                    for end_time in scheduled_vessels_end_times
                    if end_time > state.time
                ]
                if completion_times:
                    next_time = min(completion_times)
                    actions.append((next_time, -1, -1))

        return actions

    def goal_test(self, state):
        """
        Returns True if the state is a goal.

        Parameters:
        - state (State): The current state.

        Returns:
        - bool: True if the goal is achieved (all vessels moored), False otherwise.
        """
        return all(vessel[MOORING_TIME] != -1 for vessel in state.vessels)

    def path_cost(self, c, state1, action, state2):
        return c + self.compute_cost(state1, action)

    def compute_cost(self, state, action):
        new_time, berth_space, vessel_idx = action
        total_cost = 0
        time_interval = new_time - state.time

        per_unit_waiting_cost = sum(
            self.vessels[vessel[VESSEL_INDEX]]["w"]
            for vessel in state.vessels
            if vessel[MOORING_TIME] == -1
            and self.vessels[vessel[VESSEL_INDEX]]["a"] <= state.time
        )
        total_cost += per_unit_waiting_cost * time_interval

        return total_cost

    def heuristic(self, node):
        """
        Heuristic function for A* search, applied to a node.
    
        Parameters:
        - node (Node): The current node, which contains a state.
    
        Returns:
        - int: The heuristic estimate of the remaining cost.
        """
        state = node.state  # Get the state from the node
        current_cost = state.cost  # Get the current cost of the state
        total_estimated_cost = current_cost  # Start with the current cost
        latest_time = state.time  # Track the latest time for vessels already moored
    
        # Calculate the cost for vessels that are not yet moored
        for vessel in state.vessels:
            if vessel[MOORING_TIME] == -1:  # Vessel not scheduled
                vessel_info = self.vessels[vessel[VESSEL_INDEX]]
                # Add the cost of mooring this vessel at the latest time
                total_estimated_cost += (
                    latest_time + vessel_info["p"] - vessel_info["a"]
                ) * vessel_info["w"]  # Weighted remaining processing time
                # Update the latest_time for the next vessel
                latest_time += vessel_info["s"]  # Update for the space needed
    
        return total_estimated_cost


    def solve(self):
        """
        Solve the Berth Allocation Problem using A* search with a heuristic.

        Returns:
        - list: The solution, if one is found.
        """
        solution = search.astar_search(
            self, h=self.heuristic
        )  # Call astar_search with the heuristic
        if solution is not None:
            return (
                solution.state.vessels_position
            )  # Return the position of vessels in the solution
