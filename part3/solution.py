
import search  # Importing the search module which contains algorithms for solving search problems
import time  # Importing time module for time-related functions
import os  # Importing os module for interacting with the operating system

from cProfile import Profile  # Importing the Profile class from cProfile module for profiling
from pstats import Stats  # Importing the Stats class from pstats module for statistics

# Constants for vessel state indices
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

    __slots__ = ["time", "vessels", "cost", "_hash", "berth"]  # Define fixed slots for memory efficiency

    def __init__(self, time: int, vessels: tuple, berth, cost=0):
        """
        Initializes the state with the given time, vessel states, and cost.

        Parameters:
        - time (int): The current time of the state.
        - vessels (tuple): A tuple of tuples representing the vessels' states.
        - cost (int): The accumulated cost (default is 0).
        """
        self.time = time  # Set the current time
        self.vessels = vessels  # Set the vessels' states
        self.berth = berth
        self.cost = cost  # Set the accumulated cost
        self._hash = hash((self.time, self.vessels))  # Create a unique hash for the state

    @property
    def vessels_position(self):
        """
        Returns a list of [mooring_time, berth_section] for each vessel.
        This is useful for obtaining a simplified representation of the vessel positions.
        """
        return [vessel[:2] for vessel in self.vessels]  # Extract mooring time and berth section for each vessel

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
        #return self.vessels == other.vessels and self.time == other.time  # Compare the hash values for equality
        return self._hash == other._hash

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

    __slots__ = ["initial", "vessels", "S", "N"]  # Define fixed slots for memory efficiency

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

        self.S, self.N = [int(x) for x in data_lines[0].split()]  # Parse total berth size and number of vessels
        self.vessels = []  # Initialize vessel list
        for i in range(1, self.N + 1):
            # arrival time, processing time, section size, weight
            ai, pi, si, wi = [int(x) for x in data_lines[i].split()]  # Parse vessel details
            self.vessels.append({"a": ai, "p": pi, "s": si, "w": wi})  # Store vessel information in a dictionary

        # Initialize vessels state: (mooring_time, berth_section, vessel_index)
        vessels_state_list = [(-1, -1, idx) for idx in range(self.N)]  # Set initial state for each vessel
        berth = [0] * self.N
        self.initial = State(0, tuple(vessels_state_list),  0)  # Set the initial state

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
            vessel = list(new_vessels[vessel_idx])  # Convert to list to modify the specific vessel
            vessel[MOORING_TIME] = new_time  # Update mooring time
            vessel[BERTH_SECTION] = berth_space  # Update berth section

            ## ADD the vessel to berth
            new_breth = list(state.berth)
            for i in range(self.vessels[vessel[VESSEL_INDEX]]["s"]):
                new_breth[i] = self.vessels[vessel[VESSEL_INDEX]][p]

            new_vessels[vessel_idx] = tuple(vessel)  # Convert back to tuple after modification
            return State(
                new_time,
                tuple(new_vessels),
                state.cost + self.action_cost(action) # Compute new cost
            )
        else:
            # Action is to wait until new_time
            ## !! go back in the berth times the time we advanced
            return State(
                new_time,
                state.vessels,
                state.cost,  # Cost remains the same since no boat was scheduled, only heuristic will change
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

        ## ONLY THE INDEXES OF THE VESSEL ARE RETURN
        boats_not_scheduled = [
            vessel for vessel in state.vessels if vessel[MOORING_TIME] == -1  # Identify vessels that are not scheduled
        ]
        boats_scheduled = [
            vessel for vessel in state.vessels if vessel[MOORING_TIME] != -1  # Identify vessels that are scheduled
        ]

        # Build berth occupancy
        # Berth is represented as a list of availability at each berth section
        berth = [1] * self.S  # Initialize berth availability, 1 means available, 0 means occupied

        # Build a list of currently scheduled vessels and their processing end times
        scheduled_vessels_end_times = []  # Store end times of scheduled vessels

        # Fill the berth array with the vessels that are scheduled
        for vessel in boats_scheduled:
            vessel_idx = vessel[VESSEL_INDEX]  # Get the vessel index
            vessel_info = self.vessels[vessel_idx]  # Get the vessel information
            mooring_time = vessel[MOORING_TIME]  # Get mooring time
            berth_section = vessel[BERTH_SECTION]  # Get berth section
            processing_end_time = mooring_time + vessel_info["p"]  # Calculate processing end time

            if state.time >= mooring_time and state.time < processing_end_time:
                # Vessel is currently in berth
                s = berth_section
                e = berth_section + vessel_info["s"]  # Calculate the end section for the vessel

                for i in range(s, e):
                    berth[i] = 0  # Mark the berth section as occupied

            scheduled_vessels_end_times.append(processing_end_time)  # Store the processing end time

        # Create cumulative sum array for berth availability
        berth_cumsum = [0] * (self.S + 1)  # Initialize cumulative sum array
        for i in range(self.S):
            berth_cumsum[i + 1] = berth_cumsum[i] + berth[i]  # Calculate cumulative sum

        # Get vessels that have arrived and are not yet scheduled
        boats_arrived = [
            vessel
            for vessel in boats_not_scheduled
            if self.vessels[vessel[VESSEL_INDEX]]["a"] <= state.time  # Filter vessels that have arrived
        ]

        # If there are arrived boats, try to schedule them
        for vessel in boats_arrived:
            vessel_idx = vessel[VESSEL_INDEX]  # Get vessel index
            vessel_info = self.vessels[vessel_idx]  # Get vessel information
            vessel_size = vessel_info["s"]  # Get vessel size

            # Find positions where berth[i:i+vessel_size] are all 1s (available)
            for i in range(self.S - vessel_size + 1):
                if berth_cumsum[i + vessel_size] - berth_cumsum[i] == vessel_size:
                    actions.append((state.time, i, vessel_idx))  # Add scheduling action

        # Determine if there are boats that haven't arrived yet
        arrival_times = [
            self.vessels[vessel[VESSEL_INDEX]]["a"]
            for vessel in boats_not_scheduled
            if self.vessels[vessel[VESSEL_INDEX]]["a"] > state.time  # Get arrival times of unscheduled vessels
        ]

        if arrival_times:
            # There are vessels that haven't arrived yet
            # Compute the next event times
            completion_times = [
                end_time
                for end_time in scheduled_vessels_end_times
                if end_time > state.time  # Get end times of scheduled vessels
            ]

            next_times = arrival_times + completion_times  # Combine arrival and completion times

            if next_times:
                next_time = min(next_times)  # Get the earliest time
                # Add an action to wait until next_time
                actions.append((next_time, -1, -1))  # Waiting action
        else:
            # All vessels have arrived
            # Only consider waiting if there are no scheduling actions
            if not actions:
                # No possible scheduling actions, need to wait for berth space to become available
                completion_times = [
                    end_time
                    for end_time in scheduled_vessels_end_times
                    if end_time > state.time  # Get end times of scheduled vessels
                ]

                if completion_times:
                    next_time = min(completion_times)  # Get the earliest completion time
                    # Add an action to wait until next_time
                    actions.append((next_time, -1, -1))  # Waiting action

        return actions  # Return the list of possible actions

    def h (self, node):
        '''
        Returns the heuristic valus of the state associated with the node

        Currently the heuristic works as follows:
            For a given state in the search problem, the cost we have to pay is the sum of the flow time of all the scheduled vessels, with this we can see that by never scheduling a vessel we will fall into infinite solution space, since the cost of adding a boat will always be bigger than the cost of not adding it.

            To mitigate this, the heuristic applied was to simply assume we can add all the boats in the waiting list in the current time step, and then the heuristic value is the sum of the flow time of all the vessels in the waiting list, when they are added to the berth in the current step.

            This heuristic is admissible since it never overestimates the cost of reaching the goal, since the minima flow time we can have for allocating a boat is to allocate it at the current time step, and since this is what the heuristic computes, we make sure and overestimation never happens

            !! TALK about consistency for this heuristic although i cannot understand if this is consistent of not
        '''
        state = node.state


        heuristic =  sum([
            self.vessels[vessel[VESSEL_INDEX]]["w"]*(state.time + self.vessels[vessel[VESSEL_INDEX]]["p"] - self.vessels[vessel[VESSEL_INDEX]]["a"]) for vessel in state.vessels if vessel[MOORING_TIME] == -1 and self.vessels[VESSEL_INDEX]["a"] <= state.time
        ])

        #print(f"State.time : {state.time} - state.vessels : {state.vessels} - state.cost : {state.cost} - heuristic : {heuristic} + total_cost : {state.cost + heuristic}")

        return heuristic

    def action_cost(self, action):
        """
        Compute the cost of an action.

        The real cost of a action is the flow time for the allocated boat

        Parameters:
        - action (tuple): The action to be performed.

        Returns:
        - int: The cost of the action.
        """
        action_time, berth_space, vessel_idx = action
        vessel = self.vessels[vessel_idx]
        return vessel["w"] * (action_time +  vessel["p"] - vessel["a"])

    def path_cost(self, c, state1, action, state2):
        """
        Returns the cost of a solution path that arrives at state2 from state1 via action.

        Parameters:
        - c (int): Current cost.
        - state1 (State): The initial state before the action.
        - action (tuple): The action performed.
        - state2 (State): The resulting state after the action.

        Returns:
        - int: The total cost after performing the action.
        """
        action_time, berth_space, vessel_idx = action
        if berth_space == -1:
            return c
        #print(f"Path Cost : {c} + {self.action_cost(action)}")
        return c + self.action_cost(action)  # Return updated cost

    def goal_test(self, state):
        """
        Returns True if the state is a goal.

        Parameters:
        - state (State): The current state.

        Returns:
        - bool: True if the goal is achieved (all vessels moored), False otherwise.
        """
        return all(vessel[MOORING_TIME] != -1 for vessel in state.vessels)  # Check if all vessels are moored

    def solve(self):
        """
        Calls the uniform cost search algorithm.

        Returns:
        - list: A solution in the specified format.
        """
        time_star = time.time()
        solution = search.astar_search(self, self.h, False)
        time_end = time.time()
        if solution is not None:
            #print(f"Solution : {solution.state.vessels_position} cost : {solution.path_cost} time : {time_end - time_star}")
            print(solution.depth)
            return solution.state.vessels_position  # Return the position of vessels in the solution



if __name__ == "__main__":

    def get_test_files(test_dir="Tests"):
        """
        Retrieves test files from the specified directory.

        Parameters:
        - test_dir (str): The directory to search for test files.

        Returns:
        - list: A list of test file paths.
        """
        if test_dir in os.listdir():  # Check if the test directory exists
            return [
                os.path.join(test_dir, file)
                for file in os.listdir(test_dir)
                if file.endswith(".dat")  # Filter for .dat files
            ]
        return []  # Return an empty list if no test files found

    test_files = get_test_files()  # Retrieve test files

    # Optionally filter test files
    test_files = [test for test in test_files if "108" in test]  # Filter test files based on naming

    for test in test_files:  # Iterate over the test files
        with open(test) as fh:  # Open the test file
            problem = BAProblem()  # Create an instance of BAProblem
            problem.load(fh)  # Load the problem from the file
            print(f"Test {test}")  # Print the name of the test file
            with Profile() as p:
                problem.solve()
                Stats(p).print_stats()
