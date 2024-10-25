from ast import BitXor
import search  # Importing the search module which contains algorithms for solving search problems
import time  # Importing time module for time-related functions
import os  # Importing os module for interacting with the operating system
from itertools import groupby
import numpy as np


#from cProfile import (
#    Profile,
#)  # Importing the Profile class from cProfile module for profiling
#from pstats import Stats  # Importing the Stats class from pstats module for statistics

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

    __slots__ = [
        "time",
        "vessels",
        "cost",
        "_hash",
        "berth",
        "boats_scheduled",
        "boats_arrived",
        "boats_not_arrived",
    ]  # Define fixed slots for memory efficiency

    def __init__(
        self,
        time: int,
        vessels: list,
        berth,
        boats_scheduled,
        boats_arrived,
        boats_not_arrived,
        cost=0,
    ):
        """
        Initializes the state with the given time, vessel states, and cost.

        Parameters:
        - time (int): The current time of the state.
        - vessels (tuple): A tuple of tuples representing the vessels' states.
        - cost (int): The accumulated cost (default is 0).
        """
        self.time = time  # Set the current time
        self.vessels =  tuple(vessels)  # Set the vessels' states
        self.berth = berth
        self.cost = cost  # Set the accumulated cost

        self.boats_scheduled = boats_scheduled
        self.boats_arrived = boats_arrived
        self.boats_not_arrived = boats_not_arrived

        self._hash = hash(
            (self.time, self.vessels, self.cost)
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
        return self.vessels == other.vessels and self.time == other.time  # Compare the hash values for equality
        #return self._hash == other._hash

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
        "ordered_vessels",
    ]  # Define fixed slots for memory efficiency

    def __init__(self):
        """
        Initializes the Berth Allocation Problem with no initial state and empty vessels.
        """
        self.initial = None  # Initial state
        self.vessels = []  # List of vessels
        self.vessels_order = []

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

        ## Create state variables defeenition
        arrived_vessels = [
            idx for idx, vessel in enumerate(self.vessels) if vessel["a"] <= 0
        ]  ## List of all the indexes of vessels that have already arrived to the port, meaning the current time is greater or equal to their arrival time. All the vessels in this list are not yet scheduled, and are currently awaiting to be scheduled
        not_arrived_vessels = [
            idx for idx, vessel in enumerate(self.vessels) if vessel["a"] > 0
        ]  ## List of all the indexes of vessels that have not yet arrived to the port, meaning the current time is less than the arrival time of the vessels
        scheduled_vessels = (
            []
        )  ## List of all the indexed of vessels that have already been scheduled in the current time step

        vessels_state_list = [
            (-1, -1, idx) for idx in range(self.N)
        ]  # Set initial state for each vessel
        berth = [
            0
        ] * self.S  ## This variable represent the berth ocuppation in a given state, if at the index i the value is 0, then the section is empty, if the value is a different positive value, then the section will be occupied for the next berth[i] time units

        ## Create the initial state
        self.initial = State(
            0,
            vessels_state_list,
            berth,
            scheduled_vessels,
            arrived_vessels,
            not_arrived_vessels,
            0,
        )

    # This method seem to be working correctly
    def result(self, state: State, action: tuple):
        """
        Returns the state that results from executing the given action in the given state.

        Parameters:
        - state (State): The current state.
        - action (tuple): The action to be performed, containing the new time, berth space, and vessel index. The actions can be of two types:
            1.  The time associated with the action is greater than the current time of the state, and both the vessel_idx and the berth size are -1, meaning that no vessel is being scheduled to moore in this action
            2. The time associated with the action is the same as the current time of the state, and both the vessel_idx and the berth_size must different than -1, meaning that a vessel is being scheduled to moore in the this action

        Returns:
        - State: The new state after executing the action.
        """
        action_time, berth_space, boat = action  # Unpack the action parameters

        ## Get copy of the previous state variables for later manipulation to the new state
        boats_arrived = list(state.boats_arrived)
        boats_scheduled = list(state.boats_scheduled)
        boats_not_arrived = list(state.boats_not_arrived)
        vessels_state_list = list(state.vessels)
        berth = list(state.berth)

        if action_time == state.time:
            ## A boat is being scheduled to moore in the current time step

            ## Update the state of the vessel that is being scheduled
            boats_scheduled.append(boat)
            boats_arrived.remove(boat)
            vessels_state_list[boat] = (action_time, berth_space, boat)

            for i in range(self.vessels[boat]["s"]):
                berth[berth_space + i] = self.vessels[boat]["p"]

            ## Create the new state
            return State (
                action_time,
                vessels_state_list,
                berth,
                boats_scheduled,
                boats_arrived,
                boats_not_arrived,
                state.cost + self.action_cost(action)
            )

        else:
            ## No boat is being scheduled to moore in the current time step

            ## Update the boats_arrived and boats_not_arrived list taking into consideration how much time passed between the two actions

            for boat_not_arrived in boats_not_arrived[:]:
                if self.vessels[boat_not_arrived]["a"] <= action_time:
                    boats_arrived.append(boat_not_arrived)
                    boats_not_arrived.remove(boat_not_arrived)


            ## Update the berth occupancy list taking into consideration how much time passed between the two actions
            time_diff = action_time - state.time
            berth = [max(0, b - time_diff) for b in berth]

            ## Create the new state
            return State(
                action_time,
                vessels_state_list,
                berth,
                boats_scheduled,
                boats_arrived,
                boats_not_arrived,
                state.cost
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

        berth_cumsum = [0] * (self.S + 1)
        for i in range(self.S):
            berth_cumsum[i + 1] = berth_cumsum[i] + 1 if state.berth[i] == 0 else 0

        ## Schedule all boats that have already arrived and are in queue for mooring
        for boat_idx in state.boats_arrived:
            vessel = self.vessels[boat_idx]  # Get vessel information
            vessel_info = self.vessels[boat_idx]  # Get vessel information
            vessel_size = vessel_info["s"]  # Get vessel size

            ## Append 1 action per boat
            for i in range(self.S - vessel_size + 1):
                if berth_cumsum[i + vessel_size] - berth_cumsum[i] == vessel_size:
                    actions.append((state.time, i, boat_idx))

        ## The only action we can do now, after addind all scheduling options, is to wait for the next event that happens.
        # We can characterize and event as being the next a boat leave the berth, if we have boats in a pending situation, or if a when a new boat arrives, and the next event will be the one that takes less time to happen of these.
        # If the berth is empty in any given moment, we cannot wait for a boat to leave, we can only wait for the next boat to arrive

        time_delta_to_next_departure = min([b + state.time for b in state.berth if b != 0], default=float('inf'))

        time_delta_to_next_arrival = min([self.vessels[boat]["a"] for boat in state.boats_not_arrived], default=float('inf'))

        next_event_time = min(time_delta_to_next_departure, time_delta_to_next_arrival)

        ## Can always try to go the next step without any action
        if next_event_time != float('inf'):
            actions.append((next_event_time, -1, -1))

        return actions  # Return the list of possible actions

    def h(self, node):
        """
        Returns the heuristic valus of the state associated with the node

        Currently the heuristic works as follows:
            For a given state in the search problem, the cost we have to pay is the sum of the flow time of all the scheduled vessels, with this we can see that by never scheduling a vessel we will fall into infinite solution space, since the cost of adding a boat will always be bigger than the cost of not adding it.

            To mitigate this we compute the next time the berth will have a free space the size of each of the boats in waiting assuming the current allocation, and we assume we insert the boat at that time.

            This heuristic is admissible since it never overestimates the cost of reaching the goal, since the minima flow time we can have for allocating a boat is to allocate it at the current time step, and since this is what the heuristic computes, we make sure and overestimation never happens

            !! TALK about consistency for this heuristic although i cannot understand if this is consistent of not
        """
        state = node.state


        def get_next_schedule_time(berth, vessel_size):
            min_ = float('inf')
            for i in range(self.S - vessel_size + 1):
                min_ = min(min_, max(berth[i:i + vessel_size]))
            return min_

        heuristic = 0

        for boat_idx in state.boats_arrived:
            vessel = self.vessels[boat_idx]  # Get vessel information
            next_available_time = get_next_schedule_time(state.berth, vessel["s"])

            heuristic += vessel["w"] * (state.time + next_available_time + vessel["p"] - vessel["a"])

        for boat_idx in state.boats_not_arrived:
            vessel = self.vessels[boat_idx]
            next_available_time = get_next_schedule_time(state.berth, vessel["s"])
            heuristic += vessel["w"] * (next_available_time + vessel["p"] - vessel["a"])

        return heuristic
    # Heuristic 1: Earliest Possible Completion Time
    def heuristic_earliest_completion(self, node):
        """
        Heuristic based on the earliest possible completion time for unscheduled vessels.
        """
        state = node.state
        h_value = 0
        for vessel_idx in state.boats_not_arrived:
            vessel = self.vessels[vessel_idx]
            mooring_time = max(0, state.time + vessel["p"] - vessel["a"])  # Time it takes to complete
            h_value += vessel["w"] * mooring_time  # Weighted by vessel priority
        return h_value


    # Heuristic 2: Minimizing Berth Idle Time
    def heuristic_minimize_idle_time(self, node):
        """
        Heuristic based on minimizing the berth idle time.
        """
        state = node.state
        h_value = 0
        next_event_time = state.time  # Assuming no other vessel mooring currently
        for vessel_idx in state.boats_not_arrived:
            vessel = self.vessels[vessel_idx]
            idle_time = max(0, next_event_time - vessel["a"])  # Time until the next vessel can start mooring
            h_value += vessel["w"] * (vessel["p"] + idle_time)
        return h_value


    # Heuristic 3: Gap-Based Scheduling Heuristic
    def heuristic_gap_based(self, node):
        """
        Heuristic based on gap-based scheduling, trying to minimize idle berth slots.
        """
        state = node.state
        h_value = 0
        berth_cumsum = [0] * (self.S + 1)  # Cumulative sum of berth availability
        for i in range(self.S):
            berth_cumsum[i + 1] = berth_cumsum[i] + 1 if state.berth[i] == 0 else 0

        for vessel_idx in state.boats_not_arrived:
            vessel = self.vessels[vessel_idx]
            vessel_size = vessel["s"]

            for i in range(self.S - vessel_size + 1):
                if berth_cumsum[i + vessel_size] - berth_cumsum[i] == vessel_size:
                    # Vessel can be moored in this berth slot
                    next_available_time = state.time  # Assuming it can be placed immediately
                    h_value += vessel["w"] * (next_available_time + vessel["p"] - vessel["a"])
                    break  # We only need one valid slot to schedule
        return h_value


    # Heuristic 4: Weighted Slack Time Heuristic
    def heuristic_weighted_slack(self, node):
        """
        Heuristic based on the slack time (difference between arrival and current time) for unscheduled vessels.
        """
        state = node.state
        h_value = 0
        for vessel_idx in state.boats_not_arrived:
            vessel = self.vessels[vessel_idx]
            slack_time = max(0, vessel["a"] - state.time)  # Time until vessel arrives
            h_value += vessel["w"] * slack_time
        return h_value


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
        return vessel["w"] * (action_time + vessel["p"] - vessel["a"])

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
        _, berth_space, vessel_space = action
        if berth_space == -1 or vessel_space == -1:
            return c
        return c + self.action_cost(action)  # Return updated cost

    def goal_test(self, state):
        """
        Returns True if the state is a goal.

        Parameters:
        - state (State): The current state.

        Returns:
        - bool: True if the goal is achieved (all vessels moored), False otherwise.
        """
        return all(
            vessel[MOORING_TIME] != -1 for vessel in state.vessels
        )  # Check if all vessels are moored

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
            print(f"Solution : {solution.state.vessels_position} cost : {solution.path_cost} time : {time_end - time_star}")
            print(solution.depth)
            return (
                solution.state.vessels_position
            )  # Return the position of vessels in the solution


if __name__ == "__main__":
    def get_test_files(test_dir="TestePart3"):
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
    test_files = [
        test for test in test_files if "108" in test
    ]  # Filter test files based on naming

    for test in test_files:  # Iterate over the test files
        with open(test) as fh:  # Open the test file
            problem = BAProblem()  # Create an instance of BAProblem
            problem.load(fh)  # Load the problem from the file
            print(f"Test {test}")  # Print the name of the test file
#            with Profile() as p:
            problem.solve()
 #               Stats(p).sort_stats('tottime').print_stats()
