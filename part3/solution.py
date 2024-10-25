from ast import BitXor
import search  # Importing the search module which contains algorithms for solving search problems
import time  # Importing time module for time-related functions
import os  # Importing os module for interacting with the operating system
from itertools import groupby
import numpy as np
from functools import lru_cache

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

    def result(self, state: State, action: tuple):
        """
        Returns the state that results from executing the given action in the given state.

        Parameters:
        - state (State): The current state.
        - action (tuple): The action to be performed, containing the new time, berth space, and vessel index. In each action a vessel is always moored, if for a state the berth is full, then the action will more a vessel in the a future time

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

        ## Update the state variables to reflect the time of the new state, since new boats can arrive in between the time of the state and the time of the action
        if state.time < action_time:
            for boat_not_arrived in boats_not_arrived[:]:
                if self.vessels[boat_not_arrived]["a"] <= action_time:
                    boats_arrived.append(boat_not_arrived)
                    boats_not_arrived.remove(boat_not_arrived)

        ## Update the state variables to reflect the boat that was moored
        boats_scheduled.append(boat)
        boats_arrived.remove(boat)
        
        ## Update the berth occupancy list taking into consideration how much time passed between the two actions
        berth = [max(0, b - (action_time - state.time)) for b in berth]

        ## Insert the boat in the berth
        for i in range (self.vessels[boat]["s"]):
            berth[berth_space + i] = self.vessels[boat]["p"] 

        ## Update the state of the boat that was moored
        vessels_state_list[boat] = action

        ## Return the new updated state
        return State(
            action_time,
            vessels_state_list,
            berth,
            boats_scheduled,
            boats_arrived,
            boats_not_arrived,
            state.cost + self.action_cost(action),
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def get_next_available_time_and_berth(berth, vessel_size):
        '''
        This method computes the next time and position a vessel is able to be inserted in the berth, given the current berth occupancy, it returns only the first possible position and time, since otherwise the branching factor would be too big for the rest of our code to handle, this could lead to a suboptimal solution, but the tradeoff needed to be done

        Parameters:
            - berth (list): A list representing the current berth occupancy
            - vessel_size (int): The size of the vessel to be inserted

        Returns:
            - tuple: A tuple containing the time and position the vessel can be inserted

        '''
        berth = tuple(berth)
        min_time = float('inf')
        berth_new = -1

        ## Moves through all the subarrays of the berth that have the size of the vessel trying to be inserted, and computes the minimum time required
        for i in range(len(berth) - vessel_size + 1):
            min_time_new = max(berth[i:i + vessel_size])
            if min_time_new < min_time:
                min_time = min_time_new
                berth_new = i

        return min_time, berth_new

    def generate_feasible_action_for_boat(self, berth, vessel_idx, vessel, state):
        '''
        This method generates a feasible action for a given boat, this action is the one that will be executed in the current state, and will be the one that will be used to generate the next state. This takes into consideration that a certain boat might have an opening in the berth before arriving, but can only be inserted when it arrived

        Parameters:
            - berth (list): A list representing the current berth occupancy
            - vessel_idx (int): The index of the vessel to be inserted
            - vessel (dict): The dictionary representing the vessel to be inserted
            - state (State): The current state
        
        Returns:
            - tuple: A tuple containing the time, position and index of the vessel to be inserted
        '''
        time, berth_space = self.get_next_available_time_and_berth(tuple(berth), vessel["s"])

        return (max(time + state.time, vessel["a"]), berth_space, vessel_idx)

    def actions(self, state: State) -> list:
        """
        Returns the list of actions that can be executed in the given state.

        Parameters:
        - state (State): The current state.

        Returns:
        - list: A list of possible actions that can be performed.
        """
        actions = []  # Initialize actions list

        ## Itterates over all the unscheduled boats and generates the first feaseble action for each 
        for boat_idx in state.boats_arrived[:] + state.boats_not_arrived[:]:
            vessel = self.vessels[boat_idx]  # Get vessel information
            vessel_info = self.vessels[boat_idx]  # Get vessel information
            vessel_size = vessel_info["s"]  # Get vessel size

            actions.append(self.generate_feasible_action_for_boat(state.berth, boat_idx, vessel, state))

        return actions  # Return the list of possible actions

    def h(self, node):
        """
        Returns the heuristic valus of the state associated with the node

        The current heuristic itterates over all the boats already arrived and not scheduled and computes the minimal weighted flow that it can be acheived when scheduling it, knowing the current berth occupation. 

        This means that for all the boats in waiting we compute the next time that the berth will have an openning of their size, we do not care for conflits in these hipotetical insertions, and this way we make sure we can never estimate the cost of the heuristic to be higher that the real cost of the solution

        From this we know that our heuristic is admissible since it never overestimates the cost of the solution

        We can also see that the heuristic is consistent since the cost of the solution is always greater or equal to the cost of the heuristic, since in out best case scenario when we apply an action and insert a boat, we leave enough space in the bearth all the boats in waiting, meaning the heuristic of the node we move to MUST always be less or equal to the cost of a node we move to and the heurist.

        
        Parameters:
            - node (Node): The node to be evaluated.
        
        Returns:
            - int: The heuristic value of the node.
        """
        state = node.state

        heuristic = 0

        @lru_cache(maxsize = None)
        def get_next_available_berth_time(berth, vessel_size, berth_size): 
            '''
            Compute the next time a vessel can be inserted in the berth

            Parameters:
                - berth (list): A list representing the current berth occupancy
                - vessel_size (int): The size of the vessel to be inserted
                - berth_size (int): The size of the berth
            
            Returns:
                - int: The time the vessel can be inserted
            '''
            return min( [max(berth[i:i + vessel_size]) for i in range(berth_size - vessel_size + 1)] )


        berth = tuple(state.berth)
        for boat_idx in state.boats_arrived:
            vessel = self.vessels[boat_idx]  # Get vessel information
            next_available_time =  get_next_available_berth_time(berth, vessel["s"], self.S)
            heuristic += vessel["w"] * (state.time + next_available_time + vessel["p"] - vessel["a"])

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
        solution = search.astar_search(self, self.h, False)
        if solution is not None:
            return (
                solution.state.vessels_position
            )  # Return the position of vessels in the solution
