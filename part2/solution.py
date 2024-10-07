import search
import utils
import time
import os
import numpy as np

from cProfile import Profile
from pstats import SortKey, Stats

class State:
    '''
    A class to represent a state in the BAProblem class

    Since the __hash__ and __eq__ methods are implemented, the State class can be used as a key in a dictionary, meaning we donot need to lose time checking if a state is already in the explored set.

    Attributes:
        time (int): The time of the state
        vessels (list): A List of a dictionary of the vessels and the correct state

    Methods:
        __str__(): Get the string representation of the object
        __hash__(): Get the hash of the object
        __eq__(): Check if the object is equal to another object
    '''
    def __init__(self, time : int, vessels : np.ndarray, cost = 0):
        self.time = time
        self.vessels = vessels
        self.cost = cost
        # where the m is the asssigned mooring time, b is the assigned berth time and i is the index of the boat
        # vessels is a numpy array with N lines an 3 columns where N is the number of boats
        # the first column is the mooring time, the second column is the berth time and the third column is the index of the boat

    @property
    def vessels_position(self):
        return self.vessels[:, 0:2].tolist()

    def __str__(self) -> str:
        '''
            Get the string representation of the object

            Returns:
                str: The string representation of the object

        '''
        return f"State({self.time}, {self.vessels})"

    def __hash__ (self) -> int:
        '''
            Get the hash of the object

            Returns:
                int: The hash of the object
        '''
        return hash((self.time, self.vessels.tobytes()))

    def __eq__(self, other) -> bool:
        '''
            Check if the object is equal to another object

            Parameters:
                other (State): The other object to compare to

            Returns:
                bool: True if the objects are equal, False otherwise
    ('''
        return (self.vessels == other.vessels).all()

    def __lt__(self, other):
        return self.cost < other.cost

class BAProblem(search.Problem):
    def __init__(self):

        self.initial = None
        self.vessels = []
        self.berth_size = 0
    @staticmethod
    def create_vessel_dict(morring_time, berth_section, index):
        ## Create a dictionary of state for a vessel
        return {"m": morring_time, "b": berth_section, "i" :  index}

    def load(self, fh):
        """
        Load a BAP problem from the file object.

        Parameters:
        - fh: A file object representing the input file.
        """
        lines = fh.readlines()
        data_lines = [
            line.strip() for line in lines if not line.startswith("#") and line.strip()
        ]
        self.S, self.N = map(int, data_lines[0].split())
        self.vessels = []
        for i in range(1, self.N + 1) :
            # arrival time, processing time, section size, weigth
            ai, pi, si, wi = map(int, data_lines[i].split())
            self.vessels.append({"a": ai, "p": pi, "s": si, "w": wi})

        ## Each vessel is a line represented by the schedule mooring time, schedule berth section and the index of the vessel
        vessels_state_array = np.ones((self.N, 3), dtype=int) * -1;
        vessels_state_array[:, 2] = np.arange(self.N)

        self.initial = State(0, vessels_state_array, 0)

    def result(self, state : State, action : tuple):
        """
        Returns the state that results from executing
        the given action in the given state.
        """
        state_time, berth_space, vessel_idx = action

        ## Action is to moore a boat
        if vessel_idx != -1:
            new_vessels = state.vessels.copy()  # Copy the current vessels list
            new_vessels[vessel_idx, 0:2] = [state_time, berth_space]
            return State(state.time, new_vessels, state.cost + self.compute_cost(state, action))  # Return a new State object

        ## Action is to advance the time
        else:
            return State(state_time, state.vessels.copy())

    def actions(self, state : State) -> list:
        """
        Returns the list of actions that can be executed in
        the given state.
        """
        actions = []

        vessel_state_array = state.vessels

        ## Get the boats that are not yet scheduled
        boats_not_scheduled = vessel_state_array[
            (vessel_state_array[:, 0] == -1), :
        ]       ## Get the boats that are scheduled

        boats_scheduled = vessel_state_array[vessel_state_array[:, 0] != -1 , :]

        ## Create array represent berth space
        berth = np.ones(self.S, dtype=int)

        ## Fill the berth array with the boats that are scheduled
        for boat in boats_scheduled:
            ## Get the vessel information that came from the input file using the index of the boat
            vessel_info = self.vessels[boat[2]]

            if state.time < boat[0] + vessel_info["p"]:
                berth[boat[1] : boat[1] + vessel_info["s"]] = 0

        ## Compute from the boats that are not scheduled the ones that have already arrived to the port and are awaiting mooring

        boats_arrived = []
        #boats_arrived = boats_not_scheduled[boats_not_scheduled[:, 0] <= state.time, :]

        boats_arrived = [boat for boat in boats_not_scheduled if state.time >= self.vessels[boat[2]]["a"]]

        actions = []



        for boat in boats_arrived:
            vessel_info = self.vessels[boat[2]]
            ## Get all the position the boat can be moored into the berth
            convolved_spaces = np.convolve(berth, np.ones(vessel_info["s"], dtype=int), mode='valid')
            open_space = np.where(convolved_spaces ==  vessel_info["s"])[0]

            actions.extend([(state.time, space, boat[2]) for space in open_space])

        if actions == []:
            actions.append((state.time + 1, -1, -1))

        return actions

    def goal_test(self, state):
        """
            Returns True if the state is a goal.
        """
        return np.all(state.vessels[:, 0] != -1)

    def path_cost(self, c, state1, action, state2):
        """
            Returns the cost of a solution path that arrives
            at state2 from state1 via action, assuming cost c
            to get up to state1.
        """
        cost = c
        time, berth_space, vessel_idx = action
        if vessel_idx == -1:
            return cost

        vessel = self.vessels[vessel_idx]

        arrival_time, processing_time, size, weight = vessel.values()
        departure_time = time + processing_time

        flow_time = departure_time - arrival_time
        return cost + weight * flow_time

    def solve(self):
        """
            Calls the uninformed search algorithm chosen.
            Returns a solution using the specified format.
        """
        time_start = time.time()
        solution = search.best_first_graph_search(self, lambda node : node.path_cost, display=False)
        if solution is not None:
            return solution.state.vessels_position


    def compute_cost(self, state, action):
        '''
        Compute the cost of an action
        '''
        time, berth_space, vessel_idx = action
        vessels_not_schedule = state.vessels[state.vessels[:, 0] == -1, :]

        total_cost = 0

        for vessel in vessels_not_schedule:
            if self.vessels[vessel[2]]["a"] < state.time:
                ## Compute the best case cost for all the vehicles that are not yet scheduled but are already in waiting
                total_cost += (state.time + 1 + self.vessels[vessel[2]]["p"] - self.vessels[vessel[2]]["a"]) + self.vessels[vessel[2]]["w"]

        return total_cost +  self.vessels[vessel_idx]["w"] * (time - self.vessels[vessel_idx]["a"] + self.vessels[vessel_idx]["p"])

if __name__ == "__main__":
    def  get_test_files(test_dir = 'Tests'):
        if test_dir  in os.listdir():
            return [test_dir + '/' + file for file in os.listdir(test_dir) if file.endswith('.dat')]
        return []

    test_files = get_test_files()

    ## Without profiling

    test_files = [test for test in test_files if '108' in test]

    for test in test_files:
        with open(test) as fh:
            problem = BAProblem()
            problem.load(fh)
            print(f"Test {test}")

            with Profile() as profile:
                print(f"Solution is {problem.solve()}" )
                (
                    Stats(profile).sort_stats('cumulative').print_stats()
                )
