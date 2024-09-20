class BAProblem:
    """
    A class to represent and solve the Berth Allocation Problem (BAP).

    This class provides methods to load BAP data from an input file, compute the cost
    of a solution, and verify if a given solution meets the problem constraints.
    """

    def __init__(self):
        """
        Initialize the BAProblem class with default values.

        Attributes:
        - S (int): The size of the berth space.
        - N (int): The number of vessels.
        - vessels (list): A list to store information about each vessel.
        """
        self.initial = None
        self.S = 0
        self.N = 0
        self.vessels = []

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
        for i in range(1, self.N + 1):
            ai, pi, si, wi = map(int, data_lines[i].split())
            self.vessels.append({"ai": ai, "pi": pi, "si": si, "wi": wi})

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
            ci = ui + vessel["pi"]
            fi = ci - vessel["ai"]
            total_cost += vessel["wi"] * fi
        return total_cost

    def check(self, sol):
        """
        Check if a given solution satisfies all problem constraints.

        Parameters:
        - sol (list): A list of tuples representing the solution, where each tuple contains
                      the starting mooring time (ui) and the starting berth section (vi)
                      for each vessel.

        Returns:
        - bool: True if the solution is valid and meets all constraints; False otherwise.
        """
        for i, (ui, vi) in enumerate(sol):
            vessel = self.vessels[i]
            if vi + vessel["si"] > self.S:
                return False
            for j in range(i + 1, self.N):
                uj, vj = sol[j]
                other_vessel = self.vessels[j]
                if not (ui + vessel["pi"] <= uj or uj + other_vessel["pi"] <= ui):
                    if not (vi + vessel["si"] <= vj or vj + other_vessel["si"] <= vi):
                        return False
        return True
