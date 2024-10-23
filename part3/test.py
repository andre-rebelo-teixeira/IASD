from solution import BAProblem

contents = ""

with open("cost.txt", "r") as f:
    contents = f.readlines()

passed_s = [l.replace("\n", "") for l in contents if not l.startswith("!")]
not_passed_s = [
    l.replace("!", "").replace("\n", "") for l in contents if l.startswith("!")
]

passed = []
for l in passed_s:
    test, rest = l.split(": ")
    answer, cost = rest.split(" - ")
    passed.append((test, answer, cost))

not_passed = []
for l in not_passed_s:
    test, rest = l.split(": ")
    answer, cost = rest.split(" - ")
    not_passed.append((test, answer, cost))

print("Checking previous passed test: ")
for test in passed:
    with open("TestePart3/ex" + test[0] + ".dat", "r") as f:
        problem = BAProblem()
        problem.load(f)
        sol = problem.solve()

        if problem.cost(sol) == float(test[2]):
            print(f"    Test {test[0]}: Passed")
        else:
            print(f"    Test {test[0]}: Failed")

print("\nChecking previous failed test: ")
for test in not_passed:
    with open("TestePart3/ex" + test[0] + ".dat", "r") as f:
        problem = BAProblem()
        problem.load(f)
        sol = problem.solve()

        if problem.cost(sol) < float(test[2]):
            print(f"    Test {test[0]}: Maybe passed")
        else:
            print(f"    Test {test[0]}: Failed")
