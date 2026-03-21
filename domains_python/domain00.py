from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain00

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.03156},
        {'valueA': 0.02854, 'valueB': 0.0, 'valueC': 0.18426, 'valueD': 0.16216, 'valueE': 1.0, 'valueF': 0.01914, 'valueG': 0.58154, 'valueH': 0.23052},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.47162, 'valueB': 0.15076, 'valueC': 0.49341, 'valueD': 0.0, 'valueE': 0.68999, 'valueF': 0.51706, 'valueG': 0.25717, 'valueH': 1.0, 'valueI': 0.20764},
        {'valueA': 0.34474, 'valueB': 0.79995, 'valueC': 0.06008, 'valueD': 0.03069, 'valueE': 0.44156, 'valueF': 0.03404, 'valueG': 0.01181, 'valueH': 0.29136, 'valueI': 0.09436, 'valueJ': 0.0, 'valueK': 0.87914, 'valueL': 0.15109, 'valueM': 1.0, 'valueN': 0.14738, 'valueO': 0.14876},
    ],
    weights=[
        0.06667,
        0.04186,
        0.07844,
        0.70691,
        0.10612,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 0.54104, 'valueC': 1.0},
        {'valueA': 0.09595, 'valueB': 1.0, 'valueC': 0.44469, 'valueD': 0.0651, 'valueE': 0.07396, 'valueF': 0.33373, 'valueG': 0.0, 'valueH': 0.11363},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.12574, 'valueB': 0.37198, 'valueC': 0.70969, 'valueD': 0.0, 'valueE': 0.02171, 'valueF': 0.37156, 'valueG': 0.01843, 'valueH': 0.09554, 'valueI': 1.0},
        {'valueA': 0.3732, 'valueB': 0.75953, 'valueC': 0.0076, 'valueD': 0.40753, 'valueE': 0.06418, 'valueF': 0.69877, 'valueG': 0.11276, 'valueH': 0.09912, 'valueI': 0.0, 'valueJ': 1.0, 'valueK': 0.22672, 'valueL': 0.00841, 'valueM': 0.04506, 'valueN': 0.34734, 'valueO': 0.21127},
    ],
    weights=[
        0.06209,
        0.26999,
        0.10377,
        0.46332,
        0.10083,
    ],
    issues=issues,
    reserved_value=0.4,
)