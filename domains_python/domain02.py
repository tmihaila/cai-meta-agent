from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain02

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueF", values=['valueA', 'valueB']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.20821, 'valueB': 0.32362, 'valueC': 0.62079, 'valueD': 0.14729, 'valueE': 0.36743, 'valueF': 0.02912, 'valueG': 0.59976, 'valueH': 0.12284, 'valueI': 0.07043, 'valueJ': 0.10548, 'valueK': 0.39539, 'valueL': 0.09112, 'valueM': 0.48201, 'valueN': 0.42268, 'valueO': 0.54877, 'valueP': 0.43127, 'valueQ': 0.17274, 'valueR': 0.59701, 'valueS': 0.77776, 'valueT': 0.0, 'valueU': 0.03162, 'valueV': 0.37336, 'valueW': 0.12325, 'valueX': 0.13277, 'valueY': 1.0, 'valueZ': 0.02812},
        {'valueA': 1.0, 'valueB': 0.80287, 'valueC': 0.0},
        {'valueA': 0.16672, 'valueB': 0.07003, 'valueC': 0.00285, 'valueD': 0.18119, 'valueE': 0.98092, 'valueF': 0.0, 'valueG': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.35019},
        {'valueA': 1.0, 'valueB': 0.0},
    ],
    weights=[
        0.11155,
        0.09718,
        0.20828,
        0.4526,
        0.02944,
        0.10095,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.26947, 'valueB': 0.03288, 'valueC': 0.04156, 'valueD': 0.34405, 'valueE': 0.09585, 'valueF': 0.38634, 'valueG': 0.12031, 'valueH': 0.05899, 'valueI': 0.02267, 'valueJ': 0.42021, 'valueK': 0.06615, 'valueL': 0.00894, 'valueM': 0.1339, 'valueN': 0.63075, 'valueO': 0.1229, 'valueP': 0.17865, 'valueQ': 0.24229, 'valueR': 0.34444, 'valueS': 0.2116, 'valueT': 1.0, 'valueU': 0.18143, 'valueV': 0.57944, 'valueW': 0.12025, 'valueX': 0.0, 'valueY': 0.33861, 'valueZ': 0.32137},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.22691},
        {'valueA': 0.0, 'valueB': 0.74694, 'valueC': 0.14063, 'valueD': 0.29505, 'valueE': 0.44774, 'valueF': 1.0, 'valueG': 0.19785},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.41916},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.35501,
        0.0659,
        0.08512,
        0.03482,
        0.38064,
        0.07851,
    ],
    issues=issues,
    reserved_value=0.4,
)