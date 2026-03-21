from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain01

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.06457, 'valueD': 0.22144, 'valueE': 0.06526, 'valueF': 0.48547, 'valueG': 0.64751, 'valueH': 0.43528, 'valueI': 0.02772, 'valueJ': 0.07634, 'valueK': 0.10337},
        {'valueA': 0.88669, 'valueB': 1.0, 'valueC': 0.77662, 'valueD': 0.28788, 'valueE': 0.59442, 'valueF': 0.85704, 'valueG': 0.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.7956, 'valueD': 0.60893, 'valueE': 0.99131, 'valueF': 0.46653, 'valueG': 0.05495, 'valueH': 0.23401, 'valueI': 0.14778, 'valueJ': 0.09799, 'valueK': 0.83664, 'valueL': 0.1952, 'valueM': 0.31236, 'valueN': 0.56166, 'valueO': 0.07848},
        {'valueA': 0.35877, 'valueB': 0.80111, 'valueC': 0.1152, 'valueD': 0.0922, 'valueE': 1.0, 'valueF': 0.02589, 'valueG': 0.17947, 'valueH': 0.0, 'valueI': 0.37585},
    ],
    weights=[
        0.24962,
        0.24273,
        0.37965,
        0.128,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.35013, 'valueC': 0.42541, 'valueD': 0.53595, 'valueE': 0.13737, 'valueF': 0.00188, 'valueG': 0.0346, 'valueH': 0.48104, 'valueI': 0.09336, 'valueJ': 0.1767, 'valueK': 0.0},
        {'valueA': 1.0, 'valueB': 0.86229, 'valueC': 0.80441, 'valueD': 0.09233, 'valueE': 0.72376, 'valueF': 0.69304, 'valueG': 0.0},
        {'valueA': 0.01497, 'valueB': 0.20435, 'valueC': 0.38368, 'valueD': 0.44847, 'valueE': 0.66701, 'valueF': 0.24722, 'valueG': 0.43213, 'valueH': 1.0, 'valueI': 0.32452, 'valueJ': 0.0269, 'valueK': 0.1579, 'valueL': 0.63675, 'valueM': 0.0, 'valueN': 0.54679, 'valueO': 0.02385},
        {'valueA': 0.02663, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.61764, 'valueE': 0.06334, 'valueF': 0.45477, 'valueG': 0.30939, 'valueH': 0.10053, 'valueI': 0.01873},
    ],
    weights=[
        0.18453,
        0.24264,
        0.50496,
        0.06787,
    ],
    issues=issues,
    reserved_value=0.4,
)