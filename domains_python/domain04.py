from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain04

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueF", values=['valueA', 'valueB']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.14629, 'valueB': 0.33701, 'valueC': 0.57837, 'valueD': 0.17711, 'valueE': 0.01108, 'valueF': 1.0, 'valueG': 0.2934, 'valueH': 0.08129, 'valueI': 0.26335, 'valueJ': 0.0945, 'valueK': 0.03917, 'valueL': 0.12826, 'valueM': 0.19271, 'valueN': 0.04373, 'valueO': 0.0, 'valueP': 0.0806, 'valueQ': 0.28466, 'valueR': 0.01896, 'valueS': 0.23828},
        {'valueA': 0.15046, 'valueB': 0.0, 'valueC': 1.0, 'valueD': 0.33036},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.48877, 'valueD': 0.64039},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.60028, 'valueB': 0.0, 'valueC': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.02571,
        0.14084,
        0.39066,
        0.21482,
        0.17048,
        0.05749,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.04215, 'valueB': 0.41107, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.07147, 'valueF': 0.47657, 'valueG': 0.0381, 'valueH': 0.56365, 'valueI': 0.02991, 'valueJ': 0.24492, 'valueK': 0.04496, 'valueL': 0.52687, 'valueM': 0.2348, 'valueN': 0.03968, 'valueO': 0.0267, 'valueP': 0.13574, 'valueQ': 0.67125, 'valueR': 0.06984, 'valueS': 0.26753},
        {'valueA': 0.75819, 'valueB': 1.0, 'valueC': 0.79071, 'valueD': 0.0},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.58776, 'valueD': 0.4259},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.12943, 'valueC': 0.0},
        {'valueA': 1.0, 'valueB': 0.0},
    ],
    weights=[
        0.27487,
        0.02151,
        0.47061,
        0.14735,
        0.06426,
        0.0214,
    ],
    issues=issues,
    reserved_value=0.4,
)