from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain43

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM']),
    make_issue(name="issueE", values=['valueA', 'valueB']),
    make_issue(name="issueF", values=['valueA', 'valueB']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.18838, 'valueB': 0.49264, 'valueC': 0.82649, 'valueD': 1.0, 'valueE': 0.0, 'valueF': 0.05705},
        {'valueA': 0.0, 'valueB': 0.03391, 'valueC': 0.40758, 'valueD': 0.58729, 'valueE': 1.0, 'valueF': 0.14227, 'valueG': 0.61738, 'valueH': 0.34596, 'valueI': 0.30078, 'valueJ': 0.74879, 'valueK': 0.40981, 'valueL': 0.02455, 'valueM': 0.28899},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.45119, 'valueB': 0.3522, 'valueC': 0.04689, 'valueD': 0.00776, 'valueE': 0.09038, 'valueF': 0.16303, 'valueG': 0.05142, 'valueH': 0.0, 'valueI': 0.12785, 'valueJ': 0.58046, 'valueK': 1.0, 'valueL': 0.52525, 'valueM': 0.12042},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
    ],
    weights=[
        0.0297,
        0.04991,
        0.00177,
        0.18143,
        0.71969,
        0.0175,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.03808, 'valueC': 0.0854, 'valueD': 0.03408, 'valueE': 0.29674, 'valueF': 0.0},
        {'valueA': 0.04315, 'valueB': 0.63896, 'valueC': 0.00405, 'valueD': 0.13272, 'valueE': 1.0, 'valueF': 0.73, 'valueG': 0.01378, 'valueH': 0.00878, 'valueI': 0.09109, 'valueJ': 0.93472, 'valueK': 0.0, 'valueL': 0.12402, 'valueM': 0.70622},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.02809, 'valueB': 0.06409, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.06505, 'valueF': 0.09714, 'valueG': 0.19219, 'valueH': 0.02629, 'valueI': 0.64465, 'valueJ': 0.04614, 'valueK': 0.10188, 'valueL': 0.04286, 'valueM': 0.42478},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.0},
    ],
    weights=[
        0.15602,
        0.1356,
        0.13246,
        0.25086,
        0.25012,
        0.07494,
    ],
    issues=issues,
    reserved_value=0.4,
)