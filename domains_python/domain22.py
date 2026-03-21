from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain22

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB']),
    make_issue(name="issueB", values=['valueA', 'valueB']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueF", values=['valueA', 'valueB']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 0.3145, 'valueC': 1.0, 'valueD': 0.11371},
        {'valueA': 0.33316, 'valueB': 0.07352, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.41932},
        {'valueA': 0.00701, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.19745, 'valueE': 0.46161, 'valueF': 0.05966},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.24769,
        0.28595,
        0.05071,
        0.03938,
        0.05433,
        0.32194,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.03505, 'valueB': 0.13064, 'valueC': 1.0, 'valueD': 0.0},
        {'valueA': 0.41757, 'valueB': 0.94826, 'valueC': 0.0, 'valueD': 1.0, 'valueE': 0.06969},
        {'valueA': 0.48527, 'valueB': 0.15218, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.4081, 'valueF': 0.0133},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.16797,
        0.0675,
        0.24822,
        0.10929,
        0.14126,
        0.26576,
    ],
    issues=issues,
    reserved_value=0.4,
)