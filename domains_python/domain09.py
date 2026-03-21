from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain09

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB']),
    make_issue(name="issueF", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueG", values=['valueA', 'valueB', 'valueC']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 0.35526, 'valueC': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.02679, 'valueC': 0.0},
        {'valueA': 1.0, 'valueB': 0.4713, 'valueC': 0.0},
    ],
    weights=[
        0.15631,
        0.23994,
        0.04163,
        0.05652,
        0.08872,
        0.40693,
        0.00995,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.13707, 'valueB': 0.0, 'valueC': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.39636},
        {'valueA': 0.0, 'valueB': 0.05157, 'valueC': 1.0},
    ],
    weights=[
        0.07304,
        0.05993,
        0.0163,
        0.01263,
        0.50693,
        0.12721,
        0.20396,
    ],
    issues=issues,
    reserved_value=0.4,
)