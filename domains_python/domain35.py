from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain35

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.0139, 'valueB': 0.00893, 'valueC': 0.11764, 'valueD': 0.14567, 'valueE': 1.0, 'valueF': 0.00637, 'valueG': 0.0, 'valueH': 0.13716},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.20116},
        {'valueA': 1.0, 'valueB': 0.23864, 'valueC': 0.0, 'valueD': 0.37774},
        {'valueA': 0.12588, 'valueB': 0.14388, 'valueC': 0.27064, 'valueD': 0.0, 'valueE': 1.0, 'valueF': 0.93839, 'valueG': 0.38681},
        {'valueA': 0.20397, 'valueB': 0.0, 'valueC': 0.04356, 'valueD': 1.0, 'valueE': 0.19705, 'valueF': 0.23327},
    ],
    weights=[
        0.08687,
        0.53816,
        0.24125,
        0.11864,
        0.01508,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.05674, 'valueB': 0.40806, 'valueC': 0.0, 'valueD': 0.81904, 'valueE': 0.26447, 'valueF': 0.07585, 'valueG': 1.0, 'valueH': 0.03363},
        {'valueA': 0.02598, 'valueB': 0.0, 'valueC': 1.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.54296, 'valueD': 0.54082},
        {'valueA': 0.01671, 'valueB': 1.0, 'valueC': 0.25179, 'valueD': 0.15209, 'valueE': 0.43467, 'valueF': 0.0, 'valueG': 0.7154},
        {'valueA': 0.26361, 'valueB': 1.0, 'valueC': 0.36805, 'valueD': 0.0, 'valueE': 0.08932, 'valueF': 0.36444},
    ],
    weights=[
        0.05439,
        0.0315,
        0.2754,
        0.1211,
        0.51761,
    ],
    issues=issues,
    reserved_value=0.4,
)