from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain47

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.10278, 'valueB': 0.22711, 'valueC': 0.21868, 'valueD': 0.10385, 'valueE': 0.0517, 'valueF': 0.04586, 'valueG': 0.12901, 'valueH': 1.0, 'valueI': 0.05687, 'valueJ': 0.00486, 'valueK': 0.38963, 'valueL': 0.15792, 'valueM': 0.13066, 'valueN': 0.16001, 'valueO': 0.01922, 'valueP': 0.13129, 'valueQ': 0.1974, 'valueR': 0.10046, 'valueS': 0.22011, 'valueT': 0.0, 'valueU': 0.00798, 'valueV': 0.02105, 'valueW': 0.03198, 'valueX': 0.04639, 'valueY': 0.24023, 'valueZ': 0.48451},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.41379, 'valueD': 0.87259, 'valueE': 0.5369, 'valueF': 0.02257},
        {'valueA': 1.0, 'valueB': 0.0992, 'valueC': 0.0, 'valueD': 0.00869, 'valueE': 0.00439, 'valueF': 0.93733},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.03194, 'valueD': 0.28718},
    ],
    weights=[
        0.48058,
        0.02159,
        0.32561,
        0.17222,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.20714, 'valueC': 0.09964, 'valueD': 0.21511, 'valueE': 0.50128, 'valueF': 0.04848, 'valueG': 0.21886, 'valueH': 0.03172, 'valueI': 0.10452, 'valueJ': 0.04611, 'valueK': 0.31604, 'valueL': 0.21269, 'valueM': 0.0474, 'valueN': 0.01455, 'valueO': 0.15023, 'valueP': 0.12293, 'valueQ': 0.23388, 'valueR': 0.03994, 'valueS': 0.04935, 'valueT': 0.0, 'valueU': 0.14195, 'valueV': 0.0056, 'valueW': 0.0112, 'valueX': 0.20277, 'valueY': 0.77131, 'valueZ': 0.28503},
        {'valueA': 0.70322, 'valueB': 0.3027, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.54413, 'valueF': 0.89703},
        {'valueA': 0.12073, 'valueB': 0.45704, 'valueC': 0.0, 'valueD': 0.355, 'valueE': 0.96073, 'valueF': 1.0},
        {'valueA': 0.0, 'valueB': 0.51501, 'valueC': 1.0, 'valueD': 0.15393},
    ],
    weights=[
        0.21516,
        0.0606,
        0.23845,
        0.48579,
    ],
    issues=issues,
    reserved_value=0.4,
)