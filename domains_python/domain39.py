from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain39

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.43685, 'valueB': 1.0, 'valueC': 0.20353, 'valueD': 0.25149, 'valueE': 0.01952, 'valueF': 0.58787, 'valueG': 0.02579, 'valueH': 0.08958, 'valueI': 0.51161, 'valueJ': 0.38048, 'valueK': 0.09656, 'valueL': 0.24821, 'valueM': 0.0, 'valueN': 0.10875, 'valueO': 0.39716, 'valueP': 0.67816, 'valueQ': 0.00185, 'valueR': 0.4422, 'valueS': 0.23838, 'valueT': 0.11708, 'valueU': 0.24629, 'valueV': 0.42951, 'valueW': 0.00705, 'valueX': 0.12785, 'valueY': 0.46736},
        {'valueA': 0.34346, 'valueB': 0.2971, 'valueC': 0.11441, 'valueD': 0.16256, 'valueE': 0.3226, 'valueF': 0.2027, 'valueG': 0.18021, 'valueH': 0.0, 'valueI': 0.17979, 'valueJ': 0.86909, 'valueK': 0.12948, 'valueL': 0.22214, 'valueM': 0.25596, 'valueN': 0.80698, 'valueO': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.25683, 'valueB': 0.02839, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.89162},
    ],
    weights=[
        0.27754,
        0.16827,
        0.30865,
        0.15372,
        0.09182,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.14115, 'valueB': 0.02311, 'valueC': 0.05451, 'valueD': 0.04365, 'valueE': 0.20753, 'valueF': 0.13399, 'valueG': 0.0, 'valueH': 0.91397, 'valueI': 0.17971, 'valueJ': 0.18704, 'valueK': 1.0, 'valueL': 0.41461, 'valueM': 0.21369, 'valueN': 0.23663, 'valueO': 0.12643, 'valueP': 0.01583, 'valueQ': 0.22796, 'valueR': 0.15704, 'valueS': 0.22913, 'valueT': 0.00419, 'valueU': 0.23971, 'valueV': 0.00369, 'valueW': 0.10544, 'valueX': 0.12385, 'valueY': 0.44668},
        {'valueA': 0.10984, 'valueB': 0.0479, 'valueC': 0.13784, 'valueD': 0.01593, 'valueE': 0.50448, 'valueF': 0.14031, 'valueG': 0.01719, 'valueH': 1.0, 'valueI': 0.05516, 'valueJ': 0.67356, 'valueK': 0.13677, 'valueL': 0.0245, 'valueM': 0.35922, 'valueN': 0.01172, 'valueO': 0.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.11019, 'valueB': 0.8068, 'valueC': 1.0, 'valueD': 0.21403, 'valueE': 0.0},
    ],
    weights=[
        0.21793,
        0.10467,
        0.25903,
        0.34561,
        0.07276,
    ],
    issues=issues,
    reserved_value=0.4,
)