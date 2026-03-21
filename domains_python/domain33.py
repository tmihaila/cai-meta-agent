from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain33

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.12185, 'valueB': 0.12627, 'valueC': 0.0, 'valueD': 0.005, 'valueE': 1.0, 'valueF': 0.02294, 'valueG': 0.0332, 'valueH': 0.17133, 'valueI': 0.02705, 'valueJ': 0.23749, 'valueK': 0.00445, 'valueL': 0.01006, 'valueM': 0.00155, 'valueN': 0.13918, 'valueO': 0.46572, 'valueP': 0.12714, 'valueQ': 0.028, 'valueR': 0.26716},
        {'valueA': 0.52011, 'valueB': 1.0, 'valueC': 0.47284, 'valueD': 0.07819, 'valueE': 0.25307, 'valueF': 0.58355, 'valueG': 0.22319, 'valueH': 0.40536, 'valueI': 0.11456, 'valueJ': 0.08578, 'valueK': 0.42575, 'valueL': 0.95927, 'valueM': 0.02008, 'valueN': 0.0, 'valueO': 0.35},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.00219, 'valueB': 0.10215, 'valueC': 0.03139, 'valueD': 0.22437, 'valueE': 0.20816, 'valueF': 0.35675, 'valueG': 0.10247, 'valueH': 0.24359, 'valueI': 1.0, 'valueJ': 0.0, 'valueK': 0.41699, 'valueL': 0.15942},
    ],
    weights=[
        0.21776,
        0.05051,
        0.36,
        0.37173,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.13123, 'valueB': 0.55953, 'valueC': 0.04898, 'valueD': 1.0, 'valueE': 0.03501, 'valueF': 0.1807, 'valueG': 0.0, 'valueH': 0.18327, 'valueI': 0.10354, 'valueJ': 0.302, 'valueK': 0.27445, 'valueL': 0.69809, 'valueM': 0.0839, 'valueN': 0.20529, 'valueO': 0.32828, 'valueP': 0.04132, 'valueQ': 0.11465, 'valueR': 0.49284},
        {'valueA': 0.0501, 'valueB': 0.20671, 'valueC': 0.61642, 'valueD': 0.03916, 'valueE': 0.0, 'valueF': 0.16882, 'valueG': 0.69185, 'valueH': 0.13512, 'valueI': 0.20792, 'valueJ': 0.78585, 'valueK': 0.58205, 'valueL': 1.0, 'valueM': 0.23541, 'valueN': 0.56327, 'valueO': 0.72339},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.8022, 'valueC': 0.50553, 'valueD': 0.14863, 'valueE': 0.44314, 'valueF': 0.65836, 'valueG': 0.8522, 'valueH': 0.0, 'valueI': 0.27953, 'valueJ': 0.13024, 'valueK': 0.11, 'valueL': 0.83184},
    ],
    weights=[
        0.00314,
        0.37163,
        0.39981,
        0.22542,
    ],
    issues=issues,
    reserved_value=0.4,
)