from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain46

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL']),
    make_issue(name="issueB", values=['valueA', 'valueB']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.02068, 'valueB': 0.15096, 'valueC': 0.07096, 'valueD': 0.10234, 'valueE': 0.02188, 'valueF': 0.01676, 'valueG': 1.0, 'valueH': 0.04332, 'valueI': 0.09208, 'valueJ': 0.09371, 'valueK': 0.0, 'valueL': 0.09841},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.19439, 'valueB': 0.04431, 'valueC': 0.01065, 'valueD': 0.33574, 'valueE': 0.0, 'valueF': 0.086, 'valueG': 0.11228, 'valueH': 0.20038, 'valueI': 0.08097, 'valueJ': 1.0, 'valueK': 0.12823, 'valueL': 0.6316, 'valueM': 0.09596, 'valueN': 0.19, 'valueO': 0.46066, 'valueP': 0.11116, 'valueQ': 0.1133, 'valueR': 0.03527, 'valueS': 0.20852, 'valueT': 0.16773, 'valueU': 0.24111, 'valueV': 0.57412, 'valueW': 0.01568, 'valueX': 0.00588, 'valueY': 0.0768, 'valueZ': 0.12486},
        {'valueA': 1.0, 'valueB': 0.07978, 'valueC': 0.11448, 'valueD': 0.27721, 'valueE': 0.47516, 'valueF': 0.0, 'valueG': 0.67292, 'valueH': 0.59984, 'valueI': 0.52577, 'valueJ': 0.18307, 'valueK': 0.05783, 'valueL': 0.18388},
    ],
    weights=[
        0.07216,
        0.47577,
        0.41246,
        0.03961,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.08001, 'valueB': 1.0, 'valueC': 0.02897, 'valueD': 0.08838, 'valueE': 0.02078, 'valueF': 0.38435, 'valueG': 0.39699, 'valueH': 0.71305, 'valueI': 0.19129, 'valueJ': 0.0, 'valueK': 0.43396, 'valueL': 0.29563},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.08334, 'valueB': 0.37043, 'valueC': 0.38402, 'valueD': 0.03349, 'valueE': 0.13319, 'valueF': 0.53144, 'valueG': 0.00205, 'valueH': 0.3965, 'valueI': 0.50861, 'valueJ': 0.0662, 'valueK': 0.06572, 'valueL': 0.30146, 'valueM': 0.71022, 'valueN': 1.0, 'valueO': 0.28969, 'valueP': 0.17174, 'valueQ': 0.123, 'valueR': 0.78614, 'valueS': 0.20943, 'valueT': 0.16432, 'valueU': 0.78519, 'valueV': 0.39966, 'valueW': 0.08532, 'valueX': 0.09646, 'valueY': 0.0, 'valueZ': 0.03491},
        {'valueA': 0.41604, 'valueB': 1.0, 'valueC': 0.24124, 'valueD': 0.0, 'valueE': 0.5517, 'valueF': 0.17633, 'valueG': 0.59349, 'valueH': 0.55769, 'valueI': 0.97144, 'valueJ': 0.4244, 'valueK': 0.33679, 'valueL': 0.72435},
    ],
    weights=[
        0.3134,
        0.11949,
        0.38445,
        0.18266,
    ],
    issues=issues,
    reserved_value=0.4,
)