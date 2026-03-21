from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain36

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.40118, 'valueD': 0.81772},
        {'valueA': 0.22361, 'valueB': 0.02438, 'valueC': 0.2307, 'valueD': 0.30508, 'valueE': 0.3516, 'valueF': 0.02096, 'valueG': 0.00899, 'valueH': 0.02596, 'valueI': 0.12984, 'valueJ': 0.0412, 'valueK': 0.04097, 'valueL': 0.15795, 'valueM': 0.09783, 'valueN': 0.0, 'valueO': 0.1776, 'valueP': 0.12035, 'valueQ': 0.0588, 'valueR': 1.0, 'valueS': 0.01934, 'valueT': 0.0379, 'valueU': 0.20431, 'valueV': 0.01399, 'valueW': 0.12566, 'valueX': 0.31485, 'valueY': 0.11345, 'valueZ': 0.02108},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.13661, 'valueB': 1.0, 'valueC': 0.01992, 'valueD': 0.34111, 'valueE': 0.00421, 'valueF': 0.02531, 'valueG': 0.13253, 'valueH': 0.26521, 'valueI': 0.34088, 'valueJ': 0.60543, 'valueK': 0.00421, 'valueL': 0.18968, 'valueM': 0.0, 'valueN': 0.19906, 'valueO': 0.53141, 'valueP': 0.7383, 'valueQ': 0.03127},
    ],
    weights=[
        0.15267,
        0.65453,
        0.13226,
        0.06054,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.08518, 'valueB': 0.0365, 'valueC': 1.0, 'valueD': 0.0},
        {'valueA': 0.30029, 'valueB': 0.17581, 'valueC': 0.35455, 'valueD': 0.81216, 'valueE': 0.0, 'valueF': 0.10215, 'valueG': 1.0, 'valueH': 0.49136, 'valueI': 0.96312, 'valueJ': 0.49681, 'valueK': 0.86005, 'valueL': 0.73567, 'valueM': 0.37213, 'valueN': 0.0782, 'valueO': 0.42841, 'valueP': 0.38749, 'valueQ': 0.3018, 'valueR': 0.16429, 'valueS': 0.14277, 'valueT': 0.03799, 'valueU': 0.14671, 'valueV': 0.16136, 'valueW': 0.65181, 'valueX': 0.03334, 'valueY': 0.29604, 'valueZ': 0.2431},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.07812, 'valueB': 0.31679, 'valueC': 0.75205, 'valueD': 0.51595, 'valueE': 1.0, 'valueF': 0.4392, 'valueG': 0.63075, 'valueH': 0.43886, 'valueI': 0.25271, 'valueJ': 0.38682, 'valueK': 0.523, 'valueL': 0.38329, 'valueM': 0.0, 'valueN': 0.00788, 'valueO': 0.22801, 'valueP': 0.03432, 'valueQ': 0.84914},
    ],
    weights=[
        0.01116,
        0.19771,
        0.40954,
        0.38159,
    ],
    issues=issues,
    reserved_value=0.4,
)