from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain25

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueD", values=['valueA', 'valueB']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.08965, 'valueC': 0.25476, 'valueD': 0.09194, 'valueE': 0.88203, 'valueF': 0.2318, 'valueG': 0.0, 'valueH': 0.03695},
        {'valueA': 0.26217, 'valueB': 0.41744, 'valueC': 0.26528, 'valueD': 0.22322, 'valueE': 0.0, 'valueF': 0.01792, 'valueG': 0.0306, 'valueH': 0.00141, 'valueI': 1.0, 'valueJ': 0.8898, 'valueK': 0.56463, 'valueL': 0.68529, 'valueM': 0.76532, 'valueN': 0.02812, 'valueO': 0.1927, 'valueP': 0.33679, 'valueQ': 0.546, 'valueR': 0.19093, 'valueS': 0.04152, 'valueT': 0.05713, 'valueU': 0.05456, 'valueV': 0.4248, 'valueW': 0.47724, 'valueX': 0.10939, 'valueY': 0.53642, 'valueZ': 0.68778},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.67722, 'valueB': 0.12633, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.29669},
    ],
    weights=[
        0.14584,
        0.04098,
        0.1174,
        0.1877,
        0.50808,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.10029, 'valueB': 0.23704, 'valueC': 0.0, 'valueD': 0.29559, 'valueE': 0.34195, 'valueF': 1.0, 'valueG': 0.70984, 'valueH': 0.76858},
        {'valueA': 0.04056, 'valueB': 0.90794, 'valueC': 0.19946, 'valueD': 0.34567, 'valueE': 0.38464, 'valueF': 0.24216, 'valueG': 0.9035, 'valueH': 0.03595, 'valueI': 0.60949, 'valueJ': 1.0, 'valueK': 0.13954, 'valueL': 0.11886, 'valueM': 0.37416, 'valueN': 0.08104, 'valueO': 0.56724, 'valueP': 0.74354, 'valueQ': 0.0, 'valueR': 0.23399, 'valueS': 0.07536, 'valueT': 0.22654, 'valueU': 0.30368, 'valueV': 0.24775, 'valueW': 0.1272, 'valueX': 0.01668, 'valueY': 0.08015, 'valueZ': 0.41127},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.05001, 'valueC': 0.0, 'valueD': 0.25523, 'valueE': 0.81412},
    ],
    weights=[
        0.0475,
        0.01816,
        0.44308,
        0.41792,
        0.07334,
    ],
    issues=issues,
    reserved_value=0.4,
)