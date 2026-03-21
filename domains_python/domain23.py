from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain23

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.0591, 'valueB': 0.10655, 'valueC': 0.0308, 'valueD': 0.30658, 'valueE': 0.18433, 'valueF': 0.30313, 'valueG': 1.0, 'valueH': 0.15815, 'valueI': 0.0, 'valueJ': 0.02506, 'valueK': 0.04052, 'valueL': 0.2672, 'valueM': 0.00467, 'valueN': 0.0704},
        {'valueA': 0.0, 'valueB': 0.41243, 'valueC': 1.0, 'valueD': 0.60409},
        {'valueA': 0.82629, 'valueB': 0.23571, 'valueC': 0.67316, 'valueD': 0.4413, 'valueE': 1.0, 'valueF': 0.78092, 'valueG': 0.0, 'valueH': 0.06562, 'valueI': 0.10606, 'valueJ': 0.23355, 'valueK': 0.63195, 'valueL': 0.08596, 'valueM': 0.08873, 'valueN': 0.21722, 'valueO': 0.91881, 'valueP': 0.31936, 'valueQ': 0.46887, 'valueR': 0.0694, 'valueS': 0.01825},
        {'valueA': 0.14051, 'valueB': 0.13085, 'valueC': 1.0, 'valueD': 0.32794, 'valueE': 0.0},
    ],
    weights=[
        0.05104,
        0.26949,
        0.0535,
        0.62597,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.52406, 'valueB': 0.16431, 'valueC': 0.0, 'valueD': 0.2303, 'valueE': 0.07974, 'valueF': 0.05822, 'valueG': 0.05739, 'valueH': 1.0, 'valueI': 0.96605, 'valueJ': 0.13901, 'valueK': 0.16066, 'valueL': 0.02208, 'valueM': 0.32185, 'valueN': 0.65776},
        {'valueA': 0.15388, 'valueB': 0.06426, 'valueC': 0.0, 'valueD': 1.0},
        {'valueA': 0.06558, 'valueB': 1.0, 'valueC': 0.21509, 'valueD': 0.34885, 'valueE': 0.94748, 'valueF': 0.12333, 'valueG': 0.36746, 'valueH': 0.02104, 'valueI': 0.20432, 'valueJ': 0.10671, 'valueK': 0.50965, 'valueL': 0.01098, 'valueM': 0.20716, 'valueN': 0.07107, 'valueO': 0.0, 'valueP': 0.47463, 'valueQ': 0.01789, 'valueR': 0.03858, 'valueS': 0.09466},
        {'valueA': 0.0, 'valueB': 0.56146, 'valueC': 0.04442, 'valueD': 1.0, 'valueE': 0.04942},
    ],
    weights=[
        0.55884,
        0.04874,
        0.18405,
        0.20837,
    ],
    issues=issues,
    reserved_value=0.4,
)