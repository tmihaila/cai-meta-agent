from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain17

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 0.05886, 'valueC': 0.69815, 'valueD': 1.0},
        {'valueA': 0.1824, 'valueB': 0.31712, 'valueC': 0.22385, 'valueD': 0.0325, 'valueE': 0.07433, 'valueF': 0.80076, 'valueG': 0.52879, 'valueH': 0.03723, 'valueI': 0.09407, 'valueJ': 0.20943, 'valueK': 0.07336, 'valueL': 1.0, 'valueM': 0.19033, 'valueN': 0.0, 'valueO': 0.08521, 'valueP': 0.21706},
        {'valueA': 0.17403, 'valueB': 0.22166, 'valueC': 0.13462, 'valueD': 0.12237, 'valueE': 0.15157, 'valueF': 0.10638, 'valueG': 0.50804, 'valueH': 0.2636, 'valueI': 0.03335, 'valueJ': 0.41513, 'valueK': 0.2561, 'valueL': 1.0, 'valueM': 0.30554, 'valueN': 0.02236, 'valueO': 0.319, 'valueP': 0.15902, 'valueQ': 0.00171, 'valueR': 0.0, 'valueS': 0.05287},
    ],
    weights=[
        0.10064,
        0.36002,
        0.27265,
        0.26669,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.05176, 'valueB': 0.0, 'valueC': 1.0, 'valueD': 0.01881},
        {'valueA': 0.51805, 'valueB': 0.00375, 'valueC': 0.00869, 'valueD': 0.36327, 'valueE': 0.32192, 'valueF': 1.0, 'valueG': 0.57956, 'valueH': 0.0742, 'valueI': 0.27681, 'valueJ': 0.23185, 'valueK': 0.23472, 'valueL': 0.57057, 'valueM': 0.00652, 'valueN': 0.0, 'valueO': 0.1776, 'valueP': 0.37399},
        {'valueA': 0.27452, 'valueB': 0.81892, 'valueC': 0.21348, 'valueD': 0.22506, 'valueE': 0.26906, 'valueF': 0.13034, 'valueG': 0.27875, 'valueH': 0.11901, 'valueI': 1.0, 'valueJ': 0.04746, 'valueK': 0.11452, 'valueL': 0.0, 'valueM': 0.12707, 'valueN': 0.16693, 'valueO': 0.37276, 'valueP': 0.14147, 'valueQ': 0.07313, 'valueR': 0.14172, 'valueS': 0.18842},
    ],
    weights=[
        0.27669,
        0.0172,
        0.11471,
        0.5914,
    ],
    issues=issues,
    reserved_value=0.4,
)