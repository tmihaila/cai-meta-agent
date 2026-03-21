from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain42

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.1021, 'valueC': 0.0062, 'valueD': 0.05033, 'valueE': 0.16967, 'valueF': 0.06514, 'valueG': 0.32598, 'valueH': 0.1418, 'valueI': 0.14232, 'valueJ': 0.0, 'valueK': 0.00717, 'valueL': 0.13128, 'valueM': 0.23047, 'valueN': 0.03042, 'valueO': 0.14778, 'valueP': 0.15697, 'valueQ': 0.00359},
        {'valueA': 0.58732, 'valueB': 0.0, 'valueC': 1.0, 'valueD': 0.21814, 'valueE': 0.28711},
        {'valueA': 0.0, 'valueB': 0.1022, 'valueC': 0.23722, 'valueD': 1.0, 'valueE': 0.02279},
        {'valueA': 0.58632, 'valueB': 0.34203, 'valueC': 0.56616, 'valueD': 1.0, 'valueE': 0.0, 'valueF': 0.41543},
    ],
    weights=[
        0.37854,
        0.40557,
        0.18707,
        0.02882,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.26097, 'valueB': 0.69277, 'valueC': 0.32337, 'valueD': 0.48067, 'valueE': 0.24309, 'valueF': 0.0377, 'valueG': 1.0, 'valueH': 0.07345, 'valueI': 0.95154, 'valueJ': 0.24808, 'valueK': 0.497, 'valueL': 0.0, 'valueM': 0.13597, 'valueN': 0.24856, 'valueO': 0.42087, 'valueP': 0.01811, 'valueQ': 0.12338},
        {'valueA': 0.44323, 'valueB': 1.0, 'valueC': 0.25606, 'valueD': 0.40493, 'valueE': 0.0},
        {'valueA': 0.00921, 'valueB': 0.5062, 'valueC': 0.0, 'valueD': 0.26616, 'valueE': 1.0},
        {'valueA': 0.36391, 'valueB': 0.0, 'valueC': 0.52199, 'valueD': 0.4266, 'valueE': 0.05733, 'valueF': 1.0},
    ],
    weights=[
        0.33803,
        0.00421,
        0.08201,
        0.57575,
    ],
    issues=issues,
    reserved_value=0.4,
)