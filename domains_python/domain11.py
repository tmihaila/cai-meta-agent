from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain11

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 0.35554, 'valueC': 0.12916, 'valueD': 0.70187, 'valueE': 1.0, 'valueF': 0.04463, 'valueG': 0.69275, 'valueH': 0.27293, 'valueI': 0.40618, 'valueJ': 0.06381},
        {'valueA': 0.00229, 'valueB': 0.09245, 'valueC': 0.12026, 'valueD': 0.00582, 'valueE': 0.01942, 'valueF': 0.08626, 'valueG': 0.00962, 'valueH': 0.20247, 'valueI': 0.48909, 'valueJ': 0.43328, 'valueK': 0.03849, 'valueL': 0.53024, 'valueM': 0.0566, 'valueN': 0.21554, 'valueO': 0.0, 'valueP': 0.99046, 'valueQ': 0.31761, 'valueR': 0.04326, 'valueS': 0.84256, 'valueT': 1.0, 'valueU': 0.62613, 'valueV': 0.84777, 'valueW': 0.40794, 'valueX': 0.70701, 'valueY': 0.45589, 'valueZ': 0.03355},
        {'valueA': 0.00908, 'valueB': 0.07121, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.10923, 'valueF': 0.00144, 'valueG': 0.08188, 'valueH': 0.03773, 'valueI': 0.33088, 'valueJ': 0.00642},
    ],
    weights=[
        0.16559,
        0.29484,
        0.17041,
        0.36916,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.203, 'valueC': 0.22431, 'valueD': 0.01234, 'valueE': 0.05953, 'valueF': 0.3139, 'valueG': 0.38042, 'valueH': 0.06185, 'valueI': 0.0, 'valueJ': 0.0379},
        {'valueA': 0.01771, 'valueB': 0.08309, 'valueC': 0.10713, 'valueD': 0.06408, 'valueE': 0.17388, 'valueF': 0.76142, 'valueG': 0.30811, 'valueH': 0.14958, 'valueI': 0.14128, 'valueJ': 0.12076, 'valueK': 0.03668, 'valueL': 0.05105, 'valueM': 0.07896, 'valueN': 1.0, 'valueO': 0.03363, 'valueP': 0.23405, 'valueQ': 0.03926, 'valueR': 0.10223, 'valueS': 0.13552, 'valueT': 0.0, 'valueU': 0.10042, 'valueV': 0.01698, 'valueW': 0.0058, 'valueX': 0.07006, 'valueY': 0.28798, 'valueZ': 0.10446},
        {'valueA': 1.0, 'valueB': 0.47614, 'valueC': 0.00146, 'valueD': 0.21, 'valueE': 0.00613, 'valueF': 0.03066, 'valueG': 0.0, 'valueH': 0.59683, 'valueI': 0.2386, 'valueJ': 0.00644},
    ],
    weights=[
        0.0444,
        0.18726,
        0.6754,
        0.09294,
    ],
    issues=issues,
    reserved_value=0.4,
)