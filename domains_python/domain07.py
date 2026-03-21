from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain07

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.07067, 'valueB': 0.69724, 'valueC': 0.0, 'valueD': 0.58011, 'valueE': 1.0, 'valueF': 0.52364, 'valueG': 0.85131, 'valueH': 0.46694},
        {'valueA': 0.0991, 'valueB': 0.21485, 'valueC': 0.31653, 'valueD': 0.37147, 'valueE': 0.26541, 'valueF': 0.15153, 'valueG': 0.01723, 'valueH': 0.06419, 'valueI': 0.74081, 'valueJ': 0.4922, 'valueK': 0.05045, 'valueL': 1.0, 'valueM': 0.17293, 'valueN': 0.41875, 'valueO': 0.0, 'valueP': 0.10003, 'valueQ': 0.33442, 'valueR': 0.45455, 'valueS': 0.17222},
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.21506},
        {'valueA': 0.24756, 'valueB': 1.0, 'valueC': 0.25546, 'valueD': 0.54722, 'valueE': 0.75069, 'valueF': 0.28437, 'valueG': 0.66354, 'valueH': 0.22916, 'valueI': 0.46727, 'valueJ': 0.0, 'valueK': 0.19227, 'valueL': 0.42899, 'valueM': 0.36145, 'valueN': 0.20243, 'valueO': 0.52352, 'valueP': 0.81093, 'valueQ': 0.28472, 'valueR': 0.16345, 'valueS': 0.98133},
    ],
    weights=[
        0.05657,
        0.11974,
        0.41011,
        0.41358,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.212, 'valueB': 0.48783, 'valueC': 0.39124, 'valueD': 0.3428, 'valueE': 1.0, 'valueF': 0.21703, 'valueG': 0.0, 'valueH': 0.29413},
        {'valueA': 0.24759, 'valueB': 0.07723, 'valueC': 0.07229, 'valueD': 0.58655, 'valueE': 0.0, 'valueF': 0.03515, 'valueG': 0.1837, 'valueH': 0.28184, 'valueI': 0.24103, 'valueJ': 0.49767, 'valueK': 0.4361, 'valueL': 1.0, 'valueM': 0.89353, 'valueN': 0.64212, 'valueO': 0.66923, 'valueP': 0.33909, 'valueQ': 0.00868, 'valueR': 0.42445, 'valueS': 0.20312},
        {'valueA': 1.0, 'valueB': 0.0985, 'valueC': 0.0},
        {'valueA': 1.0, 'valueB': 0.38823, 'valueC': 0.43454, 'valueD': 0.36439, 'valueE': 0.72117, 'valueF': 0.05367, 'valueG': 0.14822, 'valueH': 0.0, 'valueI': 0.04999, 'valueJ': 0.27249, 'valueK': 0.01498, 'valueL': 0.48046, 'valueM': 0.10702, 'valueN': 0.61758, 'valueO': 0.07853, 'valueP': 0.51114, 'valueQ': 0.53607, 'valueR': 0.34005, 'valueS': 0.0204},
    ],
    weights=[
        0.01263,
        0.04728,
        0.55963,
        0.38046,
    ],
    issues=issues,
    reserved_value=0.4,
)