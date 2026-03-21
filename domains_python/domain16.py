from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain16

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.01248, 'valueC': 0.0},
        {'valueA': 0.20461, 'valueB': 1.0, 'valueC': 0.4364, 'valueD': 0.16083, 'valueE': 0.044, 'valueF': 0.03879, 'valueG': 0.0},
        {'valueA': 0.09975, 'valueB': 0.3021, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.21304, 'valueF': 0.41226, 'valueG': 0.21217},
        {'valueA': 0.05709, 'valueB': 0.51279, 'valueC': 0.09716, 'valueD': 0.20329, 'valueE': 0.13123, 'valueF': 0.16396, 'valueG': 0.0265, 'valueH': 0.4519, 'valueI': 0.0, 'valueJ': 0.27535, 'valueK': 0.56622, 'valueL': 0.12787, 'valueM': 0.27383, 'valueN': 0.24415, 'valueO': 0.00885, 'valueP': 0.1029, 'valueQ': 0.34607, 'valueR': 1.0, 'valueS': 0.16451, 'valueT': 0.25667, 'valueU': 0.40268, 'valueV': 0.24677, 'valueW': 0.33954},
    ],
    weights=[
        0.75236,
        0.05073,
        0.10824,
        0.08867,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.59517},
        {'valueA': 0.44977, 'valueB': 0.21894, 'valueC': 0.79384, 'valueD': 0.01592, 'valueE': 0.0, 'valueF': 1.0, 'valueG': 0.07482},
        {'valueA': 0.37192, 'valueB': 0.03683, 'valueC': 0.744, 'valueD': 0.0, 'valueE': 0.6169, 'valueF': 0.35169, 'valueG': 1.0},
        {'valueA': 0.33877, 'valueB': 0.03493, 'valueC': 0.17043, 'valueD': 0.01594, 'valueE': 0.6703, 'valueF': 0.02947, 'valueG': 0.81387, 'valueH': 0.34925, 'valueI': 0.15741, 'valueJ': 0.09122, 'valueK': 0.2532, 'valueL': 1.0, 'valueM': 0.02293, 'valueN': 0.15512, 'valueO': 0.0639, 'valueP': 0.12571, 'valueQ': 0.42154, 'valueR': 0.0, 'valueS': 0.008, 'valueT': 0.40077, 'valueU': 0.53112, 'valueV': 0.14883, 'valueW': 0.25377},
    ],
    weights=[
        0.19408,
        0.21858,
        0.3688,
        0.21854,
    ],
    issues=issues,
    reserved_value=0.4,
)