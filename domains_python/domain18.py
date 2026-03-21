from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain18

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 0.65979, 'valueC': 1.0, 'valueD': 0.22504, 'valueE': 0.30498},
        {'valueA': 0.10843, 'valueB': 0.62503, 'valueC': 0.02565, 'valueD': 0.72717, 'valueE': 0.17562, 'valueF': 0.17775, 'valueG': 0.0, 'valueH': 0.10485, 'valueI': 0.31813, 'valueJ': 0.12886, 'valueK': 0.62571, 'valueL': 0.16797, 'valueM': 0.0818, 'valueN': 0.0302, 'valueO': 0.98276, 'valueP': 0.27737, 'valueQ': 0.01026, 'valueR': 0.16168, 'valueS': 0.54826, 'valueT': 0.8813, 'valueU': 1.0, 'valueV': 0.90763, 'valueW': 0.21279, 'valueX': 0.40884, 'valueY': 0.09594, 'valueZ': 0.23661},
        {'valueA': 0.23784, 'valueB': 0.31687, 'valueC': 0.54581, 'valueD': 0.06782, 'valueE': 0.66047, 'valueF': 0.77958, 'valueG': 1.0, 'valueH': 0.32979, 'valueI': 0.15232, 'valueJ': 0.18038, 'valueK': 0.31782, 'valueL': 0.27755, 'valueM': 0.88629, 'valueN': 0.00012, 'valueO': 0.11949, 'valueP': 0.17224, 'valueQ': 0.0544, 'valueR': 0.18967, 'valueS': 0.0},
    ],
    weights=[
        0.77138,
        0.00888,
        0.09704,
        0.1227,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.41252, 'valueB': 0.12945, 'valueC': 0.0, 'valueD': 1.0, 'valueE': 0.41454},
        {'valueA': 0.72927, 'valueB': 0.00461, 'valueC': 0.83768, 'valueD': 0.20707, 'valueE': 0.08897, 'valueF': 0.99059, 'valueG': 0.2483, 'valueH': 0.7431, 'valueI': 0.74617, 'valueJ': 0.40853, 'valueK': 0.39578, 'valueL': 0.48919, 'valueM': 0.0, 'valueN': 0.32055, 'valueO': 0.07658, 'valueP': 0.36947, 'valueQ': 0.14006, 'valueR': 0.03327, 'valueS': 0.17515, 'valueT': 0.0368, 'valueU': 0.02495, 'valueV': 0.0424, 'valueW': 1.0, 'valueX': 0.50447, 'valueY': 0.2294, 'valueZ': 0.04114},
        {'valueA': 0.12445, 'valueB': 0.39663, 'valueC': 0.36662, 'valueD': 0.72435, 'valueE': 0.50022, 'valueF': 0.08107, 'valueG': 0.12767, 'valueH': 0.0, 'valueI': 0.15251, 'valueJ': 0.21039, 'valueK': 0.46258, 'valueL': 0.2015, 'valueM': 1.0, 'valueN': 0.20597, 'valueO': 0.31196, 'valueP': 0.88481, 'valueQ': 0.04053, 'valueR': 0.09148, 'valueS': 0.10989},
    ],
    weights=[
        0.22719,
        0.3308,
        0.12048,
        0.32153,
    ],
    issues=issues,
    reserved_value=0.4,
)