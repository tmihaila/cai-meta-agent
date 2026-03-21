from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain37

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW', 'valueX', 'valueY', 'valueZ']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.10454, 'valueB': 0.05182, 'valueC': 0.21114, 'valueD': 0.39937, 'valueE': 0.0, 'valueF': 0.37501, 'valueG': 0.18154, 'valueH': 0.5209, 'valueI': 0.29592, 'valueJ': 0.36763, 'valueK': 1.0},
        {'valueA': 0.34798, 'valueB': 0.33701, 'valueC': 0.38698, 'valueD': 1.0, 'valueE': 0.15778, 'valueF': 0.0732, 'valueG': 0.09562, 'valueH': 0.48984, 'valueI': 0.19694, 'valueJ': 0.48415, 'valueK': 0.1772, 'valueL': 0.11293, 'valueM': 0.03534, 'valueN': 0.20994, 'valueO': 0.581, 'valueP': 0.52713, 'valueQ': 0.37146, 'valueR': 0.63795, 'valueS': 0.04736, 'valueT': 0.22253, 'valueU': 0.63487, 'valueV': 0.19247, 'valueW': 0.0, 'valueX': 0.01949, 'valueY': 0.12934, 'valueZ': 0.65087},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.0, 'valueB': 0.03391, 'valueC': 1.0},
    ],
    weights=[
        0.10611,
        0.17355,
        0.59655,
        0.12379,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.25151, 'valueB': 0.08739, 'valueC': 0.0, 'valueD': 0.63984, 'valueE': 0.18518, 'valueF': 1.0, 'valueG': 0.50877, 'valueH': 0.50786, 'valueI': 0.08345, 'valueJ': 0.46895, 'valueK': 0.018},
        {'valueA': 0.26887, 'valueB': 0.57829, 'valueC': 0.31419, 'valueD': 0.4227, 'valueE': 0.18496, 'valueF': 0.15188, 'valueG': 0.76539, 'valueH': 0.3858, 'valueI': 0.2486, 'valueJ': 0.04198, 'valueK': 0.10456, 'valueL': 1.0, 'valueM': 0.38906, 'valueN': 0.03715, 'valueO': 0.02592, 'valueP': 0.16073, 'valueQ': 0.05146, 'valueR': 0.12571, 'valueS': 0.04092, 'valueT': 0.24565, 'valueU': 0.30502, 'valueV': 0.0, 'valueW': 0.04437, 'valueX': 0.01041, 'valueY': 0.16299, 'valueZ': 0.19889},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.128, 'valueB': 0.0, 'valueC': 1.0},
    ],
    weights=[
        0.12265,
        0.00956,
        0.78275,
        0.08504,
    ],
    issues=issues,
    reserved_value=0.4,
)