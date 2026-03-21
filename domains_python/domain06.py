from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain06

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.00414, 'valueB': 0.0, 'valueC': 0.10109, 'valueD': 1.0},
        {'valueA': 0.46005, 'valueB': 1.0, 'valueC': 0.0},
        {'valueA': 0.38672, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.56248, 'valueE': 0.41025, 'valueF': 0.20301, 'valueG': 0.48511, 'valueH': 0.63818},
        {'valueA': 1.0, 'valueB': 0.83307, 'valueC': 0.0},
        {'valueA': 0.0, 'valueB': 0.05998, 'valueC': 0.22892, 'valueD': 1.0, 'valueE': 0.6125, 'valueF': 0.11869, 'valueG': 0.13828, 'valueH': 0.72864, 'valueI': 0.17041, 'valueJ': 0.04768, 'valueK': 0.28565, 'valueL': 0.08938, 'valueM': 0.09275, 'valueN': 0.27616, 'valueO': 0.02035, 'valueP': 0.10118},
    ],
    weights=[
        0.02371,
        0.42255,
        0.33802,
        0.12867,
        0.08705,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.63689, 'valueB': 1.0, 'valueC': 0.0, 'valueD': 0.1911},
        {'valueA': 0.16634, 'valueB': 1.0, 'valueC': 0.0},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.24534, 'valueD': 0.88199, 'valueE': 0.05626, 'valueF': 0.29584, 'valueG': 0.30869, 'valueH': 0.34929},
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.22237},
        {'valueA': 0.05275, 'valueB': 1.0, 'valueC': 0.05181, 'valueD': 0.16926, 'valueE': 0.0804, 'valueF': 0.19521, 'valueG': 0.0, 'valueH': 0.17893, 'valueI': 0.09377, 'valueJ': 0.17194, 'valueK': 0.75531, 'valueL': 0.16439, 'valueM': 0.4275, 'valueN': 0.03727, 'valueO': 0.14396, 'valueP': 0.11356},
    ],
    weights=[
        0.1436,
        0.2414,
        0.00286,
        0.1359,
        0.47624,
    ],
    issues=issues,
    reserved_value=0.4,
)