from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain40

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueB", values=['valueA', 'valueB']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN']),
    make_issue(name="issueE", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueF", values=['valueA', 'valueB', 'valueC', 'valueD']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.0, 'valueC': 0.4772},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 1.0, 'valueB': 0.03214, 'valueC': 0.0},
        {'valueA': 1.0, 'valueB': 0.36196, 'valueC': 0.17657, 'valueD': 0.3738, 'valueE': 0.48411, 'valueF': 0.0, 'valueG': 0.35826, 'valueH': 0.01515, 'valueI': 0.50155, 'valueJ': 0.0991, 'valueK': 0.70916, 'valueL': 0.22968, 'valueM': 0.2173, 'valueN': 0.30364},
        {'valueA': 1.0, 'valueB': 0.08364, 'valueC': 0.66272, 'valueD': 0.30533, 'valueE': 0.76752, 'valueF': 0.0},
        {'valueA': 1.0, 'valueB': 0.1199, 'valueC': 0.24073, 'valueD': 0.0},
    ],
    weights=[
        0.10902,
        0.10464,
        0.14694,
        0.48433,
        0.0047,
        0.15037,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.13337, 'valueB': 0.0, 'valueC': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
        {'valueA': 0.0, 'valueB': 0.89844, 'valueC': 1.0},
        {'valueA': 0.27906, 'valueB': 0.06551, 'valueC': 0.14605, 'valueD': 0.03649, 'valueE': 1.0, 'valueF': 0.01368, 'valueG': 0.66118, 'valueH': 0.19742, 'valueI': 0.4703, 'valueJ': 0.70614, 'valueK': 0.29137, 'valueL': 0.09395, 'valueM': 0.51641, 'valueN': 0.0},
        {'valueA': 0.53042, 'valueB': 1.0, 'valueC': 0.42319, 'valueD': 0.88902, 'valueE': 0.0, 'valueF': 0.59777},
        {'valueA': 0.0, 'valueB': 0.01913, 'valueC': 1.0, 'valueD': 0.04211},
    ],
    weights=[
        0.01458,
        0.10561,
        0.13448,
        0.21122,
        0.0428,
        0.49131,
    ],
    issues=issues,
    reserved_value=0.4,
)