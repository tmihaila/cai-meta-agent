from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain49

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV', 'valueW']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH']),
    make_issue(name="issueC", values=['valueA', 'valueB']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ']),
    make_issue(name="issueE", values=['valueA', 'valueB']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 1.0, 'valueB': 0.65352, 'valueC': 0.16534, 'valueD': 0.26536, 'valueE': 0.09303, 'valueF': 0.30392, 'valueG': 0.07063, 'valueH': 0.00106, 'valueI': 0.37624, 'valueJ': 0.38004, 'valueK': 0.15099, 'valueL': 0.38291, 'valueM': 0.06557, 'valueN': 0.26973, 'valueO': 0.13814, 'valueP': 0.06251, 'valueQ': 0.77375, 'valueR': 0.10451, 'valueS': 0.16085, 'valueT': 0.00118, 'valueU': 0.68528, 'valueV': 0.09839, 'valueW': 0.0},
        {'valueA': 0.10957, 'valueB': 1.0, 'valueC': 0.09544, 'valueD': 0.21529, 'valueE': 0.02869, 'valueF': 0.17817, 'valueG': 0.00312, 'valueH': 0.0},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.14746, 'valueB': 0.03607, 'valueC': 0.71197, 'valueD': 0.41429, 'valueE': 0.3389, 'valueF': 0.03406, 'valueG': 0.28904, 'valueH': 0.0, 'valueI': 0.02246, 'valueJ': 1.0},
        {'valueA': 1.0, 'valueB': 0.0},
    ],
    weights=[
        0.28446,
        0.66544,
        0.0344,
        0.00288,
        0.01282,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 0.75188, 'valueC': 0.09666, 'valueD': 0.3205, 'valueE': 0.24115, 'valueF': 0.32021, 'valueG': 1.0, 'valueH': 0.15731, 'valueI': 0.10876, 'valueJ': 0.05688, 'valueK': 0.10775, 'valueL': 0.26702, 'valueM': 0.57246, 'valueN': 0.12, 'valueO': 0.04702, 'valueP': 0.07905, 'valueQ': 0.96644, 'valueR': 0.41695, 'valueS': 0.02036, 'valueT': 0.16289, 'valueU': 0.57166, 'valueV': 0.0284, 'valueW': 0.80536},
        {'valueA': 0.12284, 'valueB': 0.0, 'valueC': 0.37867, 'valueD': 0.95547, 'valueE': 0.34226, 'valueF': 1.0, 'valueG': 0.69073, 'valueH': 0.00035},
        {'valueA': 0.0, 'valueB': 1.0},
        {'valueA': 0.07811, 'valueB': 0.47397, 'valueC': 0.13064, 'valueD': 0.5668, 'valueE': 0.37947, 'valueF': 0.0, 'valueG': 0.37757, 'valueH': 1.0, 'valueI': 0.00712, 'valueJ': 0.04528},
        {'valueA': 0.0, 'valueB': 1.0},
    ],
    weights=[
        0.27233,
        0.22686,
        0.27781,
        0.09819,
        0.12481,
    ],
    issues=issues,
    reserved_value=0.4,
)