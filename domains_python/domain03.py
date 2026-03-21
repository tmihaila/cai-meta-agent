from negmas import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


# Domain: domain03

issues = [
    make_issue(name="issueA", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK']),
    make_issue(name="issueB", values=['valueA', 'valueB', 'valueC', 'valueD']),
    make_issue(name="issueC", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF']),
    make_issue(name="issueD", values=['valueA', 'valueB', 'valueC', 'valueD', 'valueE', 'valueF', 'valueG', 'valueH', 'valueI', 'valueJ', 'valueK', 'valueL', 'valueM', 'valueN', 'valueO', 'valueP', 'valueQ', 'valueR', 'valueS', 'valueT', 'valueU', 'valueV']),
]

# Agent A utility function
ufun_a = LUFun(
    values=[
        {'valueA': 0.0, 'valueB': 1.0, 'valueC': 0.28544, 'valueD': 0.07689, 'valueE': 0.26491, 'valueF': 0.04585, 'valueG': 0.12702, 'valueH': 0.12523, 'valueI': 0.18351, 'valueJ': 0.37782, 'valueK': 0.46439},
        {'valueA': 0.20168, 'valueB': 1.0, 'valueC': 0.30849, 'valueD': 0.0},
        {'valueA': 0.48165, 'valueB': 0.67742, 'valueC': 0.32659, 'valueD': 0.0, 'valueE': 1.0, 'valueF': 0.57713},
        {'valueA': 0.80324, 'valueB': 0.02095, 'valueC': 0.0067, 'valueD': 0.2116, 'valueE': 0.89275, 'valueF': 0.47904, 'valueG': 0.86322, 'valueH': 0.10834, 'valueI': 0.03045, 'valueJ': 0.38155, 'valueK': 0.63651, 'valueL': 0.20609, 'valueM': 0.24512, 'valueN': 0.0, 'valueO': 0.58527, 'valueP': 0.03393, 'valueQ': 1.0, 'valueR': 0.19658, 'valueS': 0.68963, 'valueT': 0.25004, 'valueU': 0.30103, 'valueV': 0.35296},
    ],
    weights=[
        0.27463,
        0.3612,
        0.23751,
        0.12666,
    ],
    issues=issues,
    reserved_value=0.4,
)

# Agent B utility function
ufun_b = LUFun(
    values=[
        {'valueA': 0.03588, 'valueB': 0.15608, 'valueC': 0.09708, 'valueD': 1.0, 'valueE': 0.0236, 'valueF': 0.00254, 'valueG': 0.41844, 'valueH': 0.47585, 'valueI': 0.03676, 'valueJ': 0.14055, 'valueK': 0.0},
        {'valueA': 1.0, 'valueB': 0.14713, 'valueC': 0.36439, 'valueD': 0.0},
        {'valueA': 0.15361, 'valueB': 0.19639, 'valueC': 1.0, 'valueD': 0.0, 'valueE': 0.67864, 'valueF': 0.48994},
        {'valueA': 0.33523, 'valueB': 1.0, 'valueC': 0.00963, 'valueD': 0.27031, 'valueE': 0.41771, 'valueF': 0.27574, 'valueG': 0.42579, 'valueH': 0.22154, 'valueI': 0.02387, 'valueJ': 0.2273, 'valueK': 0.17127, 'valueL': 0.48053, 'valueM': 0.18959, 'valueN': 0.49769, 'valueO': 0.0, 'valueP': 0.07237, 'valueQ': 0.21102, 'valueR': 0.06796, 'valueS': 0.50685, 'valueT': 0.00278, 'valueU': 0.52794, 'valueV': 0.26237},
    ],
    weights=[
        0.44395,
        0.54707,
        0.00585,
        0.00313,
    ],
    issues=issues,
    reserved_value=0.4,
)