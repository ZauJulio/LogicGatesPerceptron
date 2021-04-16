# Zaú Júlio A. Galvão

import pandas as pd


def f(y):
    if y >= 0:
        return 1
    elif y < 0:
        return 0


def getTable(resultExpected):
    return [
        [0, 0, resultExpected[0], None],
        [0, 1, resultExpected[1], None],
        [1, 0, resultExpected[2], None],
        [1, 1, resultExpected[3], None]
    ]


def apply(table, weights):
    """  """
    table[0][-1] = f(0 + 0 + 1 * weights[0])
    table[1][-1] = f(0 + 1 * weights[2] + 1 * weights[0])
    table[2][-1] = f(1 * weights[1] + 0 * weights[2] + 1 * weights[0])
    table[3][-1] = f(1 * weights[1] + 1 * weights[2] + 1 * weights[0])

    return table


def logicGatePerceptronValidate(resultExpected, weights):
    return apply(getTable(resultExpected), weights)


_or = logicGatePerceptronValidate(
    resultExpected=[0, 1, 1, 1], weights=[-1, 1, 1])
_and = logicGatePerceptronValidate(
    resultExpected=[0, 0, 0, 1], weights=[-2, 1, 1])
_nand = logicGatePerceptronValidate(
    resultExpected=[1, 0, 0, 0], weights=[0, -1, -1])
_nor = logicGatePerceptronValidate(
    resultExpected=[1, 1, 1, 0], weights=[2, -1, -2])
_xor = logicGatePerceptronValidate(
    resultExpected=[0, 1, 1, 0], weights=[-1, 1, -2])

index = ['', '', '', '']

print('OR\n', pd.DataFrame(_or, index, ['x1', 'x2', '(x1 v x2)', 'f(y)']))
print('AND\n', pd.DataFrame(_and, index, ['x1', 'x2', '(x1 ^ x2)', 'f(y)']))
print('NOR\n', pd.DataFrame(_nor, index, ['x1', 'x2', '(x1 /v x2)', 'f(y)']))
print('NAND\n', pd.DataFrame(_nand, index, ['x1', 'x2', '(x1 /^ x2)', 'f(y)']))
print('XOR\n', pd.DataFrame(_xor, index, ['x1', 'x2', '(x1 o x2)', 'f(y)']))
print("Um perceptron não é capaz de aprender a porta XOR.")
