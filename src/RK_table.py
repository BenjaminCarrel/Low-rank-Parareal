#%% IMPORTATIONS
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike

# %% RUNGE--KUTTA TABLE
# ORDER 1
a1 = 0
b1 = [1]
# ORDER 2
a2 = np.zeros((2, 2))
a2[1, 0] = 1
b2 = np.zeros(2)
b2[0] = 1/2
b2[1] = 1/2
# ORDER 3
a3 = np.zeros((3, 3))
a3[1, 0] = 1/3
a3[2, 0] = 0
a3[2, 1] = 2/3
b3 = np.zeros(3)
b3[0] = 1/4
b3[2] = 3/4
# ORDER 4
a4 = np.zeros((4, 4))
a4[1, 0] = 1/2
a4[2, 1] = 1/2
a4[3, 2] = 1
b4 = np.zeros(4)
b4[0] = 1/6
b4[1] = 1/3
b4[2] = 1/3
b4[3] = 1/6
# Rule 6(5)9b
a8 = np.zeros((8, 8))
a8[1, 0] = 1/8
a8[2, 0] = 1/18
a8[3, 0] = 1/16
a8[4, 0] = 1/4
a8[5, 0] = 134/625
a8[6, 0] = -98/1875
a8[7, 0] = 9/50
a8[2, 1] = 1/9
a8[3, 2] = 3/16
a8[4, 2] = -3/4
a8[5, 2] = -333/625
a8[6, 2] = 12/625
a8[7, 2] = 21/25
a8[4, 3] = 1
a8[5, 3] = 476/625
a8[6, 3] = 10736/13125
a8[7, 3] = -2924/1925
a8[5, 4] = 98/625
a8[6, 4] = -1936/1875
a8[7, 4] = 74/25
a8[6, 5] = 22/21
a8[7, 5] = -15/7
a8[7, 6] = 15/22
b8 = np.zeros(8)
b8[0] = 11/144
b8[3] = 256/693
b8[5] = 125/504
b8[6] = 125/528
b8[7] = 5/72

#%% SHORTCUT
def RK_rule(s: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Shortcut for calling the table

    Args:
        s (int): number of stages

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike]: coefficient of the corresponding table
    """
    if s == 1:
        a = a1
        b = b1
    elif s == 2:
        a = a2
        b = b2
    if s == 3:
        a = a3
        b = b3
    if s == 4:
        a = a4
        b = b4
    if s == 8:
        a = a8
        b = b8
    c = np.zeros(s)
    for i in np.arange(1, s):
        c[i] = sum(a[i, j] for j in np.arange(1, i))
    return a, b, c