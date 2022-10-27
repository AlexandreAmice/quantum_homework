import numpy as np
from numpy import kron
from scipy.linalg import block_diag

z = np.array([1,0])
o = np.array([0,1])
basis1 = (z,o)

plus = 1/np.sqrt(2) * (z+o)
minus = 1/np.sqrt(2) * (z-o)
pm_basis = (plus, minus)


zz = kron(z,z)
zo = kron(z,o)
oz = kron(o,z)
oo = kron(o,o)
basis2 = (zz,zo,oz,oo)


zzz = kron(z,zz)
zzo = kron(z,zo)
zoz = kron(z,oz)
zoo = kron(z,oo)
ozz = kron(o,zz)
ozo = kron(o,zo)
ooz = kron(o,oz)
ooo = kron(o,oo)
basis3 = (zzz,
          zzo,
          zoz,
          zoo,
          ozz,
          ozo,
          ooz,
          ooo)

bb1 = 1/np.sqrt(2)*(zz + oo)
bb2 = 1/np.sqrt(2)*(zz - oo)
bb3 = 1/np.sqrt(2)*(zo + oz)
bb4 = 1/np.sqrt(2)*(zo - oz)
bell_basis = (bb1, bb2, bb3, bb4)

I2 = np.eye(2)

H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])

sigma_x = np.array([
    [0,1],
    [1,0]
])

sigma_y = np.array([
    [0,-1j],
    [1j,0]
])

sigma_z = np.array([
    [1,0],
    [0,-1]
])


swap = np.array([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1]
])
cnot = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
])
cnot_rev = swap@cnot@swap

CX = block_diag(np.eye(2), sigma_x)
CNOT = CX
CY = block_diag(np.eye(2), sigma_y)
CZ = block_diag(np.eye(2), sigma_z)

SWAP_23 = kron(np.eye(2), swap)
SWAP_12 = kron(swap, np.eye(2))
SWAP_13 = SWAP_23 @ SWAP_12 @ SWAP_23


CNOT23 = kron(np.eye(2), cnot)
CNOT12 = kron(cnot, np.eye(2))
CNOT21 = kron(cnot_rev, np.eye(2))
CNOT13 = kron(swap, np.eye(2))@CNOT23@kron(swap,np.eye(2))
CNOT31 = SWAP_13@CNOT13@SWAP_13

ghz = 1/np.sqrt(2) * (zzz + ooo)

def compute_circuit(gate_list, ordered_left_to_right = True):
    '''
     compute the circuit by multiplying out the circuit in the gates list.
     If ordered left_to_right = True, then the first gate in the list is the first gate applied otherwise we reverse
    :return:
    '''
    if not ordered_left_to_right:
        gate_list = gate_list[::-1]
    out = np.eye(gate_list[0].shape[0])
    for g in gate_list:
        out = g @ out
    return out