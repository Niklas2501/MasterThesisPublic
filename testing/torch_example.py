import numpy as np
import torch

# Sources:
# https://graphneural.network/transforms/#gcnfilter
# https://stackoverflow.com/questions/32381299/python-computing-vertex-degree-matrix
# http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture2/lecture2.html
# Aji !!! http://www.pitt.edu/~kpele/Materials15/module1.pdf S.28, 10
# Gegenbeispiel: https://www.researchgate.net/publication/220141972_Network_science

myA = torch.Tensor([
    # A, B, C, D
    [1, 1, 0, 0],  # A
    [1, 0, 0, 0],  # B
    [1, 0, 1, 0],  # C
    [0, 0, 1, 0]  # D
])

A = torch.Tensor([[1, 1, 1, 1],
                  [1, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 1]])

# A = myA

# Wieso ist Spaltensumme Ausgangsgrade, wenn e_ij = Kante von i nach j?
# = Wieso ist Row-Sum Eingangsgrad, wenn eij = Kante von i nach j?
out_degree = torch.sum(A, dim=0)
in_degree = torch.sum(A, dim=1)
identity = torch.eye(A.size()[0])

print(A)
print()
print('out_degree = Summe über Spalten')
print(identity*out_degree)
print()
print('in_degree = Summe über Reihen')
print(identity*in_degree)
