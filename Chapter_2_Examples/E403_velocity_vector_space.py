import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)
#! /usr/bin/python3
import numpy as np
from S400_Construct_R2 import R2

# Make things print nicely
np.set_printoptions(precision=2)

# Take the manifold as R2 with a differentiable structure
Q = R2

# Construct an element of R2
q1 = Q.element([2, 0], 0)

# Construct two vectors at q1; note the one-dimensional structure
v11 = Q.vector(q1, [0, 1])
v12 = Q.vector(q1, [1, 1])

# Test summing the vectors
v1sum = v11 + v12

print("Sum of v11=", v11, "\n and v12=", v12, " is \n", v1sum)

# Test scaling the vectors (we assume that scalar multiplication of real numbers works)
v11scaled = 5 * v11
v12scaled = v12 * -2

print("5 * v11 is\n", v11scaled, "\n and v12 * -2 is\n", v12scaled)

# Construct a second element of R2
q2 = Q.element([1, 1], 0)

# Construct a vector at this point
v21 = Q.vector(q2, [1, 1])
v22 = Q.vector(q2, [1, 1])

# Verify that adding v21 and v22 works, but adding v11 and v21 does not work
v2sum = v21 + v22

print("\n\nSum of v21=", v21, "\n and v22=", v22, " is \n", v2sum)

# Verify that attempting to add vectors at different configurations produces an error
try:
    v1v2sum = v11 + v21
except Exception as error:
    print("\n\n Attempted to add v11 and v21, but TangentVector class prevented this with error: \n",
          error)

