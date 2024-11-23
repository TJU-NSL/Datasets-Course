# Description

Consider the problem of [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication). A matrix `A` of shape `(n, m)` may be multiplied by a matrix `B` of shape `(m, p)` to produce a new matrix `A * B` of shape `(n, p)`. The number of scalar multiplications required to perform this operation is `n * m * p`.

Now consider the problem of multiplying three or more matrices together. It turns out that matrix multiplication is associative, but the number of scalar multiplications required to perform such operation depends on the association.

For example, consider three matrices of the following shapes:

    A: (3, 5)
    B: (5, 7)
    C: (7, 9)

The matrix multiplication `(A * B) * C` would require 294 scalar multiplications while the matrix multiplication `A * (B * C)` would require 450 scalar multiplications.

The challenge is to find the optimal order for a chain of matrix multiplications given the shapes of the matrices.

# Formal Inputs and Outputs

Your program should accept as input the number of matrices followed by their shapes. For the example above, the inputs would be:

    3
    3 5
    5 7
    7 9

Your program should output the optimal number of scalar multiplications and the association tree, where the leaves of the tree are the indices of the matrices. So for the example above, the outputs would be:

    294
    ((0, 1), 2)

where `0` refers to the matrix `A`, `1` refers to `B` and `2` refers to `C`. Note that matrix multiplication is not commutative, so the leaves of the tree will always be in order.

# Challenge Inputs

## Challenge 1:

    4
    14 14
    14 2
    2 4
    4 5

## Challenge 2:

    8
    9 16
    16 4
    4 1
    1 7
    7 2
    2 11
    11 4
    4 16

## Challenge 3:

    16
    12 11
    11 6
    6 2
    2 10
    10 13
    13 11
    11 7
    7 8
    8 13
    13 3
    3 10
    10 4
    4 8
    8 3
    3 5
    5 8

# Bonus

An optimizer is no good if it takes longer than the solution it finds. Simply trying all combinations requires a runtime of O(2^(n)). A dynamic programming solution exists with a runtime of O(n^(3)), and the best known algorithm has a runtime cost of O(n * log(n)). Can you find these optimized solutions?

The following link contains additional test cases for 32, 64, and 128 matrices: https://gist.github.com/cbarrick/ce623ce2904fd1921a0da7aac3328b37

# Hints

This is a classic problem taught in most university level algorithms courses. Mosts textbooks covering dynamic programming will discuss this problem. It even has [its own Wikipedia page](https://en.wikipedia.org/wiki/Matrix_chain_multiplication).

# Finally

Have a good challenge idea?

Consider submitting it to /r/dailyprogrammer_ideas