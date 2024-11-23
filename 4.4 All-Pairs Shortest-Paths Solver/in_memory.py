__author__ = "Frank Schoeneman"
__copyright__ = "Copyright (c) 2019 SCoRe Group http://www.score-group.org/"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Frank Schoeneman"
__email__ = "fvschoen@buffalo.edu"
__status__ = "Development"

import numpy as np
import my_util

def run_APSP_diag_iteration(q, q_, p, rdd_partitioner, apsp_graph):

    def share_diag_block(x, q_):
        (blockId, ADJ_MAT) = x
        diagUpdate = my_util.scipy_floyd(ADJ_MAT)
        q_, q_ = blockId

        yield ((q_, q_), (-2, diagUpdate, 0))
        for j in range(q_ + 1, q, 1):
            yield ((q_, j), (-1, diagUpdate, 1))
        for i in range(0, q_, 1):
            yield ((i, q_), (-1, diagUpdate, -1))


    def updateAndshare(x, q_, q):
        ((I_, J_), xx) = x
        if isinstance(xx[0], np.ndarray):
            (InitBlock, (typeFlag, diagPass, matAlign)) = xx
        else:
            ((typeFlag, diagPass, matAlign), InitBlock) = xx

        if typeFlag == -2: submatrix = diagPass
        elif matAlign == 1: submatrix = my_util.minmpmatmul(diagPass, InitBlock, InitBlock)
        else:# matAlign == -1:
            submatrix = my_util.minmpmatmul(InitBlock, diagPass, InitBlock)

        yield ((I_, J_), submatrix) # save self - diag and row/col
        if I_ == q_ and J_ != q_:

            # RHS Updates
            for i in range(J_+1):
                if i != q_:
                    yield ((i, J_), (submatrix, 1))

            # LHS Updates
            for j in range(J_, q, 1):
                if j!= J_:
                    yield ((J_, j), (submatrix.T, -1))

                else:
                    yield ((J_, j), (submatrix.T, 2))

        elif J_ == q_ and I_ != q_:
            # RHS Updates
            for i in range(0, I_+1):
                yield ((i, I_), (submatrix.T, 1))
            # LHS Updates
            for j in [ii for ii in range(I_, q) if ii != J_]:
                if j!= I_:
                    yield ((I_, j), (submatrix, -1))
                else:
                    yield ((I_, j), (submatrix, -2))

    def UpdateMat(iter_, q_):
        for ((I_, J_), x) in iter_:
            if I_ == q_ or J_ == q_: # pass phase 1 and 2 blocks
                yield ((I_, J_), x[0])

            elif I_ == J_: # update other blocks on the diag
                if isinstance(x[0], np.ndarray): 
                    lastDiagBlock, (pMat1, pFlag1), (pMat2, pFlag2) = x
                elif isinstance(x[1], np.ndarray):
                    (pMat1, pFlag1), lastDiagBlock, (pMat2, pFlag2) = x
                else:
                    (pMat1, pFlag1), (pMat2, pFlag2), lastDiagBlock = x

                if pFlag1 == 1: yield((I_, J_), my_util.minmpmatmul(pMat2, pMat1, lastDiagBlock))
                else: yield((I_, J_), my_util.minmpmatmul(pMat2, pMat1, lastDiagBlock))

            else: # all other blocks
                if isinstance(x[0], np.ndarray):
                    lastBlock, (pMat1, pFlag1), (pMat2, pFlag2) = x
                elif isinstance(x[1], np.ndarray):
                    (pMat1, pFlag1), lastBlock, (pMat2, pFlag2) = x
                else:
                    (pMat1, pFlag1), (pMat2, pFlag2), lastBlock = x

                if pFlag1 == -1: yield((I_, J_), my_util.minmpmatmul(pMat1, pMat2, lastBlock))
                else: yield((I_, J_), my_util.minmpmatmul(pMat2, pMat1, lastBlock))


    diagBlock = apsp_graph.filter(lambda x : x[0][0] == q_ and x[0][1] == q_)\
                         .flatMap(lambda x : share_diag_block(x, q_),\
                          preservesPartitioning=False)\
                         .partitionBy(p, rdd_partitioner)
    # filter diagonal block (q,q), do seq fw, and send along row/col

    rowColBlocks = apsp_graph.filter(lambda x : x[0][0] == q_ or x[0][1] == q_)\
                            .union(diagBlock)\
                            .combineByKey((lambda x : [x]), \
                                          (lambda x, y : x + [y]), \
                                          (lambda x, y : x + y),\
                             numPartitions=p,\
                             partitionFunc=rdd_partitioner)\
                            .flatMap(lambda x : updateAndshare(x, q_, q),\
                             preservesPartitioning=False)\
                            .partitionBy(p, rdd_partitioner)
    # UNION (rowColBlocks and diagBlock) AS phase2 and update/compute

    p3Blocks = apsp_graph.filter(lambda x : x[0][0] != q_ and x[0][1] != q_)\
                        .union(rowColBlocks)\
                        .combineByKey((lambda x : [x]),\
                                      (lambda x, y : x + [y]),\
                                      (lambda x, y : x + y),\
                         numPartitions=p,\
                         partitionFunc=rdd_partitioner)\
                        .mapPartitions(lambda x : UpdateMat(x, q_),\
                         preservesPartitioning=False)\
                        .partitionBy(p, rdd_partitioner)
    # UNION (phase2 p3Blocks) AS phase3 and update/compute
    apsp_graph = p3Blocks

    return apsp_graph


def in_memory_block_fw(adj_mat, q, p, rdd_partitioner):

    apsp_graph = adj_mat
    for q_ in range(0, q, 1):
        apsp_graph = run_APSP_diag_iteration(q, q_, p, rdd_partitioner, apsp_graph)

    return apsp_graph
