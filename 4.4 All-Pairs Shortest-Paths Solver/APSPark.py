__author__ = "Frank Schoeneman"
__copyright__ = "Copyright (c) 2019 SCoRe Group http://www.score-group.org/"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Frank Schoeneman"
__email__ = "fvschoen@buffalo.edu"
__status__ = "Development"

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, 'SparkFiles.getRootDirectory()')

import my_util
import init_matrix
import in_memory
import collect_bc

from pyspark import SparkContext, SparkConf
from pyspark import StorageLevel as stglev


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_points", help="Number of points.", type=int, required=True)
    parser.add_argument("-b", "--block_size", help="Submatrix block size.", type=int, required=True)
    parser.add_argument("-p", "--partitions", help="Number of partitions.", type=int, required=True)
    parser.add_argument("-F", "--partitioner", help="Partitioning function. [md or ph]", type=str, required=True)
    parser.add_argument("-S", "--solver", help="Solver type. [im or cb]", type=str, required=True)

    parser.add_argument("-f", "--input", help="Input data. (.tsv format)", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output name. ", type=str, required=True)
    parser.add_argument("-e", "--log_dir", help="Spark event log dir.", type=str, required=False)

    args = parser.parse_args()
    n = args.num_points
    b = args.block_size
    p = args.partitions
    F = args.partitioner.lower()
    S = args.solver.lower()
    input_file = args.input
    out = args.output

    q, N = my_util.block_vars(n, b)
    rdd_partitioner = my_util.verify_partitioner(F, q)

    conf = SparkConf()

    # optional log for history server
    save_history = args.log_dir is not None
    if save_history:
        conf.set("spark.eventLog.enabled", "true")\
            .set("spark.eventLog.dir", args.log_dir)\
            .set("spark.history.fs.logDirectory", args.log_dir)

    sc = SparkContext(conf=conf)
    log4jLogger = sc._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger("APSPark")
    logger.setLevel(sc._jvm.org.apache.log4j.Level.ALL)
    logger.info('n: {}, b: {}, q: {}, p: {}, partitioner: {}'.format(n, b, q, p, F))

    # set-up matrix blocks
    block_matrix = init_matrix.initialize_blocks(n, b, q, p, rdd_partitioner, sc)
    block_matrix.persist(stglev.MEMORY_AND_DISK)
    block_matrix.count()

    # set edge weights from input_file
    adj_matrix = init_matrix.fill_blocks(b, input_file, block_matrix, p, rdd_partitioner, sc)
    adj_matrix.persist(stglev.MEMORY_AND_DISK)
    adj_matrix.count()

    t0 = time.time()

    if (S == "im"):
        # run apsp-solver Blocked-IM
        apsp_graph = in_memory.in_memory_block_fw(adj_matrix, q, p, rdd_partitioner)
        apsp_graph.persist(stglev.MEMORY_AND_DISK)
        apsp_graph.count()
    if (S == "cb"):
        t0 = time.time()
        # run apsp-solver Blocked-CB
        blocks_dir = os.getcwd() + '/_auxdir_/'
        os.system("mkdir " + blocks_dir)
        apsp_graph = collect_bc.collect_bc_block_fw(adj_matrix, q, p, rdd_partitioner, blocks_dir, sc)
        apsp_graph.persist(stglev.MEMORY_AND_DISK)
        apsp_graph.count()
        os.system("rm -r " + blocks_dir)

    t1 = time.time()

    apsp_graph.saveAsTextFile(out);

    sc.stop()

    logger.info("time to solution: " + str(t1 - t0) + " s")
    logger.info("using Python: " + sys.version)
    logger.info('done!')
