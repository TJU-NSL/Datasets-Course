# APSPark: All-Pairs Shortest-Paths Solver for Apache Spark

**Authors:**
Frank Schoeneman <fvschoen@buffalo.edu>,
Jaroslaw Zola <jaroslaw.zola@hush.com>

## About
Algorithms for computing All-Pairs Shortest-Paths (APSP) are critical building blocks underlying many practical applications. The standard sequential algorithms, such as Floyd-Warshall and Johnson, quickly become infeasible for large input graphs, necessitating parallel approaches. `APSPark` provides algorithms for parallel APSP on distributed memory clusters with Apache Spark. The Spark model allows for a portable and easy to deploy distributed implementation, and hence is attractive from the end-user perspective. At the same time, the `APSPark` remains competitive when compared to highly optimized HPC-oriented solvers in MPI. The platform provides two specialized APSP solvers augmented with a custom RDD partitioner. To learn more about the solvers, please check our ICPP 2019 paper [1].

This work has been also extended by Mohammad Javanmard, Zafar Ahmad and colleagues into [DPSPark](https://github.com/TEAlab/DPSpark) to cover a broader spectrum of dynamic programming algorithms.
You can learn more about the project from [https://github.com/TEAlab/DPSpark](https://github.com/TEAlab/DPSpark) and the related publication in IEEE Cluster: M.M. Javanmard, Z. Ahmad, J. Zola, L.-N. Pouchet, R. Chowdhury, R. Harrison, Efficient Execution of Dynamic Programming Algorithms on Apache Spark, <https://ieeexplore.ieee.org/abstract/document/9229617>.


### Acknowledgment

This framework is part of the [MEADS](https://github.com/ubdsgroup/meads) project, and is supported by the [National Science Foundation](https://www.nsf.gov/) under the award [OAC-1910539](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1910539). 

## User Guide

`APSPark` is executed by calling `APSPark.py`. The software requires additional modules that are part of the source code. The easiest way to provide these modules to Spark is by first calling `pack.sh` (one time effort to produce `APSPark.zip`), and then adding `--py-files /path/to/APSPark.zip` to `spark-submit`. Additionally, the following Python modules must be available: `numba`, `numpy` and `scipy`. Please note that for the best performance, these modules should be configured to work with a high performance BLAS library (for example, Intel MKL).

`APSPark` supports the following command line options:

* `-n` number of vertices in the input graph 
* `-b` block size for adjacency matrix decomposition
* `-p` number of Spark RDD partitions to store adjacency matrix
* `-F` RDD partitioner (use `md` for custom multi-diagonal partitioner, or `ph` for the default Spark partitioner)
* `-S` solver type (use `im` for in-memory solver, or `cb` for collect-broadcast via persistent storage)
* `-f` input graph (`tsv` format: `source target weight`)
* `-o` output folder

Please refer to the original `APSPark` paper [1] if the meaning of the options is not clear.

##### Example invocation

`spark-submit --py-files ./APSPark.zip ./APSPark.py -n 8192 -b 1024 -p 16 -F md -S im -f data/er8K-0.01.txt -o out
`
The above invocation will execute Blocked-IM approach using multi-diagonal partitioner on our toy example data `er8K-0.01.txt` with 8192 nodes. The result will be stored in `out` folder.

When calling `APSPark` with the `cb` solver, for example:

`spark-submit --py-files ./APSPark.zip ./APSPark.py -n 8192 -b 1024 -p 16 -F md -S cb -f data/er8K-0.01.txt -o out
`
the current working directory must be a shared file system, i.e., all nodes is the executing Spark cluster must be able to read/write to that directory.

If you have immediate questions, please do not hesitate to contact Jaric Zola <jaroslaw.zola@hush.com>.

## References

To cite `APSPark`, please refer to this repository and our paper:

1. F. Schoeneman, J. Zola, Solving All-Pairs Shortest-Paths Problem in Large Graphs Using Apache Spark. In Proc. of International Conference on Parallel Processing (ICPP), 2019. <https://dl.acm.org/citation.cfm?id=3337852>, <https://arxiv.org/abs/1902.04446>.
