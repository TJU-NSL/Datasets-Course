#include <iostream>
#include <random>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>

int main(int argc, char* argv[]) {
    typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS> Graph;
    typedef boost::erdos_renyi_iterator<std::mt19937, Graph> erg;

    if (argc < 3) {
       std::cout << "usage: " << argv[0] << " n p [idx0]" << std::endl;
       std::cout << "n: number of nodes" << std::endl;
       std::cout << "p: edge probability" << std::endl;
       std::cout << "idx0: starting index for nodes (default is 0)" << std::endl;
       return -1;
    }

    long int n = std::atoi(argv[1]);
    double eps = std::atof(argv[2]);

    long int s = 0 ;
    if (argc == 4) s = std::atol(argv[3]);

    //double p = (eps / n);
    double p = eps;

    std::mt19937 rng;
    Graph G(erg(rng, n, p), erg(), n);

    auto W = std::uniform_real_distribution<float>(0.0, 100.0);

    boost::graph_traits<Graph>::edge_iterator e, end;
    std::tie(e, end) = boost::edges(G);

    for (; e != end; ++e) {
        std::cout << (s + boost::source(*e, G)) << " "
                  << (s + boost::target(*e, G)) << " "
                  << W(rng) << "\n";
    }

    return 0;
} // main
