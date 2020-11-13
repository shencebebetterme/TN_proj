#include "inc.h"


int main(){
    auto i = Index(3,"i");
    auto j = Index(3,"j");
    //auto k = Index(4,"k");

    auto T = ITensor(i,j);
    T.randomize();
    PrintData(T);

    auto extractReal = [](Dense<Real> const& d)
    {
    return d.store;
    };

    auto v = applyFunc(extractReal,T.store());

    //std::vector<double> vec(v);

    for_each(v.begin(), v.end(), [](auto x){std::cout<<x<<std::endl;});

    arma::mat m(&v[0],3,3,false);
    m.print();

    return 0;
}