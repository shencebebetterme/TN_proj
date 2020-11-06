#include "itensor/all.h"
#include "itensor/util/print_macro.h"

#include <complex>
#include <algorithm>
#include <iostream>

using namespace itensor;

//find the location of max element of diagonal tensor
int findMax(const ITensor& T){
    Index T1 = T.index(1);
    Index T2 = T.index(2);
    std::vector<double> vec = {};
    for (auto i : range1(T1.dim())){
        Cplx val_i = T.eltC(T1=i,T2=i);
        vec.push_back(std::abs(val_i));
    }
    auto max_pt = std::max_element(vec.begin(),vec.end());
    auto max_loc = std::distance(vec.begin(),max_pt);
    return max_loc;
}

void chop(ITensor& T){
    auto applychop = [](Cplx& x){
        if(std::abs(x)<0.5) x=0;
    };
    T.visit(applychop);
}

int main(){
    Index i(3,"i");
    Index j(3,"j");
    ITensor A = randomITensor(i,prime(i));
    PrintData(A);
    auto add = [](double& x) {x+=1;};
    //chop(A);
    A.visit(add);
    PrintData(A);
}