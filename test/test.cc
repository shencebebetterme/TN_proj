#include "itensor/all.h"
#include "itensor/util/print_macro.h"

using namespace itensor;

int main(){
    Index i(3,"i");
    Index j(3,"j");
    Index k(2,"k");
    Index l(4,"l");
    ITensor A = randomITensor(i,j,k);
    A *= delta(i,j,l);
    PrintData(A);
}