#include "itensor/all.h"
#include "itensor/util/print_macro.h"
//#include "omp.h"
#include <armadillo>

#include <cmath>
#include <string>
#include <cstring>
#include <vector>
//#include <array>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <complex>
#include <cstdlib>
//#include <ctime>
#include <chrono>
#include <tuple>
//#include <random>

using namespace itensor;
using namespace std;
using namespace std::chrono;
using namespace arma;

#define PI 3.1415926535

double h = 1.0;
int D = 5;//bond dimension
int phys_dim = 2;
double step_time = 0.01;
int num_steps = 10; // number of iteration steps

 // M is extremely sparse, so choose a different strategy
arma::sp_mat extract_sparse_mat(const ITensor& T){
    // here T is sparse matrix ITensor
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    arma::sp_mat Tmat(di,dj);
    //
    for (auto it : iterInds(T)){
        double val = T.real(it);
        if (val != 0){
            IndexVal iv_i = it[0];//iv_i is 1-based
            IndexVal iv_j = it[1];
            int i = iv_i.val -1;// i is 0-based
            int j = iv_j.val -1;
            Tmat(i,j) = val;
        }
    }
    return Tmat;
}

class itebd{
public:
int bond_dim;
int phys_dim;
ITensor Gamma_A;
ITensor Gamma_B;
ITensor lambda_A;
ITensor lambda_B;
ITensor H;
Index Hld,Hrd,Hlu,Hru;
Index Al,Au,Ar,Bl,Bu,Br;
//Index Gamma_A_lambda_A;
//Index Gamma_A_lambda_B;
//Index Gamma_B_lambda_A;
//Index Gamma_B_lambda_B;
//constructor
itebd(ITensor& Hamil, int bd=1, int pd=2){
    H = Hamil;
    //these four indices also appear in the evolution operator
    Hld = Hamil.index(1);
    Hlu = Hamil.index(2);
    Hrd = Hamil.index(3);
    Hru = Hamil.index(4);
    bond_dim = bd;
    phys_dim = pd;
    Index bond(bond_dim);
    Index phys_leg(pd);
    Al = addTags(bond,"Al");
    Ar = addTags(bond,"Ar");
    Bl = addTags(bond,"Bl");
    Br = addTags(bond,"Br");
    Au = addTags(phys_leg,"Au");
    Bu = addTags(phys_leg,"Bu");
    Gamma_A = randomITensor(Al,Au,Ar);
    Gamma_B = randomITensor(Bl,Bu,Br);
    //Print(Gamma_A);
    //Print(Gamma_B);
    //lambda_A = ITensor(Br,Al);
    //lambda_B = ITensor(Ar,Bl);
    initial_canonicalization();
}

void initial_canonicalization(){
    ITensor AB = Gamma_A * Gamma_B * delta(Ar,Bl);
    svd(AB,Gamma_A,lambda_B,Gamma_B,{"MaxDim=",bond_dim});
    Ar = commonIndex(Gamma_A,lambda_B);
    Bl = commonIndex(Gamma_B,lambda_B);
    PrintData(lambda_B);
  
    ITensor BA = Gamma_B * Gamma_A * delta(Br,Al);
    svd(BA,Gamma_B,lambda_A,Gamma_A,{"MaxDim=",bond_dim});
    Br = commonIndex(Gamma_B,lambda_A);
    Al = commonIndex(Gamma_A,lambda_A);
    PrintData(lambda_A);
}

void initial_canonicalization2(){
    lambda_A = randomITensor(Br,Al);
    lambda_B = randomITensor(Ar,Bl);
}

// imaginary time evolution by dt
void step(double dt){
    //Print(H);
    auto evol = expHermitian(H,-dt);
    //Print(evol);
    // AB evolution
    ITensor AB = evol * delta(Au,Hld) * delta(Bu,Hrd) * Gamma_A * lambda_B * Gamma_B;
    // update the indices of Gamma_A/B to store the svd result
    //Print(AB);
    Gamma_A *= delta(Au,Hlu);
    //Print(Gamma_A);
    Gamma_B *= delta(Bu,Hru);
    svd(AB,Gamma_A,lambda_B,Gamma_B,{"MaxDim=",bond_dim});
    Au = Hlu;
    Bu = Hru;
    Ar = commonIndex(Gamma_A,lambda_B);
    Bl = commonIndex(Gamma_B,lambda_B);
    //Print(Gamma_A);
    ITensor BA = evol * delta(Bu,Hld) * delta(Au,Hrd) * Gamma_B * lambda_A * Gamma_A;
    svd(BA,Gamma_B,lambda_A,Gamma_A,{"MaxDim=",bond_dim});
    //Print(Gamma_A);
    Bu = Hlu;
    Au = Hru;
    Br = commonIndex(Gamma_B,lambda_A);
    Al = commonIndex(Gamma_A,lambda_A);
    //Print(evol);
    //std::cout << hasIndex(evol,Hld) <<std::endl;
    //normalization
    ITensor A_temp = Gamma_A * lambda_B;
    // normaliza s.t. the first singular values 
    // in lambda_B and lambda_A are not large
    Index B1 = lambda_B.index(1);
    Index B2 = lambda_B.index(2);
    Index A1 = lambda_A.index(1);
    Index A2 = lambda_A.index(2);
    double val_A = elt(lambda_A,A1=1,A2=1);
    val_A = std::abs(val_A);
    double val_B = elt(lambda_B,B1=1,B2=1);
    val_B = std::abs(val_B);
    double val = std::fmax(val_A,val_B);
    val = (val>1) ? val : 1;
    printf("\n%f\t%f\t%f\n",val_A,val_B,val);
    //Gamma_A /=  val;
    //Gamma_B /=  val;
    //lambda_A /= val_A;
    //lambda_B /= val_B;
    //double trace = elt(H * H);
    //double trace = 2.0;
    //Print(trace);
    PrintData(lambda_B);
    PrintData(lambda_A);
    PrintData(Gamma_B);
    PrintData(Gamma_A);
}
};

int main(){
    // manual construction of Hamiltonian is abandoned
    #if 0
    Index i(2);
    Index j(2);
    ITensor sigma_x(i,j);
    ITensor sigma_z(i,j);
    sigma_x.set(i=1,j=1,0);
    sigma_x.set(i=2,j=2,0);
    sigma_x.set(i=1,j=2,1);
    sigma_x.set(i=2,j=1,1);
    sigma_z.set(i=1,j=1,1);
    sigma_z.set(i=2,j=2,-1);
    sigma_z.set(i=1,j=2,0);
    sigma_z.set(i=2,j=1,0);
    Index Hld(2,"Hld");
    Index Hlu(2,"Hlu");
    Index Hrd(2,"Hrd");
    Index Hru(2,"Hru");
    ITensor H1l = sigma_x * delta(i,Hld) * delta(j,Hlu);
    ITensor H1r = sigma_x * delta(i,Hrd) * delta(j,Hru);
    ITensor H1 = H1l * H1r;
    ITensor H2l = sigma_z * delta(i,Hld) * delta(j,Hlu);
    ITensor H2r = delta(Hrd,Hru);//the identity tensor
    ITensor H2 = h/2 * H2l * H2r;
    ITensor H_ising = H1 + H2;
    Print(H_ising);

    int D = 10;
    itebd ising(H_ising,D,2);
    //ising.step(0.1);
    //Print(ising.Gamma_A);
    //Print(ising.lambda_A);
    #endif 

    SpinHalf sites(2,{"ConserveQNs=",false});
    auto ampo = AutoMPO(sites);
    ampo += 4.0, "Sx", 1, "Sx", 2;
    ampo += h, "Sz", 1;
    ampo += h, "Sz", 2;
    auto Hmpo = toMPO(ampo);
    ITensor H_ising = Hmpo.A(1) * Hmpo.A(2);
    //Print(H_ising);
    // the itebd class
    itebd ising(H_ising, D, phys_dim);
    //std::cout << hasIndex(ising.Gamma_A,ising.Al) << std::endl;
    #if 0
    for (double e = 0; e > -5.5; e -= 0.1) {
        double dt = std::pow(10, e);
        for(auto i : range1(20)){
            ising.step(dt);
        }
    }
    #else
    for (auto i : range1(num_steps)){
        ising.step(step_time);
    }
    #endif
    //PrintData(ising.H);
    //PrintData(ising.Gamma_A);
    //PrintData(ising.Gamma_B);
    PrintData(ising.lambda_B);
    PrintData(ising.lambda_A);
    //PrintData(ising.Gamma_A);
    //PrintData(ising.Gamma_B);
    
    #if 0
    ITensor T = ising.Gamma_A * ising.lambda_B;
    //PrintData(T);
    //std::cout << hasIndex(T,ising.Bl);
    // dressing the index of T
    Index l(D);
    Index r(D);
    Index u(phys_dim);
    T *= delta(ising.Al,l);
    T *= delta(ising.Bl,r);
    T *= delta(ising.Au,u);
    // T dagger
    Index lprime(D);
    Index rprime(D);
    ITensor Tprime = T * delta(l,lprime) * delta(r,rprime);
    // T Tdager
    T *= Tprime;
    // combine indices to form a D^2 * D^2 matrix
    auto [lC,lc] = combiner(lprime,l);
    auto [rC,rc] = combiner(rprime,r);
    T *= lC;
    T *= rC;
    Print(T);
    
    sp_mat Tmat = extract_sparse_mat(T);
    cx_vec eigval = eigs_gen(Tmat,2,"lm",0.0001);
    eigval.print("eigval");
    #endif
}