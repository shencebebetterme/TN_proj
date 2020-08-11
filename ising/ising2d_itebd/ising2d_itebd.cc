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

// to use the arnoldi method
class ITensorMap
  {
  ITensor const& A_;
  mutable long size_;
  public:
  ITensorMap(ITensor const& A)
    : A_(A)
    {
    size_ = 1;
    for(auto& I : A_.inds())
      {
      if(I.primeLevel() == 0)
        size_ *= dim(I);
      }
    }

  void product(ITensor const& x, ITensor& b) const
    {
    b = A_*x;
    b.noPrime();
    }

  long size() const
    {
    return size_;
    }
};

//eigenvalue decomposition of a diagonal matrix V
// return X and X_inverse_dagger
std::tuple<ITensor,ITensor> decompV(const ITensor& V){
    Index l = V.index(1);
    Index r = V.index(2);
    //int dim = l.dim();
    std::vector<Cplx> vec = {};
    auto [eigvec,eigval] = eigen(V*delta(r,prime(l)));
    Index v1 = eigval.index(1);
    Index v2 = eigval.index(2);
    // eigvec has index l and v2
    //if (hasIndex(eigvec,l)) printf("\neigvec has index l\n");
    //if (hasIndex(eigvec,r)) printf("\neigvec has index r\n");
    //if (hasIndex(eigvec,v1)) printf("\neigvec has index v1\n");
    //if (hasIndex(eigvec,v2)) printf("\neigvec has index v2\n");
    //PrintData(eigvec);
    //PrintData(eigval);

    //v1.addTags("v1");
    //v2.addTags("v2");
    int dim = v1.dim();
    for (auto i : range1(dim)){
        Cplx val = eltC(eigval,v1=i,v2=i);
        vec.push_back(val);
    }
    //
    auto sqrt_each = [](Cplx& x){ x = std::sqrt(x); };
    auto sqrt_inv_each = [](Cplx& x){ x = 1/std::sqrt(x); };
    auto sqrt_vec = vec;
    for_each(sqrt_vec.begin(), sqrt_vec.end(), sqrt_each);
    auto sqrt_inv_vec = vec;
    for_each(sqrt_inv_vec.begin(), sqrt_inv_vec.end(), sqrt_inv_each);
    //
    ITensor sqrt_eigval = diagITensor(sqrt_vec,v1,v2);
    //PrintData(sqrt_eigval);
    ITensor sqrt_inv_eigval = diagITensor(sqrt_inv_vec,v1,v2);
    //Print(sqrt_inv_eigval);

    ITensor X = eigvec * sqrt_eigval;//X has index l and v1
    //ITensor X_dag = dag(X);
    ITensor X_inv_dag = conj(eigvec * sqrt_inv_eigval);//X_inv_dag has index l and v1
    X_inv_dag *= delta(l,r);//X_inv_dag now has index r and v1
    //PrintData(X);
    //PrintData(X_inv_dag);
    return {X,X_inv_dag};
}


// one site invariant iMPS
class iMPS{
public:
int bond_dim;
int phys_dim;
Index Gl,Gd,Gr; // the three indices around Gamma
Index ll,lr; // the two indices around lambda
Index Rl1,Rl2,Rr1,Rr2,Ll1,Ll2,Lr1,Lr2; //indices around L and R matrix
ITensor Gamma, lambda;
ITensor R,L;
//ITensor X,Y;//to store X and Y transpose

// random constructor
iMPS(int bd, int pd){
    bond_dim = bd;
    phys_dim = pd;
    Gl = Index(bond_dim,"Gl");
    Gd = Index(phys_dim,"Gu");
    Gr = Index(bond_dim,"Gr");
    Gamma = randomITensor(Gl,Gd,Gr);
    Print(Gamma);
    Rl1 = Index(bond_dim,"Rl1");
    Rl2 = Index(bond_dim,"Rl2");
    Rr1 = Index(bond_dim,"Rr1");
    Rr2 = Index(bond_dim,"Rr2");
    Ll1 = Index(bond_dim,"Ll1");
    Ll2 = Index(bond_dim,"Ll2");
    Lr1 = Index(bond_dim,"Lr1");
    Lr2 = Index(bond_dim,"Lr2");
    // create a random diagonal ITensor from svd
    ITensor lambda_temp = randomITensor(Gr,Gl);
    auto [U,S,V] = svd(lambda_temp,{Gr});
    ll = commonIndex(U,S);
    lr = commonIndex(S,V);
    lambda = S;
    Print(lambda);
}
// construct from given Gamma and lambda

// get the R and L transfer matrix
void getR(){
    // upper part has indices Rl1, Gd, Rl2
    ITensor upper_part = Gamma * lambda * delta(ll,Gr) * delta(Gl,Rl1) * delta(lr,Rr1);
    //Print(upper_part);
    // lower part has indices Rl2, Gd, Rr2
    ITensor lower_part = Gamma * lambda * delta(ll,Gr) * delta(Gl,Rl2) * delta(lr,Rr2);
    R = upper_part * lower_part;
}
void getL(){
    // uper part has indices Ll1, Gd, Lr1
    ITensor upper_part = lambda * Gamma * delta(lr,Gl) * delta(ll,Ll1) * delta(Gr,Lr1);
    // lower part has indices Ll2, Gd, Lr2
    ITensor lower_part = lambda * Gamma * delta(lr,Gl) * delta(ll,Ll2) * delta(Gr,Lr2);
    L = upper_part * lower_part;
}

void canonicalize(){
    this -> getR();
    this -> getL();
    //use arnoldi method to get the dominant eigenvector
    // R_ has indices r1,2 and primed r1,2
    // L_ has indices l1,2 and primed l1,2
    ITensor R_ = R * delta(Rl1,prime(Rr1)) * delta(Rl2,prime(Rr2));
    ITensor L_ = L * delta(Lr1,prime(Ll1)) * delta(Lr2,prime(Ll2));
    auto RM = ITensorMap(R_);
    auto LM = ITensorMap(L_);
    auto VR = randomITensor(Rr1,Rr2);//store the eigenvector
    auto VL = randomITensor(Ll1,Ll2);
    // eigen decomposition
    auto etaR = arnoldi(RM,VR);
    auto etaL = arnoldi(LM,VL);
    // normalization of VR and VL
    Cplx trLR = eltC(VR*VL*delta(Rr1,Ll1)*delta(Rr2,Ll2));
    //Print(trLR);
    VR /= std::sqrt(trLR);
    VL /= std::sqrt(trLR);
    // eigenvalue decomposition of VR
    //PrintData(VR);
    auto [X,X_inv] = decompV(VR);
    Index X_l(bond_dim);
    Index X_inv_r(bond_dim);
    X *= delta(X.index(1),X_l);
    X_inv *= delta(X_inv.index(1),X_inv_r);
    Index X_c = commonIndex(X,X_inv);
    // now X is a matrix X_l * X_c
    // X_inv is a matrix X_c * X_inv_r
    //PrintData(X*X_inv);
    auto [Y,Y_inv] = decompV(VL);
    Index Y_r(bond_dim);
    Index Y_inv_l(bond_dim);
    Y *= delta(Y.index(1),Y_r);
    Y_inv *= delta(Y_inv.index(1),Y_inv_l);
    Index Y_c = commonIndex(Y,Y_inv);
    //
    //get the new lambda and Gamma
    auto [U,lambda_new,V] = svd(Y * delta(Y_r,ll) * lambda * delta(lr,X_l) * X, {Y_c});
    Index Ulink = commonIndex(U,lambda_new);
    Index Vlink = commonIndex(V,lambda_new);
    //Print(lambda_new);
    // update lambda and restore the indices
    lambda = lambda_new;
    ll = Ulink;
    lr = Vlink;
    //Print(lambda);
    // update Gamma
    ITensor Gamma_new = V * X_inv * delta(X_inv_r,Gl) * Gamma * delta(Gr,Y_inv_l) * Y_inv * U;
    // restore the indices
    Gamma = Gamma_new * delta(Vlink,Gl) * delta(Ulink,Gr);
}


};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(){
    int D = 2;
    int phys_dim = 2;
    iMPS m(D,phys_dim);
    m.canonicalize();
    //if (hasIndex(m.lambda,m.ll)&&hasIndex(m.lambda,m.lr)) printf("\nlambda updated\n");
}