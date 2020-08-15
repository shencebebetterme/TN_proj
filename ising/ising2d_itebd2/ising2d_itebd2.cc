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

double phi = 0.5*(1+sqrt(5));

// the 2d classical Ising model
ITensor getIsingTensor(){
	int dim0 = 2;
	double beta_c = 0.5*log(1+sqrt(2));//critical beta
 	Index s(dim0);
    Index u = addTags(s,"up");
    Index r = addTags(s,"right");
    Index d = addTags(s,"down");
    Index l = addTags(s,"left");

    // the initial tensor for Ising
    ITensor A = ITensor(l,u,r,d);

    // Fill the A tensor with correct Boltzmann weights:
	auto Sig = [](int s) { return 1. - 2. * (s - 1); };
    // 1 -> 1
    // 2-> -1
	for (auto sl : range1(dim0))
		for (auto sd : range1(dim0))
			for (auto sr : range1(dim0))
				for (auto su : range1(dim0))
				{
					auto E = Sig(sl) * Sig(sd) + Sig(sd) * Sig(sr)
						+ Sig(sr) * Sig(su) + Sig(su) * Sig(sl);
					auto P = exp(E * beta_c);
					A.set(l(sl), r(sr), u(su), d(sd), P);
                    //A.set(l(sl), r(sr), u(su), d(sd), 0);
				}
    // normalize A to prevent it from being too large
    double TrA = elt(A * delta(l, r) * delta(u, d));
	A /= TrA;
	return A;
}
ITensor getFibonacciTensor(){
    int dim0 = 3;
    Index u(dim0);
    Index r(dim0);
    Index d(dim0);
    Index l(dim0);
    //Print(u);
    //Print(r);
    //Print(d);
    //Print(l);
    ITensor C(l,u,r,d);
    for (auto sl : range1(dim0))
		for (auto su : range1(dim0))
			for (auto sr : range1(dim0))
				for (auto sd : range1(dim0))
				{
					double val = 0;
                    if ((sl==3)&&(su==3)&&(sr==3)&&(sd==3))  val = std::pow(phi,-1.0/2);
                    if (((sl==1)&&(su==1)&&(sr==3)&&(sd==3)) || ((sl==3)&&(su==3)&&(sr==2)&&(sd==2))) val = -std::pow(phi,1.0/3);
                    if (((sl==3)&&(su==2)&&(sr==1)&&(sd==3)) || ((sl==2)&&(su==3)&&(sr==3)&&(sd==1)) || ((sl==1)&&(su==1)&&(sr==2)&&(sd==2)) ) val = std::pow(phi,7.0/6);
					//printf("val = %f\n", val);
                    C.set(l(sl), r(sr), u(su), d(sd), val);
                    //A.set(l(sl), r(sr), u(su), d(sd), 0);
				}
    //PrintData(C);
    return C;
}


ITensor randomDiagITensor(int dim){
    Index i(dim);
    Index j = prime(i);
    ITensor randT = randomITensor(i,j);
    for (auto i1 : range1(dim)){
        for (auto j1 : range1(dim)){
            if (i1!=j1) randT.set(i=i1, j=j1, 0);
        }
    }
    return randT;
}
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

// obtain the sqrt and sqrt inv of a diagonal matrix
std::tuple<ITensor,ITensor> ssi(const ITensor& T){
    std::vector<Cplx> vec = {};
    Index T1 = T.index(1);
    Index T2 = T.index(2);
    for (auto i : range1(T1.dim())){
        Cplx Tii = eltC(T, T1=i, T2=i);
        vec.push_back(Tii);
    }
    //
    auto sqrt_each = [](Cplx& x){ x = std::sqrt(x); };
    auto sqrt_inv_each = [](Cplx& x){ x = 1/std::sqrt(x); };
    std::vector<Cplx> vec_sqrt = vec;
    std::vector<Cplx> vec_sqrt_inv = vec;
    for_each(vec_sqrt.begin(), vec_sqrt.end(), sqrt_each);
    for_each(vec_sqrt_inv.begin(), vec_sqrt_inv.end(), sqrt_inv_each);
    ITensor Tsqrt = diagITensor(vec_sqrt,T1,T2);
    ITensor TsqrtInv = diagITensor(vec_sqrt_inv,T1,T2);
    return {Tsqrt, TsqrtInv};
}

class iMPS{
public:
int bond_dim = 1;
int trunc_dim = 1;
int phys_dim = 2;
ITensor Gamma, lambda, A;
Index Gl,Gd,Gr,Al,Ad,Ar,ll,lr;
ITensor R;
Index Rl1;
Index Rl2;
Index Rr1;
Index Rr2;
//constructor
iMPS(int bd, int pd){
    bond_dim = bd;
    trunc_dim = bd;
    phys_dim = pd;
    Gl = Index(bond_dim,"Gl");
    Gr = Index(bond_dim,"Gr");
    Gd = Index(phys_dim,"Gd");
    Gamma = randomITensor(Gl,Gd,Gr);
    //Index ll(bond_dim,"ll");
    //Index lr(bond_dim,"lr");
    lambda = randomDiagITensor(bond_dim);
    ll = lambda.index(1);
    lr = lambda.index(2);
    getA();
}

// obtain A from Gamma and lambda
void getA(){
    //printf("starting getA\n");
    //Index Gr = Gamma.index(3);
    Index ll = lambda.index(1);
    Index lr = lambda.index(2);
    Al = Index(bond_dim,"Al");
    Ad = Index(phys_dim,"Ad");
    Ar = Index(bond_dim,"Ar");
    if (Gr.dim() != ll.dim()) {
        printf("dimension doesn't match in getA\n");
        std::abort();
    }
    //Print(Gamma); Print(lambda);
    A = Gamma * lambda * delta(Gr,ll);
    //Print(A);
    A *= delta(Gl,Al);
    A *= delta(lr,Ar);
    A *= delta(Gd,Ad);
    //Print(A);
}
// obtain R from A
void getR(){
    //getA();
    Rl1 = Index(bond_dim,"Rl1");
    Rl2 = Index(bond_dim,"Rl2");
    Rr1 = Index(bond_dim,"Rr1");
    Rr2 = Index(bond_dim,"Rr2");
    //Index Al = A.index(1);
    //Index Ar = A.index(3);
    R = (delta(Al,Rl1) * A * delta(Ar,Rr1)) * (delta(Al,Rl2) * A * delta(Ar,Rr2));
}

void canonicalize(){
    printf("\n\n\n\n\n starting canonicalization\n");
    getR();
    //PrintData(A);PrintData(R);
    // obtain VR
    //Print(R);Print(A);
    #if 1
    ITensor R_tmp = R * delta(Rl1,prime(Rr1)) * delta(Rl2,prime(Rr2));
    auto R_map = ITensorMap(R_tmp);
    ITensor VR = randomITensor(Rr1,Rr2);
    //PrintData(R_tmp);
    auto etaR = arnoldi(R_map,VR,{"ErrGoal=",1E-14,"WhichEig=","LargestMagnitude"});
    // obtain VL
    ITensor L_tmp = R * delta(Rr1,prime(Rl1)) * delta(Rr2,prime(Rl2));
    auto L_map = ITensorMap(L_tmp);
    ITensor VL = randomITensor(Rl1,Rl2);
    //PrintData(L_tmp);
    auto etaL = arnoldi(L_map,VL,{"ErrGoal=",1E-14,"WhichEig=","LargestMagnitude"});
    //printf("VR VL obtained\n");
    #endif 
    PrintData(etaR); PrintData(etaL); PrintData(VR); PrintData(VL);
    // normalization
    Cplx trLR = eltC(VR*VL*delta(Rr1,Rl1)*delta(Rr2,Rl2));
    VR /= std::sqrt(trLR);
    VL /= std::sqrt(trLR);
    // etaR and etaL should be equal
    if (std::abs(etaR-etaL) > 1e-7) {
        printf("etaR and etaL not equal\n");
        printf("etaR = %f\n", etaR);
        printf("etaL = %f\n\n", etaL);
        std::abort();
    }
    // normalize the operator s.t. the eigenvalue becomes 1
    A /= std::sqrt(etaR);
    // split VR and VL
    auto [eigvecR,eigvalR] = eigen( VR * delta(Rr2,prime(Rr1)) );
    auto [eigvecL,eigvalL] = eigen( VL * delta(Rl2,prime(Rl1)) );
    PrintData(eigvalR); PrintData(eigvecR); PrintData(eigvalL); PrintData(eigvecL);
    auto [eigvalR_s, eigvalR_si] = ssi(eigvalR);
    auto [eigvalL_s, eigvalL_si] = ssi(eigvalL);
    // obtain X, Xinv, Y, Yinv
    ITensor X = eigvecR * eigvalR_s;
    ITensor Xinv = eigvecR * eigvalR_si;
    ITensor Y = eigvecL * eigvalL_s;
    ITensor Yinv = eigvecL * eigvalL_si;
    // change index
    Index X_out = Index(bond_dim,"X_o");//the outer index
    Index Xinv_out = Index(bond_dim,"Xinv_o");
    X *= delta(X.index(1), X_out);
    Xinv *= delta(Xinv.index(1), Xinv_out);
    Index X_c = commonIndex(X,Xinv);
    //
    Index Y_out = Index(bond_dim,"Y_o");//the outer index
    Index Yinv_out = Index(bond_dim, "Yinv_o");
    Y *= delta(Y.index(1), Y_out);
    Yinv *= delta(Yinv.index(1), Yinv_out);
    Index Y_c = commonIndex(Y,Yinv);
    // obtain new lambda and gamma
    auto [U,lambda_new,V] = svd( Y * X * delta(Y_out,X_out), {Y_c}, {"MaxDim=",trunc_dim,"MinDim=",trunc_dim});
    Index Ulink = commonIndex(U,lambda_new);
    Index Vlink = commonIndex(V,lambda_new);
    // update lambda and Gamma, bond dimensioni
    lambda = lambda_new;
    Gamma = V * Xinv * delta(Xinv_out, Al) * A * delta(Ar,Yinv_out) * Yinv * U;
    bond_dim = trunc_dim;
    //Print(A); Print(Gamma);
    //restore index
    Gl = Index(bond_dim,"Gl");
    Gr = Index(bond_dim,"Gr");
    Gd = Index(phys_dim,"Gd");
    Gamma *= delta(Vlink,Gl);
    Gamma *= delta(Ulink,Gr);
    Gamma *= delta(Ad,Gd);
    //
    //printf("ending canonicalization\n");
}//end of canonicalize()

// evolution under H
void step(const ITensor& H){
    //printf("starting step\n");
    getA();
    //Print(A); Print(H);
    Index Hl = H.index(1);
    Index Hu = H.index(2);
    Index Hr = H.index(3);
    Index Hd = H.index(4);
    auto [AlC,Alc] = combiner(Al,Hl);
    auto [ArC,Arc] = combiner(Ar,Hr);
    ITensor A_new = A * H * delta(Ad,Hu);
    //Print(A_new);
    A_new *= AlC;
    A_new *= ArC;
    A = A_new;
    //Print(A);
    bond_dim *= Alc.dim();
    Al = Alc; Ar = Arc; Ad = Hd;
    //A *= delta(Alc,Al);
    //A *= delta(Arc,Ar);
    //A *= delta(Ad,Hd);
    //Print(A);
    canonicalize();
    //Print(A);
    //printf("step finished\n");
}

};

void setValue(iMPS& m){
    m.lambda.set(m.lambda.index(1)=1, m.lambda.index(2)=1, 0.57);
    m.lambda.set(m.lambda.index(1)=2, m.lambda.index(2)=2, 0.35);
    //
    m.Gamma.set(m.Gamma.index(1)=1, m.Gamma.index(2)=1, m.Gamma.index(3)=1, 0.1);
    m.Gamma.set(m.Gamma.index(1)=1, m.Gamma.index(2)=1, m.Gamma.index(3)=2, 0.2);
    m.Gamma.set(m.Gamma.index(1)=1, m.Gamma.index(2)=2, m.Gamma.index(3)=1, 0.3);
    m.Gamma.set(m.Gamma.index(1)=1, m.Gamma.index(2)=2, m.Gamma.index(3)=2, 0.4);
    m.Gamma.set(m.Gamma.index(1)=1, m.Gamma.index(2)=3, m.Gamma.index(3)=1, 0.5);
    m.Gamma.set(m.Gamma.index(1)=1, m.Gamma.index(2)=3, m.Gamma.index(3)=2, 0.6);
    m.Gamma.set(m.Gamma.index(1)=2, m.Gamma.index(2)=1, m.Gamma.index(3)=1, 0.7);
    m.Gamma.set(m.Gamma.index(1)=2, m.Gamma.index(2)=1, m.Gamma.index(3)=2, 0.8);
    m.Gamma.set(m.Gamma.index(1)=2, m.Gamma.index(2)=2, m.Gamma.index(3)=1, 0.9);
    m.Gamma.set(m.Gamma.index(1)=2, m.Gamma.index(2)=2, m.Gamma.index(3)=2, 1.0);
    m.Gamma.set(m.Gamma.index(1)=2, m.Gamma.index(2)=3, m.Gamma.index(3)=1, 1.1);
    m.Gamma.set(m.Gamma.index(1)=2, m.Gamma.index(2)=3, m.Gamma.index(3)=2, 1.2);
}

int main(){
    int chi = 2;
    int d = 3;
    iMPS mps(chi,d);
    setValue(mps); mps.getA();
    PrintData(mps.lambda); PrintData(mps.Gamma);
    mps.canonicalize();
     PrintData(mps.lambda); PrintData(mps.Gamma);
    mps.canonicalize();
     PrintData(mps.lambda); PrintData(mps.Gamma);
    //auto H = getFibonacciTensor();
    //mps.step(H);
    #if 0
    for (auto i : range1(100)){
        printf("step %d\n",i);
       mps.step(H);
       PrintData(mps.lambda);
    }
    #endif
    //PrintData(mps.lambda);
}