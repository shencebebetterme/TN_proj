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
	//PrintData(eigval);
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

// one site invariant iMPS
class iMPS{
public:
int bond_dim;
int trunc_dim;//max dim of svd
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
    trunc_dim = bd;
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
	Rl1 = Index(bond_dim,"Rl1");
    Rl2 = Index(bond_dim,"Rl2");
    Rr1 = Index(bond_dim,"Rr1");
    Rr2 = Index(bond_dim,"Rr2");
    // upper part has indices Rl1, Gd, Rl2
    ITensor upper_part = Gamma * lambda * delta(ll,Gr) * delta(Gl,Rl1) * delta(lr,Rr1);
    //Print(upper_part);
    // lower part has indices Rl2, Gd, Rr2
    ITensor lower_part = Gamma * lambda * delta(ll,Gr) * delta(Gl,Rl2) * delta(lr,Rr2);
    R = upper_part * lower_part;
	//PrintData(R);
}
void getL(){
	Ll1 = Index(bond_dim,"Ll1");
    Ll2 = Index(bond_dim,"Ll2");
    Lr1 = Index(bond_dim,"Lr1");
    Lr2 = Index(bond_dim,"Lr2");
    // uper part has indices Ll1, Gd, Lr1
    ITensor upper_part = lambda * Gamma * delta(lr,Gl) * delta(ll,Ll1) * delta(Gr,Lr1);
    // lower part has indices Ll2, Gd, Lr2
    ITensor lower_part = lambda * Gamma * delta(lr,Gl) * delta(ll,Ll2) * delta(Gr,Lr2);
    L = upper_part * lower_part;
	//PrintData(L);
}

void canonicalize(){
	printf("starting canonicalization\n");
    this -> getR();
    this -> getL();
	//Print(R);
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
	printf("\nstarting arnoldi algorithm\n");
    auto etaR = arnoldi(RM,VR,{"Cutoff=",1E-7});
    auto etaL = arnoldi(LM,VL,{"Cutoff=",1E-7});
	Print(VR);
	Print(VL);
    // normalization of VR and VL
    Cplx trLR = eltC(VR*VL*delta(Rr1,Ll1)*delta(Rr2,Ll2));
    VR /= std::sqrt(trLR);
    VL /= std::sqrt(trLR);
	//PrintData(VR);
	//PrintData(VL);
    // eigenvalue decomposition of VR
    //PrintData(VR);
	//Print(VR);
    auto [X,X_inv] = decompV(VR);
    Index X_l(bond_dim);
    Index X_inv_r(bond_dim);
    X *= delta(X.index(1),X_l);
    X_inv *= delta(X_inv.index(1),X_inv_r);
    Index X_c = commonIndex(X,X_inv);
	//PrintData(X);
	//PrintData(X_inv);
    // now X is a matrix X_l * X_c
    // X_inv is a matrix X_c * X_inv_r
    //PrintData(X*X_inv);
	//Print(VL);
    auto [Y,Y_inv] = decompV(VL);
    Index Y_r(bond_dim);
    Index Y_inv_l(bond_dim);
    Y *= delta(Y.index(1),Y_r);
    Y_inv *= delta(Y_inv.index(1),Y_inv_l);
    Index Y_c = commonIndex(Y,Y_inv);
	//PrintData(Y);
    //
    //get the new lambda and Gamma
	printf("lambda immediately before svd\n");
	PrintData(lambda);
	//printf("truncation dimension is %d\n", trunc_dim);
	//PrintData(Y * delta(Y_r,ll) * lambda * delta(lr,X_l) * X);
	#if 0
    auto [U,lambda_new,V] = svd(Y * delta(Y_r,ll) * lambda * delta(lr,X_l) * X, {Y_c},{"MaxDim=",trunc_dim,"MinDim=",1,"Cutoff=",1E-5});
	bond_dim = lambda_new.index(1).dim();
	printf("bond dimension is %f\n", bond_dim);
    Index Ulink = commonIndex(U,lambda_new);
    Index Vlink = commonIndex(V,lambda_new);
    //Print(lambda_new);
    // update lambda and restore the indices
    lambda = lambda_new;
	printf("lambda after svd\n");
	PrintData(lambda);
    ll = Ulink;
    lr = Vlink;
    //Print(lambda);
    // update Gamma
    ITensor Gamma_new = V * X_inv * delta(X_inv_r,Gl) * Gamma * delta(Gr,Y_inv_l) * Y_inv * U;
    // restore the indices
    Gamma = Gamma_new * delta(Vlink,Gl) * delta(Ulink,Gr);
	Print(Gamma);
	#else
	auto [U,lambda_new,V] = svd(Y * X * delta(Y_r,X_l), {Y_c},{"MaxDim=",trunc_dim});
	bond_dim = lambda_new.index(1).dim();
	Index Ulink = commonIndex(U,lambda_new);
    Index Vlink = commonIndex(V,lambda_new);
	//
	ITensor Gamma_new = X_inv * delta(X_inv_r,Gl) * Gamma * delta(Gr,ll) * lambda * delta(lr,Y_inv_l);
	Gamma = Gamma_new * delta(X_c,Gl) * delta(Y_c,Gr);
	lambda = lambda_new;
	ll = Ulink;
    lr = Vlink;
	#endif 
	printf("ending canonicalization\n");
}

//update Gamma and lambda after one layer of MPO
// the indices of A are ordered as Al, Au, Ar, Ad
// suppose dim(Au) = dim(Ad) = phys_dim, and dim(Al) = dim(Ar)
void step(const ITensor& A){
	//printf("\nstarting step\n");
    Index Al = A.index(1);
    Index Au = A.index(2);
    Index Ar = A.index(3);
    Index Ad = A.index(4);
	if (!(Au.dim()==phys_dim && Ad.dim()==phys_dim)) printf("\ndimension of A doesn't match\n");
	//
	ITensor Gamma_new = Gamma * delta(Gd,Au) * A;
	auto[GlC,Glc] = combiner(Gl,Al);
	auto[GrC,Grc] = combiner(Gr,Ar);
	Gamma = Gamma_new * GlC * GrC;
	Gl = Glc;
	Gr = Grc;
	Gd = Ad;
	//Print(Gamma);
	//
	//ITensor lambda_new = lambda * delta(Ar,Al);
	//a stupid way to make a dense identity tensor, because ITensor
	//doesn't support contracting two diagReal tensor
	ITensor idt(Ar,Al);
	for(auto i : range1(Ar.dim())){
		for(auto j : range1(Al.dim())){
			if (i==j) idt.set(Ar=i, Al=j, 1);
			else idt.set(Ar=i, Al=j, 0);
		}
	}
	ITensor lambda_new = lambda * idt;
	auto[llC,llc] = combiner(ll,Ar);
	auto[lrC,lrc] = combiner(lr,Al);
	lambda = lambda_new * llC * lrC;
	printf("lambda after combining\n");
	PrintData(lambda);
	ll = llc;
	lr = lrc;
	//Print(lambda);
	//update bond dimension
	bond_dim *= Al.dim();
	// truncation already included in canonicalization
	canonicalize();
}

};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(){
    int D = 5;//bond dimension
    int phys_dim = 2;
	int iter_steps = 9;
	ITensor A = getIsingTensor();
    iMPS m(D,phys_dim);
    m.canonicalize();
	for(auto i : range1(iter_steps)){
		printf("\nstep %d\n", i);
		m.step(A);
		PrintData(m.lambda);
	}
	PrintData(m.lambda);
	
}