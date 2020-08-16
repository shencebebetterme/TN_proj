#include "itensor/all.h"
#include "itensor/util/print_macro.h"
//#include "omp.h"
//#include <armadillo>

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
//using namespace arma;

double phi = 0.5*(1+sqrt(5));
bool showInfo = true;
int iter_steps = 5;

// the 2d classical Ising tensor
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
// the 2d fibonacci tensor
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

// get the sqrt of a diagonal matrix
ITensor sqrtDiag(const ITensor& T){
    Index T1 = T.index(1);
    Index T2 = T.index(2);
    int dim = T1.dim();
    std::vector<Cplx> vec = {};
    for (auto i : range1(dim)){
        Cplx val = T.eltC(T1=i,T2=i);
        vec.push_back(val);
    }
    std::vector<Cplx> vec_sqrt = vec;
    auto sqrt_each = [](Cplx& x){ x = std::sqrt(x); };
    for_each(vec_sqrt.begin(), vec_sqrt.end(), sqrt_each);
    ITensor sqrtT = diagITensor(vec_sqrt,prime(T1),prime(T2));
    return sqrtT;
}

// create identity matrix
ITensor idDiag(int n){
    Index T1(n);
    Index T2(n);
    ITensor T = ITensor(T1,T2);
    double val = 0;
    for (auto i : range1(n)){
        for (auto j : range1(n)){
            if (i==j) val = 1;
            T.set(T1=i,T2=j,val);
        }
    }
}

// create random diagonal tensor
ITensor randomDiag(int dim){
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

//eigenvalue decomposition of a diagonal matrix V
// return X and X_inverse_dagger
std::tuple<ITensor,ITensor> decompV(ITensor& V){
    Index l = V.index(1);//Rr1
    Index r = V.index(2);//Rr2
    //int dim = l.dim();
    //double Vval = V.eltC(l=1,r=2).real();
    //printf("Vval is %.18f\n", Vval);
    //ITensor Vd = V;
    V *= delta(r,prime(l));
    //printf("\nmatrix to be decomposed is\n");
    //PrintData(Vd);
    auto [eigvec,eigval] = eigen(V);
    //Index d1 = eigval.index(1);
    //Index d2 = eigval.index(2);
    //Cplx val1 = eigval.eltC(d1=1, d2=1);
    //Cplx val2 = eigval.eltC(d1=2, d2=2);
    //if (showInfo) {printf("in decompV, val1 is %.18f\n",val1); printf("val2 is %.18f\n",val2);}
    //auto [eigvec,eigval] = diagHermitian(V*delta(r,prime(l)),{"ErrGoal=",1E-14});
	//PrintData(eigval); //PrintData(eigvec);
    //Link indices
    Index v2 = eigval.index(1);// primed Link
    Index v1 = eigval.index(2);// Link // commonIndex(U,D);
    int dim = v1.dim();
    std::vector<Cplx> vec = {};
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
    ITensor X = eigvec * sqrt_eigval;//X has index l and v2
    ITensor X_inv_dag = conj(eigvec * sqrt_inv_eigval);//X_inv_dag has index l and v2
    X_inv_dag *= delta(l,r);//X_inv_dag now has index r and v2
    return {X,X_inv_dag};
}

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
    return (max_loc + 1);//ITensor index is 1-based
}

//project eigvecs to the dominant one
//and return the dominant eigenvalue
Cplx projToMax(ITensor& eigvec, const ITensor& eigval){
    int max_loc = findMax(eigval);
    //project eigenvector
    Index link = commonIndex(eigvec,eigval);
    ITensor proj = setElt(link=max_loc);
    eigvec *= proj;
    // return the dominant eigenvalue
    Index T1 = eigval.index(1);
    Index T2 = eigval.index(2);
    return eigval.eltC(T1=max_loc,T2=max_loc);
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
Index Glam_l, Glam_r, Glam_d;
//contraction of sqrt(lambda) * Gamma * sqrt(lambda)
// or Gamma * delta(lambda) depending on situation
ITensor Glam; 

//random constructor
iMPS(int bd, int pd, bool use_setValue=false){
    bond_dim = bd;
    trunc_dim = bd;
    phys_dim = pd;
    Gl = Index(bond_dim,"Gl");
    Gd = Index(phys_dim,"Gd");
    Gr = Index(bond_dim,"Gr");
    Gamma = randomITensor(Gl,Gd,Gr);
    //Print(Gamma);
    //initial random lambda
    lambda = randomDiag(bond_dim);
    ll = lambda.index(1);
    lr = lambda.index(2);
    //
    if (use_setValue) setValue(*this);
    Glam = Gamma * lambda * delta(Gr,ll);
    Glam_l = Gl;
    Glam_r = lr;
    Glam_d = Gd;
}

//obtain R from Glam
void getR(){
    Rl1 = Index(bond_dim,"Rl1");
    Rl2 = Index(bond_dim,"Rl2");
    Rr1 = Index(bond_dim,"Rr1");
    Rr2 = Index(bond_dim,"Rr2");
    R = (delta(Rl1,Glam_l) * Glam * delta(Glam_r,Rr1)) * (delta(Rl2,Glam_l) * Glam * delta(Glam_r,Rr2));
}

//canonicalize Glam to obtain new Gamma and lambda
void canonicalize(){
    this -> getR();
    ITensor L = R;//make a copy of R
    // obtain VR and VL
    R *= delta(Rl1,prime(Rr1));
    R *= delta(Rl2,prime(Rr2));
    auto [VR, eigvalR] = eigen(R);
    Cplx etaR = projToMax(VR,eigvalR);
    L *= delta(Rr1,prime(Rl1));
    L *= delta(Rr2,prime(Rl2));
    auto [VL, eigvalL] = eigen(L);
    Cplx etaL = projToMax(VL,eigvalL);
    // normalization
    Glam /= std::sqrt(etaR);
    Cplx trR = eltC(VR*delta(Rr1,Rr2));
    Cplx trL = eltC(VL*delta(Rl1,Rl2));
    Cplx phaseR = trR / std::abs(trR);
    Cplx phaseL = trL / std::abs(trL);
    VR /= phaseR;
    VL /= phaseL;
    Cplx trLR = eltC(VR*VL*delta(Rr1,Rl1)*delta(Rr2,Rl2));
    trLR = std::abs(trLR);
    VR /= std::sqrt(trLR);
    VL /= std::sqrt(trLR);
    // obtain X and X_inv
    auto [X,X_inv] = decompV(VR);
    Index X_l(bond_dim);
    Index X_inv_r(bond_dim);
    X *= delta(X.index(1),X_l);
    X_inv *= delta(X_inv.index(1),X_inv_r);
    Index X_c = commonIndex(X,X_inv);
    // obtain Y and Y_inv
    auto [Y,Y_inv] = decompV(VL);
    Index Y_r(bond_dim);
    Index Y_inv_l(bond_dim);
    Y *= delta(Y.index(1),Y_r);
    Y_inv *= delta(Y_inv.index(1),Y_inv_l);
    Index Y_c = commonIndex(Y,Y_inv);
    // get new lambda
    auto [U,lambda_new,V] = svd(Y * delta(Y_r,X_l) * X, {Y_c},{"MaxDim=",trunc_dim,"MinDim=",trunc_dim});
    Index Ulink = commonIndex(U,lambda_new);
    Index Vlink = commonIndex(V,lambda_new);
    bond_dim = trunc_dim;
    // get new Gamma
    ITensor Gamma_new = V * X_inv * delta(X_inv_r,Glam_l) * Glam * delta(Glam_r,Y_inv_l) * Y_inv * U;
    // restore indices of Gamma and lambda
    Gl = Index(bond_dim,"Gl");
    Gr = Index(bond_dim,"Gr");
	Gamma = Gamma_new * delta(Vlink,Gl) * delta(Ulink,Gr);//now Gamma has index Gl Gd Gr
    lambda = lambda_new;
	ll = Ulink;
    lr = Vlink;
}

//obtain Glam from Gamma,lambda and do canonicalization
void step(const ITensor& H){

}

};//end of iMPS


void setValue(iMPS& m){
    m.lambda.set(m.ll=1, m.lr=1, 0.57);
    m.lambda.set(m.ll=2, m.lr=2, 0.35);
    //
    m.Gamma.set(m.Gl=1, m.Gd=1, m.Gr=1, 0.1);
    m.Gamma.set(m.Gl=1, m.Gd=1, m.Gr=2, 0.2);
    m.Gamma.set(m.Gl=1, m.Gd=2, m.Gr=1, 0.3);
    m.Gamma.set(m.Gl=1, m.Gd=2, m.Gr=2, 0.4);
    m.Gamma.set(m.Gl=1, m.Gd=3, m.Gr=1, 0.5);
    m.Gamma.set(m.Gl=1, m.Gd=3, m.Gr=2, 0.6);
    m.Gamma.set(m.Gl=2, m.Gd=1, m.Gr=1, 0.7);
    m.Gamma.set(m.Gl=2, m.Gd=1, m.Gr=2, 0.8);
    m.Gamma.set(m.Gl=2, m.Gd=2, m.Gr=1, 0.9);
    m.Gamma.set(m.Gl=2, m.Gd=2, m.Gr=2, 1.0);
    m.Gamma.set(m.Gl=2, m.Gd=3, m.Gr=1, 1.1);
    m.Gamma.set(m.Gl=2, m.Gd=3, m.Gr=2, 1.2);
}

int main(){
    showInfo = true;
    int D = 2;
    int phys_dim = 3;
    iMPS mps(D,phys_dim,true);
    if(showInfo) PrintData(mps.lambda);PrintData(mps.Gamma);
    mps.canonicalize();
    if(showInfo) PrintData(mps.lambda);PrintData(mps.Gamma);
}