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
    Index l = V.index(1);//Rr1
    Index r = V.index(2);//Rr2
    //int dim = l.dim();
    auto [eigvec,eigval] = eigen(V*delta(r,prime(l)),{"ErrGoal=",1E-14});
	//PrintData(eigval); PrintData(eigvec);
    //Link indices
    Index v2 = eigval.index(1);// primed Link
    Index v1 = eigval.index(2);// Link // commonIndex(U,D);
    //v1.addTags("v1");
    //v2.addTags("v2");
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
    //ITensor X_dag = dag(X);
    ITensor X_inv_dag = conj(eigvec * sqrt_inv_eigval);//X_inv_dag has index l and v2
    X_inv_dag *= delta(l,r);//X_inv_dag now has index r and v2
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
ITensor Glam; //contraction of Gamma and lambda
//ITensor X,Y;//to store X and Y transpose

// random constructor
iMPS(int bd, int pd){
    bond_dim = bd;
    trunc_dim = bd;
    phys_dim = pd;
    Gl = Index(bond_dim,"Gl");
    Gd = Index(phys_dim,"Gd");
    Gr = Index(bond_dim,"Gr");
    Gamma = randomITensor(Gl,Gd,Gr);
    //Print(Gamma);
    //initial random lambda
    ll = Index(bond_dim,"ll");
    lr = Index(bond_dim,"lr");
    lambda = randomITensor(ll,lr);
    for(auto li : range1(bond_dim)){
        for(auto lj : range1(bond_dim)){
            if (li!=lj) {
                lambda.set(ll=li, lr=lj, 0);
            }
        }
    }
    //Print(lambda);
}
// construct from given Gamma and lambda

//construct Glam from Gamma and lambda
void getGlam(){
    Glam_l = prime(Gl); Glam_r = prime(lr); Glam_d = prime(Gd);
}
// get the R and L transfer matrix
void getR(){
	Rl1 = Index(bond_dim,"Rl1");
    Rl2 = Index(bond_dim,"Rl2");
    Rr1 = Index(bond_dim,"Rr1");
    Rr2 = Index(bond_dim,"Rr2");
    // upper part has indices Rl1, Gd, Rl2
	//Print(Gamma); Print(lambda);
    ITensor upper_part = delta(Rl1,Gl) * Gamma * delta(Gr,ll) * lambda * delta(lr,Rr1);
    // lower part has indices Rl2, Gd, Rr2
    ITensor lower_part = delta(Rl2,Gl) * conj(Gamma) * delta(Gr,ll) * conj(lambda) * delta(lr,Rr2);
    //PrintData(upper_part);PrintData(lower_part);
    R = upper_part * lower_part;
	//PrintData(R);
}
void getL(){
	Ll1 = Index(bond_dim,"Ll1");
    Ll2 = Index(bond_dim,"Ll2");
    Lr1 = Index(bond_dim,"Lr1");
    Lr2 = Index(bond_dim,"Lr2");
    // uper part has indices Ll1, Gd, Lr1
    ITensor upper_part = delta(Ll1,ll) * lambda * delta(lr,Gl) * Gamma * delta(Gr,Lr1);
    // lower part has indices Ll2, Gd, Lr2
    ITensor lower_part = delta(Ll2,ll) * conj(lambda) * delta(lr,Gl) * conj(Gamma) * delta(Gr,Lr2);
    L = upper_part * lower_part;
	//PrintData(L);
}

double calcEntropy(){
    double ee = 0;
    Index l1 = lambda.index(1);
    Index l2 = lambda.index(2);
    for ( auto i : range1(l1.dim())){
        double val = lambda.elt(l1=i,l2=i);
        ee += -2 * std::pow(val,2) * std::log(val); 
    }
    return ee;
}

void canonicalize(){
	//printf("\n\n\n\n\n starting canonicalization\n");
    this -> getR();
    this -> getL();
    //PrintData(R);
    ITensor R_copy = R;
    //PrintData(R); PrintData(L);
    R *= delta(Rl1,prime(Rr1));
    R *= delta(Rl2,prime(Rr2));
    auto RM = ITensorMap(R);
    auto VR = randomITensor(Rr1,Rr2);//store the eigenvector
    auto etaR = arnoldi(RM,VR,{"ErrGoal=",1E-14});
    // use transpoes(R) to get the left eigenvalue of L
    #define USE_R_TRANSPOSE 1
    #if USE_R_TRANSPOSE
        ITensor TR = R_copy;
        TR *= delta(Rr1,prime(Rl1));
        TR *= delta(Rr2,prime(Rl2));//now TR has indices Rl1,2 and primed Rl1,2
        auto RM_T = ITensorMap(TR);
        auto VL = randomITensor(Rl1,Rl2);
        auto etaL = arnoldi(RM_T,VL,{"ErrGoal=",1E-14});
    #else
        L *= delta(Lr1,prime(Ll1));
        L *= delta(Lr2,prime(Ll2));
        auto LM = ITensorMap(L);
        auto VL = randomITensor(Ll1,Ll2);
        auto etaL = arnoldi(LM,VL,{"ErrGoal=",1E-14});
    #endif
    //
    //normalize s.t. now the dominant right eigenvalue becomes 1
    //then the dominant left eigenvalue also becomes 1
    //a transfer matrix is made up of two copies of Gamma and lambda
    Gamma /= std::sqrt(etaR);
    //lambda /= std::sqrt(etaR);
    //PrintData(etaR); PrintData(etaL); PrintData(VR);PrintData(VL);
    // normalization of eigenvector is already implemented by the eigen decomposition
    Cplx trR = eltC(VR*VR);
    Cplx trL = eltC(VL*VL);
    Cplx phaseR = trR / std::abs(trR);
    Cplx phaseL = trL / std::abs(trL);
    VR /= phaseR;
    VL /= phaseL;
    //
    #if USE_R_TRANSPOSE
        Cplx trLR = eltC(VR*VL*delta(Rr1,Rl1)*delta(Rr2,Rl2));
    #else
        Cplx trLR = eltC(VR*VL*delta(Rr1,Ll1)*delta(Rr2,Ll2));
    #endif
    //Cplx phaseLR = trLR / std::abs(trLR);
    //std::cout << "trace is " << trLR << std::endl;
    trLR = std::abs(trLR);
    VR /= std::sqrt(trLR);
    VL /= std::sqrt(trLR);
    //printf("normalized eigenvectors\n");
	//PrintData(VR); PrintData(VL);
    // eigenvalue decomposition of VR
	//Print(VR);
    //printf("starting X decomp\n");
    //PrintData(VR);
    // VR and VL are real symmetric matrices
    auto [X,X_inv] = decompV(VR);
    Index X_l(bond_dim);
    Index X_inv_r(bond_dim);
    X *= delta(X.index(1),X_l);
    X_inv *= delta(X_inv.index(1),X_inv_r);
    Index X_c = commonIndex(X,X_inv);
    //Print(X);Print(X_inv);Print(X_c);
	//PrintData(X); PrintData(X_inv);
    // now X is a matrix X_l * X_c
    // X_inv is a matrix X_c * X_inv_r
    //PrintData(X*X_inv);
	//
    //printf("starting Y decomp\n");
    //Print(VL);
    auto [Y,Y_inv] = decompV(VL);
    Index Y_r(bond_dim);
    Index Y_inv_l(bond_dim);
    Y *= delta(Y.index(1),Y_r);
    Y_inv *= delta(Y_inv.index(1),Y_inv_l);
    Index Y_c = commonIndex(Y,Y_inv);
    //Print(Y);Print(Y_inv);Print(Y_c);
	//PrintData(Y); PrintData(Y_inv);
    //
    //get the new lambda and Gamma
	//printf("lambda immediately before svd\n");PrintData(lambda);
	//printf("truncation dimension is %d\n", trunc_dim);
	//PrintData(Y * delta(Y_r,ll) * lambda * delta(lr,X_l) * X);
    //
    // different scheme
	#if 0
    auto [U,lambda_new,V] = svd((Y * delta(Y_r,ll)) * lambda * (delta(lr,X_l) * X), {Y_c},{"MaxDim=",trunc_dim,"ErrGoal=",1E-14});
	bond_dim = trunc_dim;
	//printf("bond dimension is %f\n", bond_dim);
    Index Ulink = commonIndex(U,lambda_new);
    Index Vlink = commonIndex(V,lambda_new);
    //Print(lambda_new);
    // update lambda and restore the indices
    //Print(lambda);
    // update Gamma
    ITensor Gamma_new = V * X_inv * delta(X_inv_r,Gl) * Gamma * delta(Gr,Y_inv_l) * Y_inv * U;
    // restore the indices
    Gl = Index(bond_dim,"Gl");
    Gr = Index(bond_dim,"Gr");
    Gamma = Gamma_new * delta(Vlink,Gl) * delta(Ulink,Gr);
	//Print(Gamma);
    lambda = lambda_new;
    ll = Ulink;
    lr = Vlink;
	#else
    //PrintData(lambda);
	auto [U,lambda_new,V] = svd(Y * delta(Y_r,X_l) * X , {Y_c},{"MaxDim=",trunc_dim,"MinDim=",trunc_dim});
    //PrintData(lambda_new);
	//bond_dim = lambda_new.index(1).dim();
    bond_dim = trunc_dim;
    //Print(U);Print(lambda_new);Print(V);
	//printf("\nbond dimension is%d\n", bond_dim);
	Index Ulink = commonIndex(U,lambda_new);
    Index Vlink = commonIndex(V,lambda_new);
	//
	//printf("\nget Gamma_new\n");
	//Print(Gamma);Print(lambda);
    //PrintData(V); PrintData(X_inv); PrintData(Gamma * delta(Gr,ll) * lambda); PrintData(Y_inv); PrintData(U);
	ITensor Gamma_new = V * X_inv * ((delta(X_inv_r,Gl) * Gamma * delta(Gr,ll)) * lambda) * delta(lr,Y_inv_l) * Y_inv * U;
    //PrintData(Gamma_new);
	//Print(Gamma_new);
	//std::cout << hasIndex(Gamma_new,X_c) << " " << hasIndex(Gamma_new,Y_c) << std::endl;

    Gl = Index(bond_dim,"Gl");
    Gr = Index(bond_dim,"Gr");
	Gamma = Gamma_new * delta(Vlink,Gl) * delta(Ulink,Gr);//now Gamma has index Gl Gd Gr
    //Gamma = Gamma_new;
    //Gl = Vlink;
    //Gr = Ulink;
	//Print(Gamma);
	lambda = lambda_new;
	ll = Ulink;
    lr = Vlink;
	#endif 
	//printf("ending canonicalization\n");
    //getR(); getL(); PrintData(R*delta(Rr1,Rr2)); PrintData(L*delta(Ll1,Ll2));
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
	if (!(Au.dim()==phys_dim && Ad.dim()==phys_dim)) {
        printf("\ndimension of A doesn't match\n");
        std::abort();
    }
	//
    //PrintData(Gamma);
	ITensor Gamma_new = Gamma * delta(Gd,Au) * A;
	auto[GlC,Glc] = combiner(Gl,Al);
	auto[GrC,Grc] = combiner(Gr,Ar);
	Gamma = Gamma_new * GlC * GrC;
    //Print(Gamma);
    //Gamma *= delta(Gl,Glc);
    //Gamma *= delta(Gr,Grc);
    //Gamma *= delta(Gd,Ad);
    Gl = Glc;
    Gr = Grc;
    Gd = Ad;
	//PrintData(Gamma);
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
    //PrintData(lambda);
	//printf("Gamma and lambda after combining\n");
	//Print(Gamma);Print(lambda);
	ll = llc;
	lr = lrc;
	//PrintData(lambda);
	//update bond dimension
	bond_dim *= Al.dim();
	// truncation already included in canonicalization
	canonicalize();
}

};//end of iMPS

// calculate error between two diagonal tensors with the same dimension
double getError(const ITensor& A, const ITensor& B){
    //Print(A); Print(B);
    Index a1 = A.index(1);
    Index a2 = A.index(2);
    Index b1 = B.index(1);
    Index b2 = B.index(2);
    std::vector<double> Avec = {};
    //std::vector<double> Bvec = {};
    std::vector<double> diff = {};
    int comp_dim = a1.dim() < b1.dim() ? a1.dim() : b1.dim();
    for (auto i : range1(comp_dim)){
        //printf("%d\n",i);
        double valA = elt(A,a1=i,a2=i);
        double valB = elt(B,b1=i,b2=i);
        Avec.push_back(valA);
        //Bvec.push_back(valB);
        diff.push_back(valA-valB);
    }
    Index i1 = Index(comp_dim);
    Index i2 = Index(comp_dim);
    double norm_diff = norm(diagITensor(diff,i1,i2));
    double norm_org = norm(diagITensor(Avec,i1,i2));
    //printf("norm calculated\n");
    return (norm_diff/norm_org);
}

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
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(){
    int D = 5;//bond dimension
    int max_iter_steps = 500;
    int check_err_interval = 10;
    double err_threshold = 1E-5;
    //int phys_dim = 2;
	//ITensor A = getIsingTensor();
    int phys_dim = 3;
    ITensor A = getFibonacciTensor();
    iMPS m(D,phys_dim);
    //setValue(m);
    m.canonicalize();
    printf("norm of lambda is %f\n\n", norm(m.lambda));
    //m.canonicalize();
    //PrintData(m.lambda); PrintData(m.Gamma);
    //for (auto i : range1(100)){
    //    //m.canonicalize();
    //    m.step(A);
    //    //PrintData(m.lambda);
    //}
    //m.canonicalize();
    //PrintData(m.lambda);
    //PrintData(m.lambda);
    //PrintData(m.lambda);
    #if 1
	for (auto i : range1(max_iter_steps)){
        //printf("step %d\n",i);
    	m.step(A);
        //PrintData(m.lambda);
        //printf("norm of lambda is %f\n\n", norm(m.lambda));
        //check error every 100 steps
        if(i%check_err_interval==1) {
            printf("step %d\n",i);
            ITensor lambda_previous = m.lambda;
            m.step(A); i+=1;
            ITensor lambda_current = m.lambda;
            //double err = 0.1;
            printf("Entanglement entropy is %f\n", m.calcEntropy());
            double err = getError(lambda_current,lambda_previous);
            printf("error is %f\n\n", err);
            
            if (err<err_threshold) break;
        }
    }
    #endif
}//