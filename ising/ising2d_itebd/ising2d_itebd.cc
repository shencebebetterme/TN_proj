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

//eigenvalue decomposition of a diagonal matrix V
// return X and X_inverse_dagger
std::tuple<ITensor,ITensor> decompV(const ITensor& V){
    Index l = V.index(1);//Rr1
    Index r = V.index(2);//Rr2
    //int dim = l.dim();
    double Vval = V.eltC(l=1,r=2).real();
    //printf("Vval is %.18f\n", Vval);
    ITensor Vd = V;
    Vd *= delta(r,prime(l));
    //printf("\nmatrix to be decomposed is\n");
    //PrintData(Vd);
    auto [eigvec,eigval] = eigen(Vd);
    Index d1 = eigval.index(1);
    Index d2 = eigval.index(2);
    Cplx val1 = eigval.eltC(d1=1, d2=1);
    Cplx val2 = eigval.eltC(d1=2, d2=2);
    if (showInfo) {printf("in decompV, val1 is %.18f\n",val1); printf("val2 is %.18f\n",val2);}
    //auto [eigvec,eigval] = diagHermitian(V*delta(r,prime(l)),{"ErrGoal=",1E-14});
	//PrintData(eigval); //PrintData(eigvec);
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

// get the sqrt of a diagonal matrix
ITensor getSqrt(const ITensor& T){
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
ITensor getId(int n){
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
Index lamGlam_l, lamGlam_r, lamGlam_d;
ITensor lamGlam; //contraction of sqrt(lambda) * Gamma * sqrt(lambda)
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

// get the R and L transfer matrix
void getR(){
	Rl1 = Index(bond_dim,"Rl1");
    Rl2 = Index(bond_dim,"Rl2");
    Rr1 = Index(bond_dim,"Rr1");
    Rr2 = Index(bond_dim,"Rr2");
    // upper part has indices Rl1, Gd, Rl2
	//Print(Gamma); Print(lambda);
    ITensor upper_part = ((delta(Rl1,Gl) * Gamma * delta(Gr,ll)) * lambda) * delta(lr,Rr1);
    //ITensor upper_part = delta(Rl1,Gl) * Gamma * delta(Gr,Rr1);
    // lower part has indices Rl2, Gd, Rr2
    ITensor lower_part = ((delta(Rl2,Gl) * conj(Gamma) * delta(Gr,ll)) * conj(lambda)) * delta(lr,Rr2);
    //ITensor lower_part = delta(Rl2,Gl) * Gamma * delta(Gr,Rr2);
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
    ITensor upper_part = delta(Ll1,ll) * (lambda * (delta(lr,Gl) * Gamma * delta(Gr,Lr1)));
    // lower part has indices Ll2, Gd, Lr2
    ITensor lower_part = delta(Ll2,ll) * (conj(lambda) * (delta(lr,Gl) * conj(Gamma) * delta(Gr,Lr2)));
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
    //this -> getL();
    //PrintData(R);
    ITensor R_copy = R;
    double valRR = R.eltC(Rl1=2,Rl2=1,Rr1=1,Rr2=1).real();
    //printf("valRR is %.18f\n",valRR);
    //PrintData(R); PrintData(L);
    R *= delta(Rl1,prime(Rr1));
    R *= delta(Rl2,prime(Rr2));
    auto [VR, eigvalR] = eigen(R);
    Cplx etaR = projToMax(VR,eigvalR);
    // use transpoes(R) to get the left eigenvalue of L
    #define USE_R_TRANSPOSE 1
    #if USE_R_TRANSPOSE
        ITensor L = R_copy;
        L *= delta(Rr1,prime(Rl1));
        L *= delta(Rr2,prime(Rl2));//now TR has indices Rl1,2 and primed Rl1,2
        auto [VL, eigvalL] = eigen(L);
        Cplx etaL = projToMax(VL,eigvalL);
    #else
        L *= delta(Lr1,prime(Ll1));
        L *= delta(Lr2,prime(Ll2));
        auto [VL, eigvalL] = eigen(L);
        Cplx etaL = projToMax(VL,eigvalL);
    #endif
    //
    //normalize s.t. now the dominant right eigenvalue becomes 1
    //then the dominant left eigenvalue also becomes 1
    //a transfer matrix is made up of two copies of Gamma and lambda
    Gamma /= std::sqrt(etaR);
    //lambda /= std::sqrt(etaR);
    //PrintData(etaR); PrintData(etaL); PrintData(VR);PrintData(VL);
    double valR = etaR.real();
    double valL = etaL.real();
    if(showInfo){
    printf("valR is %.18f\n",valR); printf("valL is %.18f\n\n",valL);
    printf("VR11 = %.18f\n",VR.eltC(VR.index(1)=1,VR.index(2)=1));
    printf("VR12 = %.18f\n",VR.eltC(VR.index(1)=1,VR.index(2)=2));
    printf("VR21 = %.18f\n",VR.eltC(VR.index(1)=2,VR.index(2)=1));
    printf("VR22 = %.18f\n",VR.eltC(VR.index(1)=2,VR.index(2)=2));
    PrintData(VR);PrintData(VL);
    }
    // normalization of eigenvector is already implemented by the eigen decomposition
    Cplx trR = eltC(VR*delta(Rr1,Rr2));
    #if USE_R_TRANSPOSE
        Cplx trL = eltC(VL*delta(Rl1,Rl2));
    #else 
        Cplx trL = eltC(VL*delta(Ll1,Ll2));
    #endif
    Cplx phaseR = trR / std::abs(trR);
    Cplx phaseL = trL / std::abs(trL);
    if (showInfo){
    printf("trR is %.18f\n",trR); printf("trL is %.18f\n",trL); 
    printf("phaseR is %.18f\n",phaseR); printf("phaseL is %.18f\n",phaseL); 
    }
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
    if(showInfo){
    printf("trLR is %.18f\n\n",trLR);
    printf("after normalization\n");
    printf("VR11 = %.18f\n",VR.eltC(VR.index(1)=1,VR.index(2)=1));
    printf("VR12 = %.18f\n",VR.eltC(VR.index(1)=1,VR.index(2)=2));
    printf("VR21 = %.18f\n",VR.eltC(VR.index(1)=2,VR.index(2)=1));
    printf("VR22 = %.18f\n",VR.eltC(VR.index(1)=2,VR.index(2)=2));
    PrintData(VR);PrintData(VL);
    }
    //printf("\nnormalized eigenvectors\n");
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
    //ITensor Gamma_new = V * X_inv * delta(X_inv_r,Gl) * Gamma  * delta(Gr,Y_inv_l) * Y_inv * U;
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
    getR(); getL(); PrintData(R*delta(Rr1,Rr2)); //PrintData(L*delta(Ll1,Ll2));
}

//update Gamma and lambda after one layer of MPO
// the indices of A are ordered as Al, Au, Ar, Ad
// suppose dim(Au) = dim(Ad) = phys_dim, and dim(Al) = dim(Ar)
void step(const ITensor& A){
	if(showInfo) printf("\nstarting step\n");
    Index Al = A.index(1);
    Index Au = A.index(2);
    Index Ar = A.index(3);
    Index Ad = A.index(4);
	if (!(Au.dim()==phys_dim && Ad.dim()==phys_dim)) {
        printf("\ndimension of A doesn't match\n");
        std::abort();
    }
	//
    if(showInfo) PrintData(Gamma);
	ITensor Gamma_new = Gamma * delta(Gd,Au) * A;
    //if(showInfo) PrintData(Gamma);PrintData(Gamma_new);
	auto[GlC,Glc] = combiner(Gl,Al);
	auto[GrC,Grc] = combiner(Gr,Ar);
	Gamma = Gamma_new * GlC * GrC;
    if(showInfo) PrintData(Gamma);
    //Print(Gamma);
    //Gamma *= delta(Gl,Glc);
    //Gamma *= delta(Gr,Grc);
    //Gamma *= delta(Gd,Ad);
    Gl = Glc;
    Gr = Grc;
    Gd = Ad;
    //printf("gamma after contracting MPO is\n"); PrintData(Gamma);
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
    if(showInfo) PrintData(Gamma*lambda*delta(Gr,ll));
	// truncation already included in canonicalization
	canonicalize();
}

void step2(const ITensor& A){
    if(showInfo) printf("\nstarting step\n");
    Index Al = A.index(1);
    Index Au = A.index(2);
    Index Ar = A.index(3);
    Index Ad = A.index(4);
	if (!(Au.dim()==phys_dim && Ad.dim()==phys_dim)) {
        printf("\ndimension of A doesn't match\n");
        std::abort();
    }
    ITensor sqrt_lambda = getSqrt(lambda);
    //construct upper part
    Index sl1 = sqrt_lambda.index(1);
    Index sl2 = sqrt_lambda.index(2);
    Gamma = Gamma * sqrt_lambda * delta(sl2,Gl);
    Gamma *= delta(sl1,Gl);//change left index to Gl again
    Gamma = Gamma * sqrt_lambda * delta(sl1,Gr);
    Gamma *= delta(sl2,Gr);//change right index to Gr again
    // MPO contraction
    Gamma = Gamma * A * delta(Gd,Au);
    auto[GlC,Glc] = combiner(Gl,Al);
	auto[GrC,Grc] = combiner(Gr,Ar);
	Gamma *= GlC;
    Gamma *= GrC;
    Gl = Glc;
    Gr = Grc;
    Gd = Ad;
    if(showInfo) PrintData(Gamma);
    //
    bond_dim *= Al.dim();
    if (!(Gl.dim()==bond_dim && Gr.dim()==bond_dim)){
        printf("\ndimension of Gl and Gr incorrect\n");
        std::abort();
    }
    //
    lambda = getId(bond_dim);
    printf("lambda created\n");
    ll = lambda.index(1);
    lr = lambda.index(2);
    //
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
int main(int argc, char* argv[]){
    if (argc==2) {
        iter_steps = atoi(argv[1]);
    }
    showInfo = true;
    int D = 2;//bond dimension
    int check_err_interval = 10;
    double err_threshold = 1E-5;
    //int phys_dim = 2;
	//ITensor A = getIsingTensor();
    int phys_dim = 3;
    ITensor H = getFibonacciTensor();
    iMPS mps(D,phys_dim);
    setValue(mps);
    PrintData(mps.lambda); PrintData(mps.Gamma);
    mps.canonicalize();
    PrintData(mps.lambda); PrintData(mps.Gamma);
    //
    double vall = mps.lambda.eltC(mps.ll=1,mps.lr=1).real();
    double valr = mps.lambda.eltC(mps.ll=2,mps.lr=2).real();
    printf("vall is %.18f\n", vall);
    printf("valr is %.18f\n", valr);

    #if 1
    for (auto i : range1(iter_steps)){
        //showInfo = true;
        //if(i>=3) showInfo = true;
    printf("\n\n\n================================================\n");
    printf("step %d\n",i);
    mps.step2(H);
    double vall = mps.lambda.eltC(mps.ll=1,mps.lr=1).real();
    double valr = mps.lambda.eltC(mps.ll=2,mps.lr=2).real();
    PrintData(mps.lambda);
    printf("lambda_11 is %.18f\n", vall);
    printf("lambda_22 is %.18f\n", valr);
    //PrintData(mps.lambda); //PrintData(mps.Gamma);
    }
    #endif
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
    #if 0
	for (auto i : range1(max_iter_steps)){
        //printf("step %d\n",i);
    	mps.step(H);
        //PrintData(m.lambda);
        //printf("norm of lambda is %f\n\n", norm(m.lambda));
        //check error every 100 steps
        if(i%check_err_interval==1) {
            printf("step %d\n",i);
            ITensor lambda_previous = mps.lambda;
            mps.step(H); i+=1;
            ITensor lambda_current = mps.lambda;
            //double err = 0.1;
            printf("Entanglement entropy is %f\n", mps.calcEntropy());
            double err = getError(lambda_current,lambda_previous);
            printf("error is %f\n\n", err);
            
            if (err<err_threshold) break;
        }
    }
    #endif
}//