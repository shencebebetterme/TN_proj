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
double inv_threshold = 1E-5;//use pseudo inverse under this threshold
double chop_threshold = 1E-5;

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

void printFullDiag(const ITensor& T){
    Print(T);
    Index T1 = T.index(1);
    Index T2 = T.index(2);
    for(auto i : range1(T1.dim())){
        printf("T(%d,%d) = %.10e\n", i,i,T.eltC(T1=i,T2=i));
    }
}

void printFull4(const ITensor& T){
    Print(T);
    Index T1 = T.index(1);
    Index T2 = T.index(2);
    Index T3 = T.index(3);
    Index T4 = T.index(4);
    for(auto i : range1(T1.dim()))
        for(auto j : range1(T2.dim()))
            for(auto k : range1(T3.dim()))
                for(auto l : range1(T4.dim()))
                    printf("T(%d,%d,%d,%d) = %e;  ", i,j,k,l,T.eltC(T1=i,T2=j,T3=k,T4=l));
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

// get the sqrt of a diagonal matrix
ITensor sqrtDiag(const ITensor& T, bool keep_index){
    Index T1 = T.index(1); Index T2 = T.index(2);
    int dim = T1.dim();
    std::vector<Cplx> vec = {};
    for (auto i : range1(dim)){
        Cplx val = eltC(T,T1=i,T2=i);
        vec.push_back(val);
    }
    std::vector<Cplx> vec_sqrt = vec;
    auto sqrt_each = [](Cplx& x){ x = std::sqrt(x); };
    for_each(vec_sqrt.begin(), vec_sqrt.end(), sqrt_each);
    //
    ITensor sqrtT;
    if (keep_index) sqrtT = diagITensor(vec_sqrt,T1,T2);
    else {
        Index T1new(dim);
        Index T2new(dim);
        sqrtT = diagITensor(vec_sqrt,T1new,T2new);
    }
    return sqrtT;
}

//todo: decompV2 that returns a non-square X and X_inv
// with small eigenvalues truncated

//eigenvalue decomposition of a diagonal matrix V
// return X and X_inverse_dagger
std::tuple<ITensor,ITensor> decompV(ITensor& V){
    Index l = V.index(1);//Rr1
    Index r = V.index(2);//Rr2
    //int dim = l.dim();
    //double Vval = V.eltC(l=1,r=2).real();
    //printf("Vval is %e\n", Vval);
    //ITensor Vd = V;
    V *= delta(r,prime(l));
    //printf("\nmatrix to be decomposed is\n");
    //PrintData(Vd);
    auto [eigvec,eigval] = eigen(V);
    if (showInfo) {PrintData(eigval);PrintData(eigvec);}
    //auto [eigvec,eigval] = diagHermitian(V*delta(r,prime(l)),{"ErrGoal=",1E-14});
	//PrintData(eigval); //PrintData(eigvec);
    //chop2(eigvec);
    //chop2(eigval);
    //Link indices
    Index v2 = eigval.index(1);// primed Link
    Index v1 = eigval.index(2);// Link // commonIndex(U,D);
    int dim = v1.dim();
    std::vector<Cplx> vec = {};// store the original eigenvalues
    for (auto i : range1(dim)){
        Cplx val = eltC(eigval,v1=i,v2=i);
        if(showInfo) printf("%e\n",val);
        vec.push_back(val);
    }
    //for(auto i : vec) printf("%.10e\n",i);
    //
    auto sqrt_each = [](Cplx& x){ x = std::sqrt(x); };
    auto sqrt_inv_each = [=](Cplx& x){
        //if(std::abs(x)>inv_threshold) {x = 1/std::sqrt(x);} 
        //else {
        //    if (showInfo) printf("truncating %e to 0\n",x);
        //    x = 0;
        //}
        x = 1.0/std::sqrt(x);
    };
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
    ITensor X_inv_dag = conj(eigvec) * sqrt_inv_eigval;//X_inv_dag has index l and v2
    X_inv_dag *= delta(l,r);//X_inv_dag now has index r and v2
    //chop2(X);chop2(X_inv_dag);
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

// find the dominant eigenvector and eigenvalue of a matrix
std::tuple<ITensor, Cplx> eigenMax(const ITensor& T){
    auto [V, D] = eigen(T);
    Cplx valMax = projToMax(V,D);
    //auto vecMax = V;
    return {V,valMax};
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
    if (use_setValue) {
        for (auto i : range1(bond_dim)){
            lambda.set(ll=i,lr=i,0.3+0.5*i);
        }
        for (auto i : range1(bond_dim)){
            for (auto j : range1(phys_dim)){
                for (auto k : range1(bond_dim)){
                    Gamma.set(Gl=i,Gd=j,Gr=k, -0.3+0.5*i+0.7*j-0.9*k);
                }
            }
        }
    }
}


// Glam = Gamma * lambda
void get_Glam(){
    Glam = Gamma * delta(Gr,ll) * lambda;
    Glam_l = Gl; Glam_r = lr; Glam_d = Gd;
}

// Glam = sqrt(lambda) * Gamma * sqrt(lambda)
void get_lamGlam(){
    ITensor sqrt_lambda1 = sqrtDiag(lambda,false);
    Index sl1 = sqrt_lambda1.index(1); Index sl2 = sqrt_lambda1.index(2);
    ITensor sqrt_lambda2 = sqrtDiag(lambda,false);
    Index sr1 = sqrt_lambda2.index(1); Index sr2 = sqrt_lambda2.index(2);
    Glam = sqrt_lambda1 * (delta(sl2,Gl) * Gamma * delta(Gr,sr1)) * sqrt_lambda2;
    Glam_l = sl1; Glam_r = sr2; Glam_d = Gd;//change indices of Glam
}

//obtain R from Glam
void getR(){
    if (!(bond_dim==Glam_l.dim() && bond_dim==Glam_r.dim())) printf("\n mismatching dimension!\n");
    Rl1 = Index(bond_dim,"Rl1");
    Rl2 = Index(bond_dim,"Rl2");
    Rr1 = Index(bond_dim,"Rr1");
    Rr2 = Index(bond_dim,"Rr2");
    R = (delta(Rl1,Glam_l) * Glam * delta(Glam_r,Rr1)) * (delta(Rl2,Glam_l) * conj(Glam) * delta(Glam_r,Rr2));
    //R += 1E-10 * randomITensor(Rl1,Rl2,Rr1,Rr2);
}

//canonicalize Glam to obtain new Gamma and lambda
//equivalent to normalization + A_to_gamma
void canonicalizeBase(){
    if(showInfo) printf("begin conanicalization base\n");
    this -> getR();
    //chop4(R);
    //printFull4(R);
    ITensor L = conj(R);//make a copy of R
    // obtain VR and VL
    if(showInfo) Print(R);
    R *= delta(Rl1,prime(Rr1));
    R *= delta(Rl2,prime(Rr2));
    auto [VR, etaR] = eigenMax(R);
    
    L *= delta(Rr1,prime(Rl1));
    L *= delta(Rr2,prime(Rl2));
    auto [VL, etaL] = eigenMax(L);
   
    // normalization
    if(showInfo) printf("normalization\n");
    Glam /= std::sqrt(etaR);
    Cplx trR = eltC(VR*delta(Rr1,Rr2));
    Cplx trL = eltC(VL*delta(Rl1,Rl2));
    Cplx phaseR = trR / std::abs(trR);
    Cplx phaseL = trL / std::abs(trL);
    VR /= phaseR;
    VL /= phaseL;
    Cplx trLR = eltC(conj(VR)*VL*delta(Rr1,Rl1)*delta(Rr2,Rl2));
    trLR = std::abs(trLR);
    if(showInfo) printf("\ntrLR = %e\n",trLR);
    VR /= std::sqrt(trLR);
    VL /= std::sqrt(trLR);
   
    if(showInfo) printf("\nobtain X and X_inv\n");
    auto [X,X_inv] = decompV(VR);
    Index X_l(bond_dim);
    Index X_inv_r(bond_dim);
    X *= delta(X.index(1),X_l);
    X_inv *= delta(X_inv.index(1),X_inv_r);
    Index X_c = commonIndex(X,X_inv);
    printf("\n");
    // obtain Y and Y_inv
    if(showInfo) printf("\nobtain Y and Y_inv\n");
    auto [Y,Y_inv] = decompV(VL);
    Index Y_r(bond_dim);
    Index Y_inv_l(bond_dim);
    Y *= delta(Y.index(1),Y_r);
    Y_inv *= delta(Y_inv.index(1),Y_inv_l);
    Index Y_c = commonIndex(Y,Y_inv);
   
    //if(showInfo) {PrintData(Y * delta(Y_r,X_l) * X);}
    auto [U,lambda_new,V] = svd(conj(Y) * delta(Y_r,X_l) * X, {Y_c},{"MaxDim=",trunc_dim,"MinDim=",trunc_dim});
    //
    //chop2(U);chop2(V);
    //
    //if(showInfo) {printf("\nlambda_new = \n");printFullDiag(lambda_new);}
    Index Ulink = commonIndex(U,lambda_new);
    Index Vlink = commonIndex(V,lambda_new);
    bond_dim = trunc_dim;
    // get new Gamma
    //if(showInfo) PrintData(Glam);
    ITensor Gamma_new = V * X_inv * delta(X_inv_r,Glam_l) * Glam * delta(Glam_r,Y_inv_l) * Y_inv * U;
    //if(showInfo) PrintData(Gamma_new);
    // restore indices of Gamma and lambda
    Gl = Index(bond_dim,"Gl");
    Gr = Index(bond_dim,"Gr");
    Gd = Index(phys_dim,"Gd");
	Gamma = Gamma_new * delta(Vlink,Gl) * delta(Ulink,Gr) * delta(Glam_d,Gd);//now Gamma has index Gl Gd Gr
    //chop3(Gamma,1E-15);
    lambda = lambda_new;
    //lambda = toDense(lambda); chop2(lambda);
	ll = Ulink;
    lr = Vlink;
}

void canonicalize(){
    this -> get_Glam();
    this -> canonicalizeBase();
}

//obtain Glam from Gamma,lambda and do canonicalization
void step(const ITensor& H){
    if(showInfo) {printf("beginning step\n");Print(lambda);Print(Gamma);}
    Index Hl = H.index(1);
    Index Hu = H.index(2);
    Index Hr = H.index(3);
    Index Hd = H.index(4);
    if (!(Hu.dim()==phys_dim && Hd.dim()==phys_dim)) {
        printf("\ndimension of H doesn't match\n");
        std::abort();
    }
    if (hasIndex(Gamma,Hd)){
        printf("Gamma has repeated Hd index\n");
        std::abort();
    }
    //if(showInfo) printf("before contracting lambda\n");PrintData(Gamma);
    // Glam = sqrt(lambda) * Gamma * sqrt(lambda)
    this -> get_lamGlam();
    // contract with H
    //if(showInfo) {printf("before contracting MPO\n");Print(sl1);Print(sr2);PrintData(Glam);}
    Glam = Glam * delta(Glam_d,Hu) * H;
    //Print(Glam);Print(Glam_l);Print(Glam_r);
    auto [GlC,Glc] = combiner(Glam_l,Hl);
    auto [GrC,Grc] = combiner(Glam_r,Hr);
    //Print(GlC);Print(GrC);
    Glam *= GlC;
    Glam *= GrC;
    bond_dim = Glc.dim();
    //if(showInfo) {printf("after contracting MPO\n");Print(Glc);Print(Grc);PrintData(Glam);}
    // restore indices
    Glam_l = Glc;
    Glam_r = Grc;
    Glam_d = Hd;
    this -> canonicalizeBase();
}

};//end of iMPS



int main(int argc, char* argv[]){
    if (argc==2) {
        iter_steps = atoi(argv[1]);
    }
    showInfo = false;
    int D = 8;
    int phys_dim = 3;
    ITensor H = getFibonacciTensor();
    iMPS mps(D,phys_dim,false);
    if(showInfo) {PrintData(mps.lambda);PrintData(mps.Gamma);}
    mps.canonicalize();
    printFullDiag(mps.lambda);
    //mps.canonicalize();
    //printFullDiag(mps.lambda);
    //mps.canonicalize();
    //PrintData(mps.lambda);PrintData(mps.Gamma);
    #if 1
    for (auto i : range1(iter_steps)){
        printf("\n\n\n================================================\n");
        printf("step %d\n",i);
        //mps.step(H);
        mps.step(H);
        printFullDiag(mps.lambda);
        //double vall = mps.lambda.eltC(mps.ll=1,mps.lr=1).real();
        //double valr = mps.lambda.eltC(mps.ll=2,mps.lr=2).real();
        //printFullDiag(mps.lambda);
        //printf("lambda_11 is %e\n", vall);
        //printf("lambda_22 is %e\n", valr);
    }
    #endif
}