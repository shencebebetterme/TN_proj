// make && ./ising_arma
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


#define PI 3.1415926535
#define USE_MAINTEXT_CONVENTION 1

using namespace itensor;
using namespace std;
using namespace std::chrono;
using namespace arma;

double phi = 0.5*(1+sqrt(5));
int dim0 = 2;//initial A tensor leg dimension
 int new_dim0 = 4;
int len_chain = 5; // the actual length is len_chain + 1
int len_chain_L = 3;
int len_chain_R = 3;
//int period = len_chain + 1;
int num_states = 20;// final number of dots in the momentum diagram



std::vector<int> legToLegs(const int& leg_val, const int& num_inds, const int& dim);
int legsToLeg(const std::vector<int>& vec, const int& num_inds, const int& dim);
template<typename T>
void permute(std::vector<T>& vec);
int ind_translate(const int& input, const int& num_inds, const int& dim);


// convert ITensor matrix to arma matrix
arma::mat extract_mat(const ITensor& T);
arma::sp_mat extract_sparse_mat_twisted(const ITensor& T);

void extract_cft_data(const arma::sp_mat& TM);

//the F symbol value F^{tau tau x}_{y tar z}
double F(int x, int y, int z){
    if ((x==1) && (y==2) && (z==2)) return 1.0;
    if ((x==2) && (y==1) && (z==2)) return 1.0;
    if ((x==2) && (y==2) && (z==1)) return 1/sqrt(phi);
    if ((x==2) && (y==2) && (z==2)) return -1/phi;
    return 0;
}

// quantum dim
double qd(int x){
    if (x==1) return 1.0;
    if (x==2) return phi;
}


double test(const arma::sp_mat& M){
    return M(1,1);
}


// in {a,b,c} one is 1 and the other two are 2
bool oneIs1(int a, int b, int c){
    if ((a==1)&&(b==2)&&(c==2)) {return true;}
    else if ((a==2)&&(b==1)&&(c==2)) {return true;}
    else if ((a==2)&&(b==2)&&(c==1)) {return true;}
    else return false;
}

int main(int argc, char* argv[]){
    if (argc==2) {
        len_chain = atoi(argv[1]);
    }
    // decompose the chain into  two parts
    // stupid way to reduce memory usage and avoid
    // lapack 32bit 64vit dgemm parameter 3 issue
    if (len_chain%2==0) {
        len_chain_L = len_chain_R = len_chain/2;
    }
    else
    {
        len_chain_L = (len_chain-1)/2;
        len_chain_R = len_chain - len_chain_L;
    }
    
    //todo: set the tensor value to that in the main text
    // build A tensor
    #if USE_MAINTEXT_CONVENTION
    Index alpha1(dim0);
    Index alpha2(dim0);
    Index beta1(dim0);
    Index beta2(dim0);
    Index gamma1(dim0);
    Index gamma2(dim0);
    ITensor A(alpha1,alpha2,beta1,beta2,gamma1,gamma2);
    for(auto sa1 : range1(dim0)){
        for(auto sa2 : range1(dim0)){
            for(auto sb1 : range1(dim0)){
                for(auto sb2 : range1(dim0)){
                    for(auto sg1 : range1(dim0)){
                        for(auto sg2 : range1(dim0)){
                            if((sa1==sa2) && (sb1==sb2) && (sg1==sg2)){
                                int alpha = sa1;
                                int beta = sb1;
                                int gamma = sg1;
                                //double val = std::pow(phi,0.25) / sqrt(qd(beta)) * F(gamma,alpha,beta);
                                if ((alpha==2)&&(beta==2)&&(gamma==2)){
                                    double val = -1/std::pow(phi,1.0/4);
                                    A.set(alpha1=sa1,alpha2=sa2,beta1=sb1,beta2=sb2,gamma1=sg1,gamma2=sg2,val);
                                }
                                else if (oneIs1(alpha,beta,gamma)){
                                    double val = std::pow(phi,7.0/12);
                                    A.set(alpha1=sa1,alpha2=sa2,beta1=sb1,beta2=sb2,gamma1=sg1,gamma2=sg2,val);
                                }
                                else{
                                    A.set(alpha1=sa1,alpha2=sa2,beta1=sb1,beta2=sb2,gamma1=sg1,gamma2=sg2,0);
                                }
                            }
                            else{
                                A.set(alpha1=sa1,alpha2=sa2,beta1=sb1,beta2=sb2,gamma1=sg1,gamma2=sg2,0);
                            }
                        }
                    }
                }
            }
        }
    }
    #endif
    PrintData(A);
    //build B tensor
    #if USE_MAINTEXT_CONVENTION
    Index mu1(dim0);
    Index mu2(dim0);
    Index rho1(dim0);
    Index rho2(dim0);
    Index nu1(dim0);
    Index nu2(dim0);
    ITensor B(mu1,mu2,rho1,rho2,nu1,nu2);
    for(auto sm1 : range1(dim0)){
        for(auto sm2 : range1(dim0)){
            for(auto sr1 : range1(dim0)){
                for(auto sr2 : range1(dim0)){
                    for(auto sn1 : range1(dim0)){
                        for(auto sn2 : range1(dim0)){
                            if((sm1==sm2) && (sr1==sr2) && (sn1==sn2)){
                                int mu = sm1;
                                int rho = sr1;
                                int nu = sn1;
                                //double val = std::pow(phi,0.25) / sqrt(qd(nu)) * F(rho,mu,nu);
                                //B.set(mu1=sm1,mu2=sm2,rho1=sr1,rho2=sr2,nu1=sn1,nu2=sn2,val);
                                 if ((mu==2)&&(rho==2)&&(nu==2)){
                                    double val = -1/-1/std::pow(phi,1.0/4);
                                    B.set(mu1=sm1,mu2=sm2,rho1=sr1,rho2=sr2,nu1=sn1,nu2=sn2,val);
                                }
                                else if (oneIs1(mu,rho,nu)){
                                    double val = std::pow(phi,7.0/12);
                                   B.set(mu1=sm1,mu2=sm2,rho1=sr1,rho2=sr2,nu1=sn1,nu2=sn2,val);
                                }
                                else{
                                    B.set(mu1=sm1,mu2=sm2,rho1=sr1,rho2=sr2,nu1=sn1,nu2=sn2,0);
                                }
                            }
                            else{
                                B.set(mu1=sm1,mu2=sm2,rho1=sr1,rho2=sr2,nu1=sn1,nu2=sn2,0);
                            }
                        }
                    }
                }
            }
        }
    }
    #endif
    PrintData(B);

    //build the C tensor, the building block of transfer matrix
    ITensor C = A * B * delta(beta2,mu1) * delta(gamma1,nu2);
    //PrintData(C);
    // combine legs to form a 4 leg tensor C
    auto [uT,u] = combiner(alpha2,beta1);
    auto [rT,r] = combiner(mu2,rho1);
    auto [dT,d] = combiner(nu1,rho2);
    auto [lT,l] = combiner(alpha1,gamma2);
    C *= uT;
    C *= rT;
    C *= dT;
    C *= lT;
    PrintData(C);
    // now C is a 4-leg tensor, each index has dim = 4
    new_dim0 = 4;
    double TrC = elt(C * delta(l, r) * delta(u, d));
    Print(TrC);
    C /= TrC;



//////////////////////////////////////////////////////////////////////////////
    // build left and right transfer matrix independently
    // and contract them in the end
    Index u0L = addTags(u,"L,u_i="+str(0));
    Index d0L = addTags(d,"L,d_i="+str(0));
    Index l0L = addTags(l,"L,site="+str(0));
    Index r0L = addTags(r,"L,site="+str(0));
    Index u0R = addTags(u,"R,u_i=" +str(len_chain_L));
    Index d0R = addTags(d,"R,d_i=" +str(len_chain_L));
    Index l0R = addTags(l,"R,site="+str(len_chain_L));
    Index r0R = addTags(r,"R,site="+str(len_chain_L));
    //initial TM tensor
    ITensor ML = C * delta(u,u0L) * delta(d,d0L) * delta(l,l0L) * delta(r,r0L);
    ITensor MR = C * delta(u,u0R) * delta(d,d0R) * delta(l,l0R) * delta(r,r0R);
    // indexset for decomposition
    std::vector<Index> upL_inds_vec = {u0L};
    std::vector<Index> downL_inds_vec = {d0L};
    std::vector<Index> upR_inds_vec = {u0R};
    std::vector<Index> downR_inds_vec = {d0R};

    for (int i : range1(len_chain_L-1)){
        Index uiL = addTags(u,"L,u_i="+str(i));
        Index diL = addTags(d,"L,d_i="+str(i));
        Index liL = addTags(l,"L,site="+str(i));
        Index riL = addTags(r,"L,site="+str(i));
        //construct the i-th copy of A tensor
        ITensor CiL = C * delta(u,uiL) * delta(d,diL) * delta(r,riL) * delta(l,liL);
        Index previous_riL = addTags(r,"L,site="+str(i-1));
        printf("eating %dth A tensor, left\n",i);
        ML *= (CiL * delta(previous_riL, liL));
        //TM.swapTags("d_i="+str(i),"d_i="+str(i-1));
        upL_inds_vec.push_back(uiL);
        downL_inds_vec.push_back(diL);
    }
    for (int i : range1(len_chain_R-1)){
        Index uiR = addTags(u,"R,u_i="+ str(i+len_chain_L));
        Index diR = addTags(d,"R,d_i="+ str(i+len_chain_L));
        Index liR = addTags(l,"R,site="+str(i+len_chain_L));
        Index riR = addTags(r,"R,site="+str(i+len_chain_L));
        //construct the i-th copy of A tensor
        ITensor CiR = C * delta(u,uiR) * delta(d,diR) * delta(r,riR) * delta(l,liR);
        Index previous_riR = addTags(r,"R,site="+str(i-1+len_chain_L));
        printf("eating %dth A tensor, right\n",i);
        MR *= (CiR * delta(previous_riR, liR));
        //TM.swapTags("d_i="+str(i),"d_i="+str(i-1));
        upR_inds_vec.push_back(uiR);
        downR_inds_vec.push_back(diR);
    }

    Index L_rightmost = addTags(r,"L,site="+str(len_chain_L-1));
    Index R_rightmost = addTags(r,"R,site="+str(len_chain_L+len_chain_R-1));
    printf("\ncombining left and right\n\n");
    // use reference and *= to reduce memory usage
    ITensor& M = ML;
    M *= MR * delta(L_rightmost,l0R) * delta(l0L,R_rightmost);
    Print(M);

    //group up and down indices
    std::vector<Index>& up_inds = upL_inds_vec;
    up_inds.insert(up_inds.end(), upR_inds_vec.begin(), upR_inds_vec.end());
    std::vector<Index>& down_inds = downL_inds_vec;
    down_inds.insert(down_inds.end(), downR_inds_vec.begin(), downR_inds_vec.end());
    // combine the up and down indices into uL and dL
    auto[uLC,uL] = combiner(up_inds);
    auto[dLC,dL] = combiner(down_inds);
    M *= uLC;
    M *= dLC;
    Print(M);

    //for(auto it : iterInds(M)){
    //    Print(it[0]);
    //    IndexVal iv = it[0];
    //    int i = iv.val;
    //    Print(it[1]);
    //    Print(M.real(it));
    //    println();
    //}
    
    // M is extremely sparse, so choose a different strategy
    arma::sp_mat TM_sparse = extract_sparse_mat_twisted(M);
    //TM_sparse.print();
    extract_cft_data(TM_sparse);
    //TM_sparse.print();
    //arma::sp_mat H = sprandu(2,2,0.5);
    //double sss = test(H);
    return 0;
}



 // M is extremely sparse, so choose a different strategy
arma::sp_mat extract_sparse_mat_twisted(const ITensor& T){
    // here T is sparse matrix ITensor
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    arma::sp_mat Tmat(di,dj);
    //
    for (auto it : iterInds(T)){
        if (T.real(it) != 0){
            IndexVal iv_i = it[0];
            IndexVal iv_j = it[1];
            int i = iv_i.val -1;
            int j = iv_j.val -1;
            int twisted_j = ind_translate(j-1,len_chain,new_dim0)+1;
            Tmat(i,j) = T.real(it);
        }
    }
    return Tmat;
}


void extract_cft_data(const arma::sp_mat& TM_sparse){
    //arma::sp_mat TM_sparse(TM);
    int nT = TM_sparse.n_rows;
    if(nT<num_states){
        printf("\nToo many states requested!\n");
    }
    // eigen
    printf("\nStarting eigenvalue decomposition\n\n");
    auto eig_start = high_resolution_clock::now();
    cx_vec eigval = eigs_gen(TM_sparse,num_states,"lm",0.0001);
    auto eig_stop = high_resolution_clock::now();
    auto eig_duration = duration_cast<seconds>(eig_stop-eig_start).count();
    printf("Eigenvalue decomposition finished, lasting %d seconds\n\n",eig_duration);
    //eigval.print("eigval=");
    //eigval.print();
    std::vector<double> log_lambdas = {};
    std::vector<double> scaling_dims = {};
    std::vector<double> spins = {};
    //obtain spins
    for (auto lambda_i_tilde : eigval){
        //eigen_vals.push_back(lambda_i_tilde);
        Cplx log_lambda_i_tilde = std::log(lambda_i_tilde);
        log_lambdas.push_back(log_lambda_i_tilde.real());
        spins.push_back(len_chain/(2*PI)*log_lambda_i_tilde.imag());
    }
    //obtain scaling dimensions
    double log_lambda_max = log_lambdas[0];
    for (const auto& log_lambda_i : log_lambdas){
        double i_th_scaling_dim = -len_chain/(2*PI)*(log_lambda_i-log_lambda_max);
        scaling_dims.push_back(i_th_scaling_dim);
    }
    //print
    for(int i = 0; i<num_states; i++){
        printf("%6.6f\t%6.6f\n",scaling_dims[i],spins[i]);
    }
}


// the inverse process of combining indices
// e.g.,  len_val =10, num_inds=4, dim=2, then 10=8+2 -> {1,0,1,0}
std::vector<int> legToLegs(const int& leg_val, const int& num_inds, const int& dim){
    //int num_inds = inds.size();
    //leg_val is 0-based
    if (leg_val > std::pow(dim,num_inds)-1) print("\nleg_val is too large for the Legs set\n");
    //initial vector
    std::vector<int> result = {};
    int val = leg_val;
    while (val>=dim)
    {
        int quotient = val/dim;
        int remainder = val%dim;
        result.push_back(remainder);
        val = quotient;
    }
    result.push_back(val);
    // 0 padding
    int len = result.size();
    result.insert(result.end(), num_inds-len, 0);
    //std::reverse(result.begin(), result.end());
    return result;
}

int legsToLeg(const std::vector<int>& vec, const int& num_inds, const int& dim){
    int leg_val = 0;
    if (vec.size()!=num_inds) printf("\ndimension doesn't match\n");
    for (int i=0; i<num_inds; i++){
        leg_val += vec[i]*std::pow(dim,i);
    }
    return leg_val;
}

//permute to the right
template<typename T>
void permute(std::vector<T>& vec){
    int len = vec.size();
    //inline int next(int i) {return (i+1) % len;};
    T last = vec[len-1];
    for (auto i=len-1; i>0; i--){
        vec[i] = vec[i-1];
    }
    vec[0] = last;
}

int ind_translate(const int& input, const int& num_inds, const int& dim){
    auto vec = legToLegs(input, num_inds, dim);
    permute(vec);
    auto output = legsToLeg(vec, num_inds, dim);
    //printf("%d translated to %d\n", input, output);
    return output;
}

