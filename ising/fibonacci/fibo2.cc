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
int dim0 = 3;//initial A tensor leg dimension
//int new_dim0 = 4;
int len_chain = 5; // the actual length is len_chain + 1
int len_chain_L = 3;
int len_chain_R = 3;
//int period = len_chain + 1;
int num_states = 10;// final number of dots in the momentum diagram



std::vector<int> legToLegs(const int& leg_val, const int& num_inds, const int& dim);
int legsToLeg(const std::vector<int>& vec, const int& num_inds, const int& dim);
template<typename T>
void permute(std::vector<T>& vec);
int ind_translate(const int& input, const int& num_inds, const int& dim);


// convert ITensor matrix to arma matrix
arma::mat extract_mat(const ITensor& T);
arma::sp_mat extract_sparse_mat(const ITensor& T, bool twisted=false);

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
    printf("\nlength of chain is %d\n\n", len_chain);
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
    
    
    Index u(dim0);
    Index r(dim0);
    Index d(dim0);
    Index l(dim0);
    Print(u);
    Print(r);
    Print(d);
    Print(l);
    ITensor C(u,r,d,l);
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
    PrintData(C);


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
    //initial M tensor
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
        //printf("eating %dth A tensor, left\n",i);
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
        //printf("eating %dth A tensor, right\n",i);
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
    // now M is a matrix
    ITensor& Mmat = M;
    printf("the transfer matrix M is\n");
    //PrintData(Mmat);

    //for(auto it : iterInds(M)){
    //    Print(it[0]);
    //    IndexVal iv = it[0];
    //    int i = iv.val;
    //    Print(it[1]);
    //    Print(M.real(it));
    //    println();
    //}
    //PrintData(M);
    // M is extremely sparse, so choose a different strategy
    arma::sp_mat TM_sparse = extract_sparse_mat(Mmat,false);
    //TM_sparse.print("\n\nT.M is ");
    //TM_sparse.print();
    arma::sp_mat TM2 = TM_sparse * (TM_sparse.t());
    TM2.print("TM2");
    extract_cft_data(TM2);
    //TM_sparse.print();
    return 0;
}



 // M is extremely sparse, so choose a different strategy
arma::sp_mat extract_sparse_mat(const ITensor& T, bool twisted=false){
    // here T is sparse matrix ITensor
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    arma::sp_mat Tmat(di,dj);
    //
    for (auto it : iterInds(T)){
        if (T.real(it) != 0){
            IndexVal iv_i = it[0];//iv_i is 1-based
            IndexVal iv_j = it[1];
            int i = iv_i.val -1;// i is 0-based
            int j = iv_j.val -1;
            // the parameter leg_val is 0-based
            if (twisted) {
                int twisted_j = ind_translate(j,len_chain,dim0);
                printf("j is %d, twisted_j is %d\n", j, twisted_j);
                Tmat(i,twisted_j) = T.real(it);
            }
            else Tmat(i,j) = T.real(it);
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
    //eigval.print("eigval");
    //eigval.print("eigval=");
    //eigval.print();
    std::vector<double> log_lambdas = {};
    std::vector<double> scaling_dims = {};
    std::vector<double> spins = {};
    //obtain spins
    for (auto lambda_i_tilde : eigval){
        //eigen_vals.push_back(lambda_i_tilde);
        Cplx log_lambda_i_tilde = std::log(lambda_i_tilde);
        //std::cout<< log_lambda_i_tilde << std::endl;
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
    printf("scal_dim \t spin\n");
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

//translate to the right by one site
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

