#pragma once

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
#include <omp.h>
#include <thread>
//#include <random>
#include <execution>
#include <numeric>

#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <armadillo>


using namespace itensor;
using namespace std;
using namespace std::chrono;
using namespace arma;


extern int len_chain;
int len_chain_L = 3;
int len_chain_R = 3;

// gluing building blocks to a loop
ITensor glue(const ITensor& A){
    // decompose the chain into  two parts
    // stupid way to reduce memory usage and avoid
    // blas 32bit 64bit dgemm parameter 3 issue
    if (len_chain%2==0) {
        len_chain_L = len_chain_R = len_chain/2;
    }
    else
    {
        len_chain_L = (len_chain-1)/2;
        len_chain_R = len_chain - len_chain_L;
    }

    Index u = A.index(1);
    Index r = A.index(2);
    Index d = A.index(3);
    Index l = A.index(4);
    // horizontal and vertical dimension
    if(!((u.dim()==d.dim()) && (r.dim()==l.dim()))) printf("dimension of A doesn't match.\n");
    int dimH = r.dim();
    int dimV = u.dim();
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
    ITensor ML = A * delta(u,u0L) * delta(d,d0L) * delta(l,l0L) * delta(r,r0L);
    ITensor MR = A * delta(u,u0R) * delta(d,d0R) * delta(l,l0R) * delta(r,r0R);
     // indexset for decomposition
    std::vector<Index> upL_inds_vec = {u0L};
    std::vector<Index> downL_inds_vec = {d0L};
    std::vector<Index> upR_inds_vec = {u0R};
    std::vector<Index> downR_inds_vec = {d0R};
    //
    for (int i : range1(len_chain_L-1)){
        Index uiL = addTags(u,"L,u_i="+str(i));
        Index diL = addTags(d,"L,d_i="+str(i));
        Index liL = addTags(l,"L,site="+str(i));
        Index riL = addTags(r,"L,site="+str(i));
        //construct the i-th copy of A tensor
        ITensor AiL = A * delta(u,uiL) * delta(d,diL) * delta(r,riL) * delta(l,liL);
        Index previous_riL = addTags(r,"L,site="+str(i-1));
        //printf("eating %dth A tensor, left\n",i);
        ML *= (AiL * delta(previous_riL, liL));
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
        ITensor AiR = A * delta(u,uiR) * delta(d,diR) * delta(r,riR) * delta(l,liR);
        Index previous_riR = addTags(r,"R,site="+str(i-1+len_chain_L));
        //printf("eating %dth A tensor, right\n",i);
        MR *= (AiR * delta(previous_riR, liR));
        //TM.swapTags("d_i="+str(i),"d_i="+str(i-1));
        upR_inds_vec.push_back(uiR);
        downR_inds_vec.push_back(diR);
    }
    //
    Index L_rightmost = addTags(r,"L,site="+str(len_chain_L-1));
    Index R_rightmost = addTags(r,"R,site="+str(len_chain_L+len_chain_R-1));
    printf("\ncombining left and right\n\n");
    // form a ring-shaped MPO
    ITensor& M = ML;
    M *= MR * delta(L_rightmost,l0R) * delta(l0L,R_rightmost);
    Print(M);
    // remove the L and R tags since we've combined the left and right part
    M.removeTags("L");
    M.removeTags("R");
    
    // compose with the translational operator
    for (auto i=0; i<len_chain-1; i++){
        M.swapTags("d_i="+str(i), "d_i="+str(i+1));
    }
    ITensor& TM = M;

    //combine up indices and down indices respectively
    std::vector<Index>& up_inds = upL_inds_vec;
    up_inds.insert(up_inds.end(), upR_inds_vec.begin(), upR_inds_vec.end());
    std::vector<Index>& down_inds = downL_inds_vec;
    down_inds.insert(down_inds.end(), downR_inds_vec.begin(), downR_inds_vec.end());
    // combine the up and down indices into uL and dL
    auto[uLC,uL] = combiner(up_inds);
    auto[dLC,dL] = combiner(down_inds);
    uLC.removeTags("L");
    uLC.removeTags("R");
    dLC.removeTags("L");
    dLC.removeTags("R");
    TM *= uLC;
    TM *= dLC;
    Print(TM);
    // now TM is a matrix
    ITensor& TMat = TM;
    return(TMat);
}