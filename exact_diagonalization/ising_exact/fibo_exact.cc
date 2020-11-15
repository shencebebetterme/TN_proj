#include "inc.h"
#include "glue.h"
#include "extract_mat.h"
#include "extract_cft_data.h"

double phi = (std::sqrt(5)+1)/2;
int dim0 = 3;//initial A tensor leg dimension
int svd_dim = 6;
int len_chain = 6; // the actual length is len_chain + 1
int num_states = 10;// final number of dots in the momentum diagram

int main(int argc, char* argv[]){
    if (argc==2) {
        len_chain = atoi(argv[1]);
    }
    
    //construct the triangle 3-leg tensor
    Index s(dim0);
    Index uc = addTags(s,"uc");//upper center
    Index ll = addTags(s,"ll");//lower left
    Index lr = addTags(s,"lr");//lower right
    //
    Index lc = addTags(s,"lc");//lower center
    Index ul = addTags(s,"ul");//upper left
    Index ur = addTags(s,"ur");//upper right
    //
    ITensor aVtx(uc,lr,ll);
    for(int s_uc : range1(dim0))
        for(int s_lr : range1(dim0))
            for(int s_ll : range1(dim0))
                {
                    double val = 0;
                    //
                    if(s_uc==1 && s_lr==1 && s_ll==1) val= -std::pow(phi,-3.0/4);
                    //
                    if(s_uc==2 && s_lr==2 && s_ll==1) val= std::pow(phi,1.0/12);
                    if(s_uc==3 && s_lr==1 && s_ll==3) val= std::pow(phi,1.0/12);
                    if(s_uc==1 && s_lr==3 && s_ll==2) val= std::pow(phi,1.0/12);
                    //
                    aVtx.set(uc(s_uc),lr(s_lr),ll(s_ll), val);
                }
    //
    ITensor Vtx(ul,ur,lc);
    for(int s_ul : range1(dim0))
        for(int s_ur : range1(dim0))
            for(int s_lc : range1(dim0))
                {
                    double val = 0;
                    //
                    if(s_ul==1 && s_ur==1 && s_lc==1) val= -std::pow(phi,-3.0/4);
                    //
                    if(s_ul==1 && s_ur==2 && s_lc==2) val= std::pow(phi,1.0/12);
                    if(s_ul==3 && s_ur==1 && s_lc==3) val= std::pow(phi,1.0/12);
                    if(s_ul==2 && s_ur==3 && s_lc==1) val= std::pow(phi,1.0/12);
                    //
                    Vtx.set(ul(s_ul),ur(s_ur),lc(s_lc), val);
                }
    //
    Print(aVtx);Print(Vtx);
    //construct the Vertex-anti-Vertex and do svd
    ITensor VaV = Vtx * delta(lc,uc) * aVtx;
    //Print(VaV);
    // VaV.replaceTags({"ul"},{"ul_new"},{ul,ur,lr,ll});
    // VaV.replaceTags({"ur"},{"ur_new"},{ul,ur,lr,ll});
    // VaV.replaceTags({"lr"},{"lr_new"},{ul,ur,lr,ll});
    // VaV.replaceTags({"ll"},{"ll_new"},{ul,ur,lr,ll});
    Index ul_new(dim0); VaV *= delta(ul,ul_new);
    Index ur_new(dim0); VaV *= delta(ur,ur_new);
    Index lr_new(dim0); VaV *= delta(lr,lr_new);
    Index ll_new(dim0); VaV *= delta(ll,ll_new);
    Print(VaV);
    auto[lVtx,rVtx] = factor(VaV,{ul_new,ll_new},{ur_new,lr_new},{"MaxDim=",svd_dim,"Tags=","ccr","ShowEigs=",true});
    Index ccr = commonIndex(lVtx,rVtx);
    Index ccl = replaceTags(ccr,"ccr","ccl");
    rVtx *= delta(ccl,ccr);
    //
    ITensor BB = aVtx * delta(lr,ul_new) * lVtx * delta(ll_new,ur) * Vtx * delta(ul,lr_new) * rVtx * delta(ur_new,ll);
    Print(BB);
    double trBB = elt(BB * delta(ccl,ccr) * delta(uc,lc));
    BB /= trBB;


    ITensor TMmat = glue(BB);
    //Index Mi = M.index(1);
    //Index Mj = M.index(2);
    //auto TM_dense = extract_mat(TMmat);
    //obtain the first k eigenvalues from a sparse matrix
    //arma::sp_mat TM_sparse(TM_dense);
    printf("\nextracting matrix to armadillo matrix\n\n");
    arma::sp_mat TM_sparse = extract_spmat(TMmat);
    //TM_sparse *= TM_sparse.t();
    //arma::sp_mat TM_sparse(TM_dense);
    printf("\nextracting CFT data\n\n");
    extract_cft_data(TM_sparse);

    return 0;
}