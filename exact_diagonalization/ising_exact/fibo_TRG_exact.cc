//#define ARMA_USE_HDF5

#include "inc.h"
#include "glue.h"
#include "extract_mat.h"
#include "extract_cft_data.h"


double phi = (std::sqrt(5)+1)/2;
//const double beta_c = 0.5*log(1+sqrt(2));//critical beta
// double beta_ = beta_c;
int svd_dim = 6;
int dim0 = 3;//initial A tensor leg dimension
int len_chain = 1; 
int num_states = 10;// final number of dots in the momentum diagram
//
int maxdim = 20;
int topscale = 8;


int main(int argc, char* argv[]){
    if (argc==4) {
        len_chain = atoi(argv[1]);
        maxdim = atoi(argv[2]);
        topscale = atoi(argv[3]);
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
    double trBB = elt(BB * delta(ccl,ccr) * delta(uc,lc));
    BB /= trBB;
    Print(BB);



    ITensor A=BB;
    Index u = A.index(1);
    Index r = A.index(2);
    Index d = A.index(3);
    Index l = A.index(4);


    //keep track of the normalization factor
    //Real TrA = elt(A * delta(l, r) * delta(u, d));
    // Real pfps = 0;//partition function per site
    for (auto scale : range1(topscale))
	{
		printfln("\n---------- Scale %d -> %d  ----------", scale - 1, scale);

		// Get the upper-left and lower-right tensors
		auto [Fl, Fr] = factor(A, { r,d }, { l,u }, { "MaxDim=",maxdim,
											 "Tags=","left,scale=" + str(scale),
											 "ShowEigs=",true });

		// Grab the new left Index
		auto l_new = commonIndex(Fl, Fr);

		// Get the upper-right and lower-left tensors
		auto [Fu, Fd] = factor(A, { l,d }, { u,r }, { "MaxDim=",maxdim,
											 "Tags=","up,scale=" + str(scale),
											 "ShowEigs=",true });

		// Grab the new up Index
		auto u_new = commonIndex(Fu, Fd);

		// Make the new index of Fl distinct
		// from the new index of Fr by changing
		// the tag from "left" to "right"
		auto r_new = replaceTags(l_new, "left", "right");
		Fr *= delta(l_new, r_new);

		// Make the new index of Fd distinct
		// from the new index of Fu by changing the tag
		// from "up" to "down"
		auto d_new = replaceTags(u_new, "up", "down");
		Fd *= delta(u_new, d_new);

		// relabel the indices to contract the 4 F tensors
		// to form the new A tensor
		Fl *= delta(r, l);
		Fu *= delta(d, u);
		Fr *= delta(l, r);
		Fd *= delta(u, d);
		A = Fl * Fu * Fr * Fd;

		//Print(A);

		// Update the indices
		l = l_new;
		r = r_new;
		u = u_new;
		d = d_new;

		// Normalize the current tensor and keep track of
		// the total normalization
		Real TrA = elt(A * delta(l, r) * delta(u, d));
		A /= TrA;
		//PrintData(A);
		// pfps += 1.0/pow(2,1+scale) * log(TrA);
		Print(A);
	}

	// printfln("log(Z)/N_s = %.12f\n", pfps);


    ITensor Amat = (len_chain>1) ? glue(A) : A*delta(l,r);
    //Index Mi = M.index(1);
    //Index Mj = M.index(2);
    //auto TM_dense = extract_mat(TMmat);
    //obtain the first k eigenvalues from a sparse matrix
    //arma::sp_mat TM_sparse(TM_dense);
    //ITensor Amat = A*delta(l,r);
    printf("\nextracting matrix to armadillo matrix\n\n");
    arma::sp_mat A_sparse = extract_spmat(Amat);
    //arma::sp_mat TM_sparse(TM_dense);
    printf("\nextracting CFT data\n\n");
    extract_cft_data(A_sparse);

    return 0;
}



