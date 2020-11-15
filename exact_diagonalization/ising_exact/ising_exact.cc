#include "inc.h"
#include "glue.h"
#include "extract_mat.h"
#include "extract_cft_data.h"



double beta_c = 0.5*log(1+sqrt(2));//critical beta
int dim0 = 2;//initial A tensor leg dimension
int len_chain = 6; 
int num_states = 10;// final number of dots in the momentum diagram


int main(int argc, char* argv[]){
    if (argc==2) {
        len_chain = atoi(argv[1]);
    }
    
    // initial tensor legs
    Index s(dim0);
    Index u = addTags(s,"up");
    Index r = addTags(s,"right");
    Index d = addTags(s,"down");
    Index l = addTags(s,"left");

    // the initial tensor for Ising
    ITensor A = ITensor(u,r,d,l);

    // Fill the A tensor with correct Boltzmann weights:
    // 1 -> 1
    // 2-> -1
	auto Sig = [](int s) { return 1. - 2. * (s - 1); };
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
	A /= (TrA/2);
    Print(TrA);
    //PrintData(A);


    ITensor TMmat = glue(A);
    //Index Mi = M.index(1);
    //Index Mj = M.index(2);
    //auto TM_dense = extract_mat(TMmat);
    //obtain the first k eigenvalues from a sparse matrix
    //arma::sp_mat TM_sparse(TM_dense);
    printf("\nextracting matrix to armadillo matrix\n\n");
    arma::sp_mat TM_sparse = extract_spmat(TMmat);
    //arma::sp_mat TM_sparse(TM_dense);
    printf("\nextracting CFT data\n\n");
    extract_cft_data(TM_sparse);

    return 0;
}



