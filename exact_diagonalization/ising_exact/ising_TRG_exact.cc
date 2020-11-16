#include "inc.h"
#include "glue.h"
#include "extract_mat.h"
#include "extract_cft_data.h"



double beta_c = 0.5*log(1+sqrt(2));//critical beta
double beta_ = 0.5;
int dim0 = 2;//initial A tensor leg dimension
int len_chain = 3; 
int num_states = 10;// final number of dots in the momentum diagram
//
int maxdim = 20;
int topscale = 10;


int main(int argc, char* argv[]){
    if (argc==3) {
        len_chain = atoi(argv[1]);
        topscale = atoi(argv[2]);
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
					auto P = exp(E * beta_);
					A.set(l(sl), r(sr), u(su), d(sd), P);
                    //A.set(l(sl), r(sr), u(su), d(sd), 0);
				}
    // normalize A to prevent it from being too large
    // double TrA = elt(A * delta(l, r) * delta(u, d));
	// A /= (TrA/2);
    // Print(TrA);
    PrintData(A);
    ITensor Acopy = A;
    Acopy *= delta(l,r); PrintData(Acopy);

    //keep track of the normalization factor
    //Real TrA = elt(A * delta(l, r) * delta(u, d));
    Real pfps = 0;//partition function per site
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
		pfps += 1.0/pow(2,1+scale) * log(TrA);
		Print(A);
	}

	printfln("log(Z)/N_s = %.12f\n", pfps);


    // ITensor TMmat = glue(A);
    // //Index Mi = M.index(1);
    // //Index Mj = M.index(2);
    // //auto TM_dense = extract_mat(TMmat);
    // //obtain the first k eigenvalues from a sparse matrix
    // //arma::sp_mat TM_sparse(TM_dense);
    // printf("\nextracting matrix to armadillo matrix\n\n");
    // arma::sp_mat TM_sparse = extract_spmat(TMmat);
    // //arma::sp_mat TM_sparse(TM_dense);
    // printf("\nextracting CFT data\n\n");
    // extract_cft_data(TM_sparse);
    // ITensor Mmat = A*delta(l,r);
    // arma::sp_mat M_sparse = extract_spmat(Mmat);
    // extract_cft_data2(M_sparse);

    return 0;
}



