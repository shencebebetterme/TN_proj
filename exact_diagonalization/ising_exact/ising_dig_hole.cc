//#define ARMA_USE_HDF5

#include "inc.h"
#include "glue.h"
#include "extract_mat.h"
#include "extract_cft_data.h"



const double beta_c = 0.5*log(1+sqrt(2));//critical beta
double beta_ = beta_c;
int dim0 = 2;//initial A tensor leg dimension
int len_chain = 1; 
int num_states = 10;// final number of dots in the momentum diagram
//
int maxdim = 4;
int pre_scale = 1;
int top_scale = pre_scale + 2;


int main(int argc, char* argv[]){
    if (argc==3) {
        pre_scale = atoi(argv[1]);
        maxdim = atoi(argv[2]);
        // topscale = atoi(argv[3]);
    }
    
    ITensor As; // A tensor at pre-scale
    ITensor As4; // 4 A tensor at pre-scale
    Index us, rs, ds, ls;
    //
    // Index u1,u2,r1,r2,d1,d2,l1,l2;
    // Index i1,i2,i3,i4;
    //
    int dims;
    std::vector<ITensor> vec_U; // U tensor from pre-scale+1
    std::vector<ITensor> vec_R; // R tensor from pre-scale+1
    std::vector<ITensor> vec_D; // D tensor from pre-scale+1
    std::vector<ITensor> vec_L; // L tensor from pre-scale+1
    //used to store A tensor at each scale
    // std::vector<ITensor> vec_A = {};
    // std::vector<Real> vec_trA = {};
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
    // Real pfps = 0;//partition function per site
    for (auto scale : range1(top_scale))
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

        if (scale>pre_scale){
            ITensor U = Fl * Fu;
            ITensor R = Fu * Fr;
            ITensor D = Fd * Fr;
            ITensor L = Fl * Fd;
            vec_U.push_back(U);
            vec_R.push_back(R);
            vec_D.push_back(D);
            vec_L.push_back(L);
            // As4 *= (U * delta(d,u1) * delta(u,u2));
        }
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

        if (scale==pre_scale){
            As = A;
            Print(As);
            //
            us = u;
            rs = r;
            ds = d;
            ls = l;
            //
            dims = l.dim();
            //
           
            // Print(A11);Print(A12);Print(A21);Print(A22);
            // Print(As4);
        }
	}

    
    Index u1(dims,"u1"), u2(dims,"u2"), r1(dims,"r1"), r2(dims,"r2"), d1(dims,"d1"), d2(dims,"d2"), l1(dims,"l1"), l2(dims,"l2");
    Index i1(dims), i2(dims), i3(dims), i4(dims);
    // Print(delta(u,u1));
    ITensor A11 = As * delta(us,u1) * delta(ls,l1) * delta(rs,i1) * delta(ds,i4);
    ITensor A12 = As * delta(ls,i1) * delta(us,u2) * delta(rs,r1) * delta(ds,i2);
    ITensor A21 = As * delta(us,i4) * delta(rs,i3) * delta(ds,d1) * delta(ls,l2);
    ITensor A22 = As * delta(ls,i3) * delta(us,i2) * delta(rs,r2) * delta(ds,d2);
    As4 = A11 * A12 * A22 * A21;
    

    printf("\n bond dimension is %d at scale %d\n\n", dims, pre_scale);
    Print(As4);

    // Print(vec_U[0]); Print(vec_R[0]); Print(vec_D[0]); Print(vec_L[0]);

    ITensor T = As4;
    //
    ITensor U1 = vec_U[0];
    ITensor R1 = vec_R[0];
    ITensor D1 = vec_D[0];
    ITensor L1 = vec_L[0];
    //
    Print(T);
    T *= (U1 * delta(u1,U1.index(1)) * delta(u2,U1.index(3)));
    T *= (D1 * delta(d1,D1.index(1)) * delta(d2,D1.index(3)));
    L1.prime(); R1.prime();
    T *= (R1 * delta(r1,R1.index(1)) * delta(r2,R1.index(3)));
    T *= (L1 * delta(l1,L1.index(1)) * delta(l2,L1.index(3)));
    Print(T);
    // relabel the external legs
    u1 = U1.index(4);
    u2 = R1.index(2);
    r1 = R1.index(4);
    r2 = D1.index(4);
    d2 = D1.index(2);
    d1 = L1.index(4);
    l2 = L1.index(2);
    l1 = U1.index(2); 

    ITensor U2 = vec_U[1];
    ITensor R2 = vec_R[1];
    ITensor D2 = vec_D[1];
    ITensor L2 = vec_L[1];
    //
    Print(T);
    T *= (U2 * delta(u1,U2.index(1)) * delta(u2,U2.index(3)));
    T *= (D2 * delta(d1,D2.index(1)) * delta(d2,D2.index(3)));
    L2.prime(); R2.prime();
    T *= (R2 * delta(r1,R2.index(1)) * delta(r2,R2.index(3)));
    T *= (L2 * delta(l1,L2.index(1)) * delta(l2,L2.index(3)));
    Print(T);
    // relabel the external legs
    l1 = L2.index(4);
    u1 = L2.index(2);
    u2 = U2.index(2);
    r1 = U2.index(4);
    r2 = R2.index(2);
    d2 = R2.index(4);
    d1 = D2.index(4);
    l2 = D2.index(2);
    Print(T);

    //match legs of As4 and T 
    As4 *= delta(l1,As4.index(1));
    As4 *= delta(u1,As4.index(2));
    As4 *= delta(u2,As4.index(3));
    As4 *= delta(r1,As4.index(4));
    As4 *= delta(r2,As4.index(5));
    As4 *= delta(d2,As4.index(6));
    As4 *= delta(d1,As4.index(7));
    As4 *= delta(l2,As4.index(8));

    //normalization
    T /= norm(T);
    As4 /= norm(As4);
    std::cout<< norm(T-As4) << " " << norm(As4) << std::endl;

    return 0;
}



