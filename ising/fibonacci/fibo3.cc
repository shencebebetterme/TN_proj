#include "inc.h"


double phi = 0.5*(1+sqrt(5));
int dim0 = 2;//initial A tensor leg dimension
int new_dim0 = 4;
int len_chain = 5; // the actual length is len_chain + 1
int len_chain_L = 3;
int len_chain_R = 3;
//int period = len_chain + 1;
int num_states = 20;// final number of dots in the momentum diagram
int svd_maxdim = 6;
// convention can be "maintext" or "appendix" or "ruoshui" or "mine"
//std::string convention = "maintext";
std::string convention = "ruoshui";


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


    //
    // build A tensor
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
                                double val = 0;
                                //double val = std::pow(phi,0.25) / sqrt(qd(beta)) * F(gamma,alpha,beta);
                                if ((alpha==2)&&(beta==2)&&(gamma==2)){
                                    if (convention=="ruoshui") {val = -std::pow(phi,-1.0/4);}
                                    if (convention=="maintext") {val = -std::pow(phi,-1.0/2);}
                                    if (convention=="mine") {val = -std::pow(phi,1/4);}
                                }
                                if (oneIs1(alpha,beta,gamma)){
                                    if (convention=="ruoshui") {val = std::pow(phi,7.0/12);}
                                    if (convention=="maintext") {val = std::pow(phi,1.0/2);}
                                    if (convention=="mine") {val = std::pow(phi,-1/4);}
                                }
                                A.set(alpha1=sa1,alpha2=sa2,beta1=sb1,beta2=sb2,gamma1=sg1,gamma2=sg2,val);
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
    auto [abC,ab] = combiner(alpha1,beta2);
    auto [agC,ag] = combiner(alpha2,gamma1);
    auto [bgC,bg] = combiner(beta1,gamma2);
    A *= abC;
    A *= agC;
    A *= bgC;
    // now A is a 3 leg tensor, each has dim 4
    Print(A);

    //build B tensor
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
                                double val = 0;
                                //double val = std::pow(phi,0.25) / sqrt(qd(nu)) * F(rho,mu,nu);
                                //B.set(mu1=sm1,mu2=sm2,rho1=sr1,rho2=sr2,nu1=sn1,nu2=sn2,val);
                                 if ((mu==2)&&(rho==2)&&(nu==2)){
                                    if (convention=="ruoshui") {val = -std::pow(phi,-1.0/4);}
                                    if (convention=="maintext") {val = -std::pow(phi,-1.0/2);}
                                    if (convention=="mine") {val = -std::pow(phi,1/4);}
                                }
                                else if (oneIs1(mu,rho,nu)){
                                    if (convention=="ruoshui") {val = std::pow(phi,7.0/12);}
                                    if (convention=="maintext") {val = std::pow(phi,1.0/2);}
                                    if (convention=="mine") {val = std::pow(phi,-1/4);}
                                }
                                B.set(mu1=sm1,mu2=sm2,rho1=sr1,rho2=sr2,nu1=sn1,nu2=sn2,val);
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
    auto [mnC,mn] = combiner(mu1,nu2);
    auto [mrC,mr] = combiner(mu2,rho1);
    auto [nrC,nr] = combiner(nu1,rho2);
    B *= mnC;
    B *= mrC;
    B *= nrC;
    // now B is a 3 leg tensor, each has dim 4
    Print(B);

    ITensor C = A * B * delta(ab,nr);
    Print(C);
    //Print(mn);
    //Print(ag);
    //Print(mr);
    //Print(bg);
    auto [Fr,Fl] = factor(C,{mn,ag},{mr,bg},{"MaxDim=",svd_maxdim,"Tags=","right"});
    Index r = commonIndex(Fr,Fl);
    Index l = replaceTags(r,"right","left");
    Fl *= delta(r,l);//now Fr has svd index r, and Fl has svd index l


    Index u(new_dim0,"up");
    Index a1(new_dim0);
    Index a2(new_dim0);
    Index d(new_dim0,"down");
    Index b1(new_dim0);
    Index b2(new_dim0);
    ITensor Aprime = A * delta(ab,u) * delta(ag,a1) * delta(bg,a2);
    ITensor Bprime = B * delta(mn,b1) * delta(mr,b2) * delta(nr,d);
    ITensor D = Fl * Fr * Aprime * Bprime * delta(a1,mr) * delta(a2,mn) * delta(b1,bg) * delta(b2,ag);
    Print(D);

    int chi1 = dim0;
    int chi2 = svd_maxdim;

    Index u0L = addTags(u,"L,u_i="+str(0));
    Index d0L = addTags(d,"L,d_i="+str(0));
    Index l0L = addTags(l,"L,site="+str(0));
    Index r0L = addTags(r,"L,site="+str(0));
    Index u0R = addTags(u,"R,u_i=" +str(len_chain_L));
    Index d0R = addTags(d,"R,d_i=" +str(len_chain_L));
    Index l0R = addTags(l,"R,site="+str(len_chain_L));
    Index r0R = addTags(r,"R,site="+str(len_chain_L));
    //initial M tensor
    ITensor ML = D * delta(u,u0L) * delta(d,d0L) * delta(l,l0L) * delta(r,r0L);
    ITensor MR = D * delta(u,u0R) * delta(d,d0R) * delta(l,l0R) * delta(r,r0R);
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
        ITensor DiL = D * delta(u,uiL) * delta(d,diL) * delta(r,riL) * delta(l,liL);
        Index previous_riL = addTags(r,"L,site="+str(i-1));
        //printf("eating %dth A tensor, left\n",i);
        ML *= (DiL * delta(previous_riL, liL));
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
        ITensor DiR = D * delta(u,uiR) * delta(d,diR) * delta(r,riR) * delta(l,liR);
        Index previous_riR = addTags(r,"R,site="+str(i-1+len_chain_L));
        //printf("eating %dth A tensor, right\n",i);
        MR *= (DiR * delta(previous_riR, liR));
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
    // remove the L and R tags since we've combined the left and right
    M.removeTags("L");
    M.removeTags("R");
    // compose with the translational operator
    for (auto i=0; i<len_chain-1; i++){
        M.swapTags("d_i="+str(i), "d_i="+str(i+1));
    }

    ITensor& TM = M;
    Print(TM);
    //group up and down indices
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
    // now TM becomes a matrix
    ITensor& TMmat = TM;
    Print(TMmat);
    arma::sp_mat TM_sparse = extract_sparse_mat_par(TMmat,false);
    extract_cft_data(TM_sparse);
}