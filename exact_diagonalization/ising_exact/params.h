#pragma once


double beta_c = 0.5*log(1+sqrt(2));//critical beta
int dim0 = 2;//initial A tensor leg dimension
int len_chain = 6; // the actual length is len_chain + 1
int len_chain_L = 3;
int len_chain_R = 3;
//int period = len_chain + 1;
int num_states = 10;// final number of dots in the momentum diagram

//threads
int max_num_threads = 30;
unsigned int n = std::thread::hardware_concurrency();
int num_threads = (n>max_num_threads ? max_num_threads : n);