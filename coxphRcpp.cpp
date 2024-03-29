#include <Rcpp.h>
#include <RcppEigen.h>
#include <iostream>
#include <vector>
#include "sort.h"
using namespace Eigen;
using namespace std;

typedef Eigen::VectorXd VecXd;
typedef Eigen::VectorXi VecXi;
typedef Eigen::Map<const Eigen::MatrixXd> MapMat;
typedef Eigen::MappedSparseMatrix<double> MapSpMat;
typedef Eigen::Map<const Eigen::VectorXd> MapVec;
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
VecXd coxphRcpp(Eigen::MatrixXd &y, Eigen::MatrixXd &x,
                 const double thresh = 1e-7, const int iter_max = 1000)
{
    // sort y, x by time in ascending order
    vector<size_t> index = sort_index(y.col(0));
    order(y, index);
    order(x, index);
    
    int n = y.rows();
    int p = x.cols();
    VecXd delta = y.col(1);
    VecXd beta = VecXd::Zero(p);
    VecXd beta_prime(p);
    int iter = 0;
    double deviation = 1e10;
    
    // get unique event times, and number of events at each event time
    int idx = 1;
    vector<double> D;
    vector<double> d;
    D.push_back(-1);
    for (int k = 0; k < n; k++) {
        if (delta[k]==1 && y(k,0)!=D[idx-1]) {
            D.push_back(y(k,0));
            d.push_back(1);
            idx += 1;
        } else if (delta[k]==1 && y(k,0)==D[idx-1]) {
            d[idx-2] += 1;
        }
    }
    D.erase(D.begin());
    int m = D.size();
    
    // get ck, and ri, risk sets
    int ck_prime = 0;
    VecXi ck(n+1);
    VecXi ri(m+1);
    ri[0] = n;
    for (int k = 1; k <= n; k++) {
        ck[k] = ck_prime;
        for (int j = ck_prime; j < m; j++) {
            if (D[j] <= y((k-1),0)) {
                ck[k] += 1;
                ri[ck[k]] = n - k + 1;
            } else {
                break;
            }
            ck_prime = ck[k];
        }
    }
    
    while (deviation >= thresh && iter <= iter_max)
    {
        // update quadratic
        beta_prime = beta;
        VecXd exp_eta = exp((x * beta_prime).array());
        double sum_exp_eta_prime = exp_eta.sum();
        VecXd sum_exp_eta(m);
        for (int i = 0; i < m; i++) {
            if (ri[i] == ri[i+1]) {
                sum_exp_eta[i] = sum_exp_eta_prime;
            } else {
                sum_exp_eta[i] = sum_exp_eta_prime - exp_eta.segment((n-ri[i]), (ri[i]-ri[i+1])).sum();
                sum_exp_eta_prime = sum_exp_eta[i];
            }
        }
        VecXd W(n);
        VecXd wr(n);
        double u_prime = 0;
        double u2_prime = 0;
        for (int k = 0; k < n; k++) {
            if (ck[k+1] == ck[k]) {
                W[k] = exp_eta[k] * u_prime - exp_eta[k] * exp_eta[k] * u2_prime;
                wr[k] = W[k] * log(exp_eta[k]) + delta[k] - exp_eta[k] * u_prime;
            } else {
                u_prime += d[ck[k+1] - 1] / sum_exp_eta[ck[k+1] - 1];
                u2_prime += d[ck[k+1 - 1]] / (sum_exp_eta[ck[k+1] - 1] * sum_exp_eta[ck[k+1] - 1]);
                W[k] = exp_eta[k] * u_prime - exp_eta[k] * exp_eta[k] * u2_prime;
                wr[k] = W[k] * log(exp_eta[k]) + delta[k] - exp_eta[k] * u_prime;
            }
        }
        
        beta = (x.transpose() * W.asDiagonal() * x).ldlt().solve(x.transpose() * wr);
        iter += 1;
        deviation = abs((beta - beta_prime).array()).maxCoeff();
    }
    
    return beta;
}
