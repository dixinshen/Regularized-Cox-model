#include <Rcpp.h>
#include <RcppEigen.h>
#include <iostream>
#include <vector>
using namespace Eigen;
using namespace std;

typedef Eigen::VectorXd VecXd;
typedef Eigen::VectorXi VecXi;
typedef Eigen::Map<const Eigen::MatrixXd> MapMat;
typedef Eigen::MappedSparseMatrix<double> MapSpMat;
typedef Eigen::Map<const Eigen::VectorXd> MapVec;
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
VecXd coxphRcpp (const Eigen::Map<Eigen::MatrixXd> y, const Eigen::Map<Eigen::MatrixXd> x,
                 const double thresh = 1e-7, const int iter_max = 1000)
{
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



// int main()
// {
//     MatrixXd y(42, 2);
//     y.col(0) << 1,1,2,2,3,4,4,5,5,6,6,6,6,7,8,8,8,8,9,10,10,11,11,11,12,12,13,15,16,17,17,19,20,22,22,23,23,25,32,32,34,35;
//     y.col(1) << 1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0;
//     MatrixXd x(42, 1);
//     x << 0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,0,1,0,0,0,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,1;
//     
//     //    VecXd y(13);
//     //    y << 1,2,3,3,4,5,5,5,6,7,8,8,9;
//     //    VecXd delta(13);
//     //    delta << 1,0,0,1,1,0,1,1,0,1,1,1,0;
//     //    int n = y.size();
//     
//     
//     VecXd beta = coxphRcpp(y, x);
//     
//     cout << "beta: " << "\n" << beta << endl;
//     
//     //    cout << endl << "ck: " << endl << ck.transpose() << endl;
//     //    cout << ck.size() << endl;
//     //    cout << endl << "ri: " << endl << ri.transpose() << endl;
//     //
//     //
//     //    cout << "Here is the vector y: \n" << y.transpose() << endl;
//     //    cout << "Here is the vector delta: \n" << delta.transpose() << endl;
//     //    cout << endl << "D: " << endl;
//     //    for (const auto & t : D)
//     //    {
//     //        cout << t << " ";
//     //    }
//     //    cout << endl;
//     //    cout << "d: " << endl;
//     //    for (const auto & t : d)
//     //    {
//     //        cout << t << " ";
//     //    }
//     //    cout << endl;
//     //    cout << endl;
//     //    cout << m << endl;
//     //    cout << d.size() << endl;
//     
//     
//     
//     return 0;
// }
// 
