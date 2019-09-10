#include <Rcpp.h>
#include <RcppEigen.h>
#include <iostream>
#include <vector>
#include "quad.h"
using namespace Eigen;
using namespace std;
// [[Rcpp::depends(RcppEigen)]]


// [[Rcpp::export]]
Rcpp::List cdlcoxRcpp (const Map<MatrixXd> &y, const Map<MatrixXd> &x, const MatrixXd z = MatrixXd(0,0),
                  const bool standardize = true, const double alpha = 1, 
                  const double alpha1 = 0, const double alpha2 = 1, const double thresh = 1e-7)
{
    int n = y.rows();
    int p = x.cols();
    VectorXd delta = y.col(1);
    
    // get unique event time, D, and ties at event time, d
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
    VectorXi ck(n+1);
    VectorXi ri(m+1);
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
    
    RowVectorXd xm;
    RowVectorXd xs;
    if (z.size() == 0) {
        if (standardize) {
            xm = x.colwise().mean();
            xs = 1 / (((x.rowwise() - xm).cwiseProduct(x.rowwise() - xm)).colwise().sum() / n).array().sqrt();   /// xs = 1/sd
        } else {
            xm = RowVectorXd::Zero(p);
            xs = RowVectorXd::Ones(p);
        }
        
        //// compute lambda path
        int nlam = 100;
        MatrixXd betas(p, nlam);
        VectorXd beta = VectorXd::Zero(p);
        VectorXd W(n);
        VectorXd r(n);
        update_quadratic(x, beta, xm, xs, delta, ck, ri, d, n, m, W, r);
        
        double lambdaMax;
        VectorXd lambdas(nlam);
        if (alpha > 0 && alpha <= 1) {
            lambdaMax = ((((x.array().colwise() * r.array()).colwise().sum().array() - (r.sum() * xm).array()) * xs.array()).abs()).maxCoeff() / alpha;
        } else {
            lambdaMax = 1000 * ((((x.array().colwise() * r.array()).colwise().sum().array() - (r.sum() * xm).array()) * xs.array()).abs()).maxCoeff();
        }
        if (n >= p) {
            double alf = 1/exp(log(1/0.0001)/(nlam-1));
            lambdas[0] = lambdaMax;
            for (int i = 1; i < nlam; i++) {
                lambdas[i] = lambdas[i-1] * alf;
            }
        } else {
            double alf = 1/exp(log(1/0.01)/(nlam-1));
            lambdas[0] = lambdaMax;
            for (int i = 1; i < nlam; i++) {
                lambdas[i] = lambdas[i-1] * alf;
            }
        }
        
        //// penalty path loop
        for (int l = 0; l < nlam; l++) {
            double lambda_current = lambdas[l];
            double dev_out = 1e10;
            int iter = 0;
            VectorXd beta_old(p);
            VectorXd xv(p);
            //// outer re-weighted least squares loop
            while (dev_out >= thresh) {
                beta_old = beta;
                xv = (x.cwiseProduct(x).transpose() * W - (x.transpose() * W).cwiseProduct(2*xm.transpose()) +
                    W.sum() * xm.transpose().cwiseProduct(xm.transpose())).cwiseProduct(xs.transpose().cwiseProduct(xs.transpose()) / n);
                //// inner coordinate descent loop
                double dev_in = 1e10;
                while (dev_in >= thresh) {
                    dev_in = 0;
                    for (int j = 0; j < p; j++) {
                        double gj = (x.col(j).dot(r) - xm[j] * r.sum()) * xs[j];
                        double bj = beta[j];
                        double wls = gj + bj * xv[j];
                        double arg = abs(wls) - alpha * lambda_current;
                        if (arg > 0) {
                            beta[j] = copysign(arg, wls) / (xv[j] + (1 - alpha) * lambda_current);
                        } else {
                            beta[j] = 0;
                        }
                        double del = beta[j] - bj;
                        dev_in = max(dev_in, abs(del));
                        r -= del * (x.col(j).cwiseProduct(W) - xm[j] * W) * xs[j] / n;
                    }
                    iter += 1;
                }   // end of inner coordinate descent loop
                dev_out = (beta - beta_old).array().abs().maxCoeff();
                update_quadratic(x, beta, xm, xs, delta, ck, ri, d, n, m, W, r);
            }   // end of outer re-weighted loop
            if (standardize) {
                betas.col(l) = beta.cwiseProduct(xs.transpose());
            } else {
                betas.col(l) = beta;
            }
        }   // end of lambada path loop
        
        return Rcpp::List::create(Rcpp::Named("lambdas") = lambdas,
                                  Rcpp::Named("beta") = betas);
        
    } else {
        int q = z.cols();   //// external information model
        MatrixXd X(n, p+q);
        RowVectorXd xzs;
        RowVectorXd xzs_norm;
        RowVectorXd zs;
        if (standardize) {
            xzs = 1 / ((((x*z).rowwise() - (x*z).colwise().mean()).cwiseProduct((x*z).rowwise() - (x*z).colwise().mean())).colwise().sum() / n).array().sqrt();
            xm.resize(p+q);
            xs.resize(p+q);
            xm.head(p) = x.colwise().mean();
            xm.tail(q) = RowVectorXd::Zero(q);
            xs.head(p) = 1 / (((x.rowwise() - xm.head(p)).cwiseProduct(x.rowwise() - xm.head(p))).colwise().sum() / n).array().sqrt();   //// xs = 1/sd
            zs = 1 / (((z.rowwise() - z.colwise().mean()).cwiseProduct(z.rowwise() - z.colwise().mean())).colwise().sum() / p).array().sqrt();   //// 1/sd of z
            X.leftCols(p) = x;
            X.rightCols(q) = ((x.rowwise() - xm.head(p)).array().rowwise() * xs.head(p).array()).matrix() * (z.array().rowwise() * zs.array()).matrix();
            xs.tail(q) = RowVectorXd::Ones(q);
            xzs_norm = 1 / (((X.rightCols(q).rowwise() - X.rightCols(q).colwise().mean()).cwiseProduct(X.rightCols(q).rowwise() - X.rightCols(q).colwise().mean())).colwise().sum() / n).array().sqrt();
        } else {
            X.leftCols(p) = x;
            X.rightCols(q) = x*z;
            xm = RowVectorXd::Zero(p+q);
            xs = RowVectorXd::Ones(p+q);
        }
        //// compute lambda path, 2D grid
        int nlam = 20;
        //        vector< vector < vector <double> > > betas(nlam, vector<vector<double>> (nlam, vector<double> ((p+q), 0)));
        MatrixXd betas(p+q, nlam*nlam);
        VectorXd beta_l11 = VectorXd::Zero(p+q);
        VectorXd W_l11(n);
        VectorXd r_l11(n);
        update_quadratic(X, beta_l11, xm, xs, delta, ck, ri, d, n, m, W_l11, r_l11);
        
        double lambda1_max;
        double lambda2_max;
        VectorXd lambda1(nlam);
        VectorXd lambda2(nlam);
        if (alpha2 > 0 && alpha2 <= 1) {
            lambda2_max = ((((X.rightCols(q).array().colwise() * r_l11.array()).colwise().sum().array() - (r_l11.sum() * xm.tail(q)).array()) * xs.tail(q).array()).abs()).maxCoeff() / alpha2;
        } else {
            lambda2_max = 1000 * ((((X.rightCols(q).array().colwise() * r_l11.array()).colwise().sum().array() - (r_l11.sum() * xm.tail(q)).array()) * xs.tail(q).array()).abs()).maxCoeff();
        }
        if (alpha1 > 0 && alpha1 <= 1) {
            lambda1_max = ((((X.leftCols(p).array().colwise() * r_l11.array()).colwise().sum().array() - (r_l11.sum() * xm.head(p)).array()) * xs.head(p).array()).abs()).maxCoeff() / alpha1;
        } else {
            lambda1_max = 1000 * ((((X.leftCols(p).array().colwise() * r_l11.array()).colwise().sum().array() - (r_l11.sum() * xm.head(p)).array()) * xs.head(p).array()).abs()).maxCoeff();
        }
        lambda2[0] = lambda2_max;
        lambda1[0] = lambda1_max;
        if (n >= p+q) {
            double alf = 1/exp(log(1/0.0001)/(nlam-1));
            for (int i = 1; i < nlam; i++) {
                lambda2[i] = lambda2[i-1] * alf;
                lambda1[i] = lambda1[i-1] * alf;
            }
        } else {
            double alf = 1/exp(log(1/0.01)/(nlam-1));
            for (int i = 1; i < nlam; i++) {
                lambda2[i] = lambda2[i-1] * alf;
                lambda1[i] = lambda1[i-1] * alf;
            }
        }
        
        //// penalty path loop
        int ncols = 0;
        for (int l2 = 0; l2 < nlam; l2++) {
            double lambda2_current = lambda2[l2];
            VectorXd beta = beta_l11;
            VectorXd W = W_l11;
            VectorXd r = r_l11;
            
            for (int l1 = 0; l1 < nlam; l1++) {
                double lambda1_current = lambda1[l1];
                double dev_out = 1e10;
                VectorXd beta_old(p+q);
                VectorXd xv(p);
                //// outer re-weighted least squares loop
                while (dev_out >= thresh) {
                    beta_old = beta;
                    xv = (X.cwiseProduct(X).transpose() * W - (X.transpose() * W).cwiseProduct(2*xm.transpose()) +
                        W.sum() * xm.transpose().cwiseProduct(xm.transpose())).cwiseProduct(xs.transpose().cwiseProduct(xs.transpose()) / n);
                    //// inner coordinate descent loop
                    double dev_in = 1e10;
                    while (dev_in >= thresh) {
                        dev_in = 0;
                        for (int j = 0; j < p; j++) {
                            double gj = (X.col(j).dot(r) - xm[j] * r.sum()) * xs[j];
                            double bj = beta[j];
                            double wls = gj + bj * xv[j];
                            double arg = abs(wls) - alpha1 * lambda1_current;
                            if (arg > 0) {
                                beta[j] = copysign(arg, wls) / (xv[j] + (1 - alpha1) * lambda1_current);
                            } else {
                                beta[j] = 0;
                            }
                            double del = beta[j] - bj;
                            dev_in = max(dev_in, abs(del));
                            r -= del * (X.col(j).cwiseProduct(W) - xm[j] * W) * xs[j] / n;
                        }
                        for (int j = p; j < (p+q); j++) {
                            double gj = (X.col(j).dot(r) - xm[j] * r.sum()) * xs[j];
                            double bj = beta[j];
                            double wls = gj + bj * xv[j];
                            double arg = abs(wls) - alpha2 * lambda2_current;
                            if (arg > 0) {
                                beta[j] = copysign(arg, wls) / (xv[j] + (1 - alpha2) * lambda2_current);
                            } else {
                                beta[j] = 0;
                            }
                            double del = beta[j] - bj;
                            dev_in = max(dev_in, abs(del));
                            r -= del * (X.col(j).cwiseProduct(W) - xm[j] * W) * xs[j] / n;
                        }
                    }   // end of inner coordinate descent loop
                    dev_out = (beta - beta_old).array().abs().maxCoeff();
                    update_quadratic(X, beta, xm, xs, delta, ck, ri, d, n, m, W, r);
                }   // end of outer re-weighted loop
                //// hole the estimated beta, W, r of lambda1 at l1=1, for warm start of next lambda2 and lambda1 at l1=1
                if (l1==0) {
                    beta_l11 = beta;
                    W_l11 = W;
                    r_l11 = r;
                }
                if (standardize) {
                    for (int j = 0; j < p; j++) {
                        betas(j, ncols) = ( beta[j] + beta.tail(q).cwiseProduct(z.row(j).transpose().cwiseProduct(zs.transpose())).sum() ) * xs[j];
                    }
                    for (int j = p; j<(p+q); j++) {
                        betas(j, ncols) = beta[j] * xzs[j-p] / xzs_norm[j-p];
                    }
                } else {
                    for (int j = 0; j < p; j++) {
                        betas(j, ncols) = beta[j] + beta.tail(q).cwiseProduct(z.row(j).transpose()).sum();
                    }
                    for (int j = p; j<(p+q); j++) {
                        betas(j, ncols) = beta[j];
                    }
                }
                ncols += 1;
            }   // end of lambda1 loop
        }   // end of lambda2 loop
        
        return Rcpp::List::create(Rcpp::Named("lambda1") = lambda1,
                                         Rcpp::Named("lambda2") = lambda2,
                                         Rcpp::Named("beta") = betas);
        
    }
}
