//
// Created by Dixin Shen on 8/28/19.
//

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "quad.h"
#include "sort.h"

using namespace Eigen;
using namespace std;

int csvRead(MatrixXd& outputMatrix, const string& fileName, const streamsize dPrec);

MatrixXd cdl_cox (MatrixXd &y, MatrixXd &x, const Ref<const MatrixXd> &z = MatrixXd(0,0),
                  const bool standardize = true, const double alpha = 1, const double alpha1 = 0, const double alpha2 = 1,
                  const VectorXd &lambda = VectorXd(0), const VectorXd &lambda1 = VectorXd(0), const VectorXd &lambda2 = VectorXd(0),
                  const VectorXd &penalty_factor = VectorXd(0), const double thresh = 1e-7, const int iter_max_outer = 500, const int iter_max_inner = 500)
{
    typedef Matrix<bool, Dynamic, 1> VectorXb;
    // sort y, x by time in ascending order
    vector<size_t> index = sort_index(y.col(0));
    order(y, index);
    order(x, index);

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
    ri[m] = 0;
    for (int k = 1; k <= n; k++) {
        ck[k] = ck_prime;
        for (int j = ck_prime; j < m; j++) {
            if (D[j] <= y((k-1),0)) {
                ck[k] += 1;
                ri[ck[k]-1] = n - k + 1;
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
        VectorXd cmult;
        if (penalty_factor.size() == 0) {
            cmult = VectorXd::Ones(p);
        } else {
            cmult = penalty_factor * p / penalty_factor.sum();
        }
        int nlam;
        VectorXd lam;
        VectorXd beta = VectorXd::Zero(p);
        VectorXd W(n);
        VectorXd r(n);
        update_quadratic(x, beta, xm, xs, delta, ck, ri, d, n, m, W, r);
        if (lambda.size() == 0) {
            nlam = 100;
            lam.resize(nlam);
            double lambdaMax = 0;
            if (alpha > 0 && alpha <= 1) {
                for (int j = 0; j < p; j++) {
                    if (cmult[j] > 0) {
                        lambdaMax = max(lambdaMax,
                                        abs((x.col(j).dot(r) - r.sum() * xm[j]) * xs[j] / (cmult[j] * alpha)));
                    }
                }
            } else {
                for (int j =0; j < p; j++) {
                    if (cmult[j] > 0) {
                        lambdaMax = max(lambdaMax,
                                        abs((x.col(j).dot(r) - r.sum() * xm[j]) * xs[j] / cmult[j]));
                    }
                }
                lambdaMax *= 1000;
            }
            if (n >= p) {
                double alf = 1 / exp(log(1 / 0.0001) / (nlam - 1));
                lam[0] = lambdaMax;
                for (int i = 1; i < nlam; i++) {
                    lam[i] = lam[i - 1] * alf;
                }
            } else {
                double alf = 1 / exp(log(1 / 0.01) / (nlam - 1));
                lam[0] = lambdaMax;
                for (int i = 1; i < nlam; i++) {
                    lam[i] = lam[i - 1] * alf;
                }
            }
        } else {
            nlam = lambda.size();
            lam.resize(nlam);
            lam = lambda;
        }

        MatrixXd betas(p, nlam);
        VectorXb strong_set = VectorXb::Constant(p, false);
        VectorXb active_set = VectorXb::Constant(p, true);
        double lambda_old = 0.0;

        //// penalty path loop
        for (int l = 0; l < nlam; l++) {
            double lambda_current = lam[l];
            int iter = 0;
            VectorXd gradient = ((x.array().colwise() * r.array()).colwise().sum().array() - (r.sum() * xm).array()) * xs.array();
            for (int j = 0; j < p; j++) {
                strong_set[j] = std::abs(gradient[j]) > alpha * (2*lambda_current - lambda_old) * cmult[j];
            }
            VectorXd beta_old(p);
            VectorXd xv(p);
            //// outer re-weighted least squares loop
            int iter_outer = 0;
            bool converge_outer = false;
            while (!converge_outer && iter_outer <= iter_max_outer) {
                iter_outer += 1;
                beta_old = beta;
                xv = (x.cwiseProduct(x).transpose() * W - (x.transpose() * W).cwiseProduct(2*xm.transpose()) +
                      W.sum() * xm.transpose().cwiseProduct(xm.transpose())).cwiseProduct(xs.transpose().cwiseProduct(xs.transpose()) / n);
                //// inner coordinate descent loop
                int iter_inner = 0;
                bool converge_inner = false;
                while (!converge_inner && iter_inner <= iter_max_inner) {
                    iter_inner +=1;
                    double dlx = 0.0;
                    for (int j = 0; j < p; j++) {
                        if (strong_set[j] && active_set[j]) {
                            double gj = (x.col(j).dot(r) - xm[j] * r.sum()) * xs[j];
                            double bj = beta[j];
                            double wls = gj + bj * xv[j];
                            double arg = abs(wls) - alpha * cmult[j] * lambda_current;
                            if (arg > 0) {
                                beta[j] = copysign(arg, wls) / (xv[j] + (1 - alpha) * cmult[j] * lambda_current);
                            } else {
                                beta[j] = 0;
                            }
                            double del = beta[j] - bj;
                            if (abs(del) > 0.0) {
                                dlx = max(dlx, xv[j]*del*del);
                                r -= del * (x.col(j).cwiseProduct(W) - xm[j] * W) * xs[j] / n;
                            } else {
                                active_set[j] = false;
                            }
                        }
                    }
                    iter += 1;
                    if (dlx < thresh) {
                        converge_inner = true;
                        active_set.setConstant(true);
                    }
                }   // end of inner coordinate descent loop
                update_quadratic(x, beta, xm, xs, delta, ck, ri, d, n, m, W, r);
                if ( (xv.cwiseProduct((beta-beta_old).cwiseProduct(beta-beta_old))).maxCoeff() < thresh ) {
                    //// check kkt violation
                    gradient = ((x.array().colwise() * r.array()).colwise().sum().array() - (r.sum() * xm).array()) * xs.array();
                    int num_violations = 0;
                    for (int j = 0; j < p; j++) {
                        if (!strong_set[j]) {
                            if (gradient[j] > alpha * cmult[j] * lambda_current) {
                                strong_set[j] = true;
                                num_violations += 1;
                            }
                        }
                    }
                    if (num_violations == 0) {converge_outer = true;}
                }
            }   // end of outer re-weighted loop
            lambda_old = lambda_current;

            if (standardize) {
                betas.col(l) = beta.cwiseProduct(xs.transpose());
            } else {
                betas.col(l) = beta;
            }
        }   // end of lambada path loop

        return betas;

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
        VectorXd cmult;
        if (penalty_factor.size() == 0) {
            cmult = VectorXd::Ones(p+q);
        } else {
            cmult = penalty_factor * (p+q) / penalty_factor.sum();
        }
        int nlam1;
        int nlam2;
        VectorXd lam1;
        VectorXd lam2;
        //        vector< vector < vector <double> > > betas(nlam, vector<vector<double>> (nlam, vector<double> ((p+q), 0)));
        VectorXd beta_l11 = VectorXd::Zero(p + q);
        VectorXd W_l11(n);
        VectorXd r_l11(n);
        update_quadratic(X, beta_l11, xm, xs, delta, ck, ri, d, n, m, W_l11, r_l11);
        if (lambda1.size() == 0 && lambda2.size() == 0) {
            nlam1 = 20;
            nlam2 = 20;
            double lambda1_max = 0;
            double lambda2_max = 0;
            lam1.resize(nlam1);
            lam2.resize(nlam1);
            if (alpha2 > 0 && alpha2 <= 1) {
                for (int j = p; j < (p+q); j++) {
                    if (cmult[j] > 0) {
                        lambda2_max = max(lambda2_max,
                                          abs((X.col(j).dot(r_l11) - r_l11.sum() * xm[j]) * xs[j] /
                                              (cmult[j] * alpha2)));
                    }
                }
            } else {
                for (int j = p; j < (p+q); j++) {
                    if (cmult[j] > 0) {
                        lambda2_max = max(lambda2_max,
                                          abs((X.col(j).dot(r_l11) - r_l11.sum() * xm[j]) * xs[j] / cmult[j]));
                    }
                }
                lambda2_max *= 1000;
            }
            if (alpha1 > 0 && alpha1 <= 1) {
                for (int j = 0; j < p; j++) {
                    if (cmult[j] > 0) {
                        lambda1_max = max(lambda1_max,
                                          abs((X.col(j).dot(r_l11) - r_l11.sum() * xm[j]) * xs[j] /
                                              (cmult[j] * alpha1)));
                    }
                }
            } else {
                for (int j = 0; j < p; j++) {
                    if (cmult[j] > 0) {
                        lambda1_max = max(lambda1_max,
                                          abs((X.col(j).dot(r_l11) - r_l11.sum() * xm[j]) * xs[j] / cmult[j]));
                    }
                }
                lambda1_max *= 1000;
            }
            lam2[0] = lambda2_max;
            lam1[0] = lambda1_max;
            if (n >= p + q) {
                double alf2 = 1 / exp(log(1 / 0.0001) / (nlam2 - 1));
                double alf1 = 1 / exp(log(1 / 0.0001) / (nlam1 - 1));
                for (int i = 1; i < nlam2; i++) {
                    lam2[i] = lam2[i - 1] * alf2;
                }
                for (int i = 1; i < nlam1; i++) {
                    lam1[i] = lam1[i - 1] * alf1;
                }
            } else {
                double alf2 = 1 / exp(log(1 / 0.01) / (nlam2 - 1));
                double alf1 = 1 / exp(log(1 / 0.01) / (nlam1 - 1));
                for (int i = 1; i < nlam2; i++) {
                    lam2[i] = lam2[i - 1] * alf2;
                }
                for (int i = 1; i < nlam1; i++) {
                    lam1[i] = lam1[i - 1] * alf1;
                }
            }
        } else {
            nlam1 = lambda1.size();
            nlam2 = lambda2.size();
            lam1.resize(nlam1);
            lam2.resize(nlam2);
            lam1 = lambda1;
            lam2 = lambda2;
        }

        MatrixXd betas(p + q, nlam1 * nlam2);
        VectorXb strong_set = VectorXb::Constant(p+q, false);
        VectorXb active_set = VectorXb::Constant(p+q, true);
        double lambda1_old = 0.0;
        double lambda2_old = 0.0;

        //// penalty path loop
        int ncols = 0;
        for (int l2 = 0; l2 < nlam2; l2++) {
            double lambda2_current = lam2[l2];
            VectorXd beta = beta_l11;
            VectorXd W = W_l11;
            VectorXd r = r_l11;

            for (int l1 = 0; l1 < nlam1; l1++) {
                double lambda1_current = lam1[l1];
                VectorXd gradient = ((X.array().colwise() * r.array()).colwise().sum().array() - (r.sum() * xm).array()) * xs.array();
                for (int j = 0; j < p+q; j++) {
                    if (j < p) {
                        strong_set[j] = std::abs(gradient[j]) > alpha1 * (2 * lambda1_current - lambda1_old) * cmult[j];
                    } else {
                        strong_set[j] = std::abs(gradient[j]) > alpha2 * (2 * lambda2_current - lambda2_old) * cmult[j];
                    }
                }
                VectorXd beta_old(p+q);
                VectorXd xv(p);

                //// outer reweighted least squares loop
                int iter_outer = 0;
                bool converge_outer = false;
                while (!converge_outer && iter_outer <= iter_max_outer) {
                    iter_outer += 1;
                    beta_old = beta;
                    xv = (X.cwiseProduct(X).transpose() * W - (X.transpose() * W).cwiseProduct(2*xm.transpose()) +
                          W.sum() * xm.transpose().cwiseProduct(xm.transpose())).cwiseProduct(xs.transpose().cwiseProduct(xs.transpose()) / n);

                    //// inner coordinate descent loop
                    int iter_inner = 0;
                    bool converge_inner = false;
                    while (!converge_inner && iter_inner <= iter_max_inner) {
                        iter_inner += 1;
                        double dlx = 0;
                        for (int j = 0; j < p; j++) {
                            if (strong_set[j] && active_set[j]) {
                                double gj = (X.col(j).dot(r) - xm[j] * r.sum()) * xs[j];
                                double bj = beta[j];
                                double wls = gj + bj * xv[j];
                                double arg = abs(wls) - alpha1 * cmult[j] * lambda1_current;
                                if (arg > 0) {
                                    beta[j] = copysign(arg, wls) / (xv[j] + (1 - alpha1) * cmult[j] * lambda1_current);
                                } else {
                                    beta[j] = 0;
                                }
                                double del = beta[j] - bj;
                                if (abs(del) > 0.0) {
                                    dlx = max(dlx, xv[j]*del*del);
                                    r -= del * (X.col(j).cwiseProduct(W) - xm[j] * W) * xs[j] / n;
                                } else {
                                    active_set[j] = false;
                                }
                            }
                        }
                        for (int j = p; j < (p+q); j++) {
                            if (strong_set[j] && active_set[j]) {
                                double gj = (X.col(j).dot(r) - xm[j] * r.sum()) * xs[j];
                                double bj = beta[j];
                                double wls = gj + bj * xv[j];
                                double arg = abs(wls) - alpha2 * cmult[j] * lambda2_current;
                                if (arg > 0) {
                                    beta[j] = copysign(arg, wls) / (xv[j] + (1 - alpha2) * cmult[j] * lambda2_current);
                                } else {
                                    beta[j] = 0;
                                }
                                double del = beta[j] - bj;
                                if (abs(del) > 0.0) {
                                    dlx = max(dlx, xv[j]*del*del);
                                    r -= del * (X.col(j).cwiseProduct(W) - xm[j] * W) * xs[j] / n;
                                } else {
                                    active_set[j] = false;
                                }
                            }
                        }
                        if (dlx < thresh) {
                            converge_inner = true;
                            active_set.setConstant(true);
                        }
                    }   // end of inner coordinate descent loop
                    update_quadratic(X, beta, xm, xs, delta, ck, ri, d, n, m, W, r);
                    double diff_b = (xv.cwiseProduct((beta-beta_old).cwiseProduct(beta-beta_old))).maxCoeff();
                    if ( diff_b < thresh ) {
                        //// check kkt violation
                        gradient = ((X.array().colwise() * r.array()).colwise().sum().array() - (r.sum() * xm).array()) * xs.array();
                        int num_violations = 0;
                        for (int j = 0; j < p; j++) {
                            if (!strong_set[j]) {
                                if (gradient[j] > alpha1 * cmult[j] * lambda1_current) {
                                    strong_set[j] = true;
                                    num_violations += 1;
                                }
                            }
                        }
                        for (int j = p; j < (p+q); j++) {
                            if (!strong_set[j]) {
                                if (gradient[j] > alpha2 * cmult[j] * lambda2_current) {
                                    strong_set[j] = true;
                                    num_violations += 1;
                                }
                            }
                        }
                        if (num_violations == 0) {converge_outer = true;}
                    }
                }   // end of outer re-weighted loop
                lambda1_old = lambda1_current;

                //// hold the estimated beta, W, r of lambda1 at l1=1, for warm start of next lambda2 and lambda1 at l1=1
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
            lambda2_old = lambda2_current;
        }   // end of lambda2 loop

        return betas;

    }
}


int main()
{
    int error;
    MatrixXd dat;
    error = csvRead(dat, "/Users/dixinshen/Dropbox/hierr_cox_logit/PlayEigen/test_xy.csv", 15);
    cout << "Function call csvRead(), exit code: " <<  error << endl;
//    if (error == 0) {
//        cout << "Matrix (" << dat.rows() << "x" << dat.cols() << "):" << endl;4
//        cout << dat << endl;
//    }

    MatrixXd y(100,2);
    y.col(0) = dat.col(0);
    y.col(1) = dat.col(1);
    MatrixXd x(100, 200);
    x = dat.rightCols(200);
//    cout << "y: " << endl << y << endl;
//    cout << "x: " << endl << x << endl;

    int errorz;
    MatrixXd datz;
    errorz = csvRead(datz, "/Users/dixinshen/Dropbox/hierr_cox_logit/PlayEigen/test_z.csv", 15);
    cout << "Function call csvRead(), exit code: " <<  errorz << endl;
    MatrixXd z(200,200);
    z = datz;
//    cout << "z: " << endl << z << endl;

//    MatrixXd beta = cdl_cox(y, x, z);
//    cout << "beta_1_1: " << endl << beta.col(0).transpose() << endl;
//    cout << "beta_4_20: " << endl << beta.col(99).transpose() << endl;

//    MatrixXd z;
//    z.resize(10,2);
//    z.col(0) << 1,1,0,0,1,1,1,0,1,1;
//    z.col(1) << 0,0,1,1,0,0,0,1,0,0;

    MatrixXd beta_glmnet = cdl_cox(y, x);
    cout << "beta_1: " << endl << beta_glmnet.col(0).transpose() << endl;
    cout << "beta_10: " << endl << beta_glmnet.col(9).transpose() << endl;
    cout << "beta_30: " << endl << beta_glmnet.col(29).transpose() << endl;
//
//
    MatrixXd beta_ext = cdl_cox(y, x, z);
    cout << "beta_1_1: " << endl << beta_ext.col(0).transpose() << endl;
    cout << "beta_3_3: " << endl << beta_ext.col(42).transpose() << endl;
    cout << "beta_10_10: " << endl << beta_ext.col(189).transpose() << endl;
    cout << "beta_4_20: " << endl << beta_ext.col(383).transpose() << endl;


//    cout << "beta_1_1: " << endl;
//    for (int i = 0; i < (p+q); i++) {
//        cout << betas[0][0][i] << " ";
//    }
//    cout << endl << "beta_3_3: " << endl;
//    for (int i = 0; i < (p+q); i++) {
//        cout << betas[2][2][i] << " ";
//    }
//    cout << endl << "beta_10_19: " << endl;
//    for (int i = 0; i < (p+q); i++) {
//        cout << betas[9][18][i] << " ";
//    }


    return 0;
}