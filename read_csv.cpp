//
// Created by Dixin Shen on 8/28/19.
//

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using namespace std;

int csvRead(MatrixXd& outputMatrix, const string& fileName, const streamsize dPrec) {
    ifstream inputData;
    inputData.open(fileName);
    cout.precision(dPrec);
    if (!inputData)
        return -1;
    string fileline, filecell;
    unsigned int prevNoOfCols = 0, noOfRows = 0, noOfCols = 0;
    while (getline(inputData, fileline)) {
        noOfCols = 0;
        stringstream linestream(fileline);
        while (getline(linestream, filecell, ',')) {
            try {
                stod(filecell);
            }
            catch (...) {
                return -1;
            }
            noOfCols++;
        }
        if (noOfRows++ == 0)
            prevNoOfCols = noOfCols;
        if (prevNoOfCols != noOfCols)
            return -1;
    }
    inputData.close();
    outputMatrix.resize(noOfRows, noOfCols);
    inputData.open(fileName);
    noOfRows = 0;
    while (getline(inputData, fileline)) {
        noOfCols = 0;
        stringstream linestream(fileline);
        while (getline(linestream, filecell, ',')) {
            outputMatrix(noOfRows, noOfCols++) = stod(filecell);
        }
        noOfRows++;
    }
    return 0;
}

//int main()
//{
//    int error;
//    MatrixXd A;
//    error = csvRead(A, "/Users/dixinshen/Dropbox/hierr_cox_logit/PlayEigen/CoxExample.csv",20);
//    cout << error << endl;
//    if (error == 0) {
//        cout << "Matrix (" << A.rows() << "x" << A.cols() << "):" << endl;
//        cout << A << endl;
//    }
//    return 0;
//}
