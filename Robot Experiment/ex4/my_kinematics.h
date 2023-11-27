#pragma once
#include <iostream>
#include <string>

using namespace std;

class my_kinematics
{
public:
    my_kinematics();

    void forward(double *q);
    void getForward();
    
    bool inverse(double T[4][4], bool *flag_Q);
    void getInverse();
    
    // Output of forward kinetic
    double T06[4][4];
    
    // Output of inverse kinetic
    double Q[6];

private:
    // Parameters of manipulator, mm
    double d2 = 140.5;
    double a2 = 408;
    double a3 = 376;
    double d4 = 19;
    double d5 = -102.5;
};