#include "my_exp4/my_kinematics.h"

// Set joint angles, rad
double q[6] = {0.3, 0.3, 0.3, 0.3, 0.3, 0.3};

// Set homogeneous matrix of position and posture
double Te1[4][4] = {
    {0.996555, -0.0649952, 0.0515054, 477.381},
    {0.0649952, 0.226404, -0.971862, 399.243},
    {0.0515054, 0.971862, 0.229849, -222.046},
    {0, 0, 0, 1}};

double Te2[4][4] = {
    {0.996555, -0.0649952, 0.0515054, 477.381},
    {0.0649952, 0.226404, -0.971862, 399.243},
    {0.0515054, 0.971862, 0.229849, -22.046},
    {0, 0, 0, 1}};

// Strategy of choosing Q1, Q3 and Q5
// 0 for positive and 1 for negative
bool flag_Q[3] = {0, 0, 0};

int main(int argc, char **argv)
{
    int flag_main = 1;
    int flag_switch = 0;

    my_kinematics kin;
    while (flag_main)
    {
        cout << "=============================" << endl;
        cout << "1: diy fk" << endl;
        cout << "2: diy ik" << endl;
        cout << "3: verify through fk and ik" << endl;
        cout << "0: exit" << endl;
        cout << "=============================" << endl;
        cout << "Please input a number：";
        cin >> flag_switch;
        cout << endl;

        switch (flag_switch)
        {
        case 1: // Calculate T06 from q1~q6
            cout << "1: diy fk " << endl;
            cout << "input：" << endl;
            for (int i = 0; i < 6; i++)
                cout << "q" << i + 1 << " = " << q[i] << endl;
            cout << endl;
            kin.forward(q);
            cout << "output：" << endl;
            kin.getForward();
            break;

        case 2: // Calculate Q1~Q6 from Te
            cout << "2: diy ik" << endl;
            if (kin.inverse(Te1, flag_Q))
                cout << "Singularity reached." << endl;
            else
                kin.getInverse();
            break;

        case 3: // Calcualte Q1~Q6 from q1~q6
            cout << "3: verify through fk and ik" << endl;
            cout << "input：" << endl;
            for (int i = 0; i < 6; i++)
                cout << "q" << i + 1 << " = " << q[i] << endl;
            cout << endl;
            kin.forward(q);
            cout << "output：" << endl;
            if (kin.inverse(kin.T06, flag_Q)) // input from the output of "1: diy fk"
                cout << "Singularity reached." << endl;
            else
                kin.getInverse();
            break;
            
        case 0:
            flag_main = 0;
            break;
        }
        cout << endl;
    }
    return 0;
}