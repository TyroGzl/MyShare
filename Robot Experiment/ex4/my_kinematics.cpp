#include "my_exp4/my_kinematics.h"
#include <math.h>

my_kinematics::my_kinematics(void)
{
    cout << "Initialization completed." << endl;
}

void my_kinematics::forward(double *q)
{
    // Matlab formula
    // Match the name of parameters
    double q1 = q[0];
    double q2 = q[1];
    double q3 = q[2];
    double q4 = q[3];
    double q5 = q[4];
    double q6 = q[5];

    // First row: nx, ox, ax, px
    T06[0][0] = cos(q6) * (sin(q1) * sin(q5) + cos(q2 + q3 - q4) * cos(q1) * cos(q5)) + sin(q2 + q3 - q4) * cos(q1) * sin(q6);
    T06[0][1] = sin(q2 + q3 - q4) * cos(q1) * cos(q6) - sin(q6) * (sin(q1) * sin(q5) + cos(q2 + q3 - q4) * cos(q1) * cos(q5));
    T06[0][2] = cos(q5) * sin(q1) - cos(q2 + q3 - q4) * cos(q1) * sin(q5);
    T06[0][3] = d4 * sin(q1) - d2 * sin(q1) - d5 * sin(q2 + q3 - q4) * cos(q1) + a3 * cos(q2 + q3) * cos(q1) + a2 * cos(q1) * cos(q2);

    // Second row: ny, oy, ay, py
    T06[1][0] = sin(q2 + q3 - q4) * sin(q1) * sin(q6) - cos(q6) * (cos(q1) * sin(q5) - cos(q2 + q3 - q4) * cos(q5) * sin(q1));
    T06[1][1] = sin(q6) * (cos(q1) * sin(q5) - cos(q2 + q3 - q4) * cos(q5) * sin(q1)) + sin(q2 + q3 - q4) * cos(q6) * sin(q1);
    T06[1][2] = -cos(q1) * cos(q5) - cos(q2 + q3 - q4) * sin(q1) * sin(q5);
    T06[1][3] = d2 * cos(q1) - d4 * cos(q1) - d5 * sin(q2 + q3 - q4) * sin(q1) + a3 * cos(q2 + q3) * sin(q1) + a2 * cos(q2) * sin(q1);

    // Third row: nz, oz, az, pz
    T06[2][0] = cos(q2 + q3 - q4) * sin(q6) - sin(q2 + q3 - q4) * cos(q5) * cos(q6);
    T06[2][1] = cos(q2 + q3 - q4) * cos(q6) + sin(q2 + q3 - q4) * cos(q5) * sin(q6);
    T06[2][2] = sin(q2 + q3 - q4) * sin(q5);
    T06[2][3] = -d5 * cos(q2 + q3 - q4) - a3 * sin(q2 + q3) - a2 * sin(q2);

    // Fourth row: 0, 0, 0, 1
    T06[3][0] = 0;
    T06[3][1] = 0;
    T06[3][2] = 0;
    T06[3][3] = 1;
}

void my_kinematics::getForward()
{
    cout << "T06 = " << endl;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            cout.width(8);
            cout << T06[i][j] << " ";
        }
        cout << endl;
    }
}

bool my_kinematics::inverse(double T[4][4], bool *flag_Q)
{
    bool flag_Error = 0;

    // Q1
    if (flag_Error != 0 || T[0][3] + T[1][3] == 0)
    {
        flag_Error = 1;
        Q[0] = 0;
        cout << "error1" << endl;
    }
    else if (flag_Q[0] == 0)
    {
        Q[0] = +acos((d2 - d4) / sqrt(T[0][3] * T[0][3] + T[1][3] * T[1][3])) - atan(T[0][3] / T[1][3]);
    }
    else
    {
        Q[0] = -acos((d2 - d4) / sqrt(T[0][3] * T[0][3] + T[1][3] * T[1][3])) - atan(T[0][3] / T[1][3]);
    }

    // Q5
    if (flag_Error != 0)
    {
        Q[4] = 0;
        cout << "error5" << endl;
    }
    else if (flag_Q[2] == 0)
    {
        Q[4] = +acos(T[0][2] * sin(Q[0]) - T[1][2] * cos(Q[0]));
    }
    else
    {
        Q[4] = -acos(T[0][2] * sin(Q[0]) - T[1][2] * cos(Q[0]));
    }

    // Q6
    if (flag_Error != 0 || T[1][0] * cos(Q[0]) - T[0][0] * sin(Q[0]) == 0)
    {
        flag_Error = 1;
        Q[5] = 0;
        cout << "error6" << endl;
    }
    else
    {
        Q[5] = atan((T[0][1] * sin(Q[0]) - T[1][1] * cos(Q[0])) / (T[1][0] * cos(Q[0]) - T[0][0] * sin(Q[0])));
    }

    // Q3
    double m3 = T[0][3] * cos(Q[0]) + T[1][3] * sin(Q[0]) + d5 * (cos(Q[5]) * (T[0][1] * cos(Q[0]) + T[1][1] * sin(Q[0])) + sin(Q[5]) * (T[0][0] * cos(Q[0]) + T[1][0] * sin(Q[0])));
    double n3 = -T[2][3] - d5 * (T[2][1] * cos(Q[5]) + T[2][0] * sin(Q[5]));
    if (flag_Error != 0)
    {
        Q[2] = 0;
        cout << "error3" << endl;
    }
    else
    {
        if (flag_Q[1] == 0)
        {
            Q[2] = +acos((m3 * m3 + n3 * n3 - a2 * a2 - a3 * a3) / (2 * a2 * a3));
        }
        else
        {
            Q[2] = -acos((m3 * m3 + n3 * n3 - a2 * a2 - a3 * a3) / (2 * a2 * a3));
        }
    }

    // Q2
    if (flag_Error != 0 || m3 * (a2 + a3 * cos(Q[2])) + n3 * a3 * sin(Q[2]) == 0)
    {
        flag_Error = 1;
        Q[1] = 0;
        cout << "error2" << endl;
    }
    else
    {
        Q[1] = atan((n3 * (a2 + a3 * cos(Q[2])) - m3 * a3 * sin(Q[2])) / (m3 * (a2 + a3 * cos(Q[2])) + n3 * a3 * sin(Q[2])));
    }

    // Q4
    if (flag_Error != 0 || T[2][0] * sin(Q[5]) + T[2][1] * cos(Q[5]) == 0)
    {
        flag_Error = 1;
        Q[3] = 0;
        cout << "error4" << endl;
    }
    else
    {
        Q[3] = Q[1] + Q[2] - atan((cos(Q[5]) * (T[0][1] * cos(Q[0]) + T[1][1] * sin(Q[0])) + sin(Q[5]) * (T[0][0] * cos(Q[0]) + T[1][0] * sin(Q[0]))) / (T[2][0] * sin(Q[5]) + T[2][1] * cos(Q[5])));
    }
    return flag_Error;
}

void my_kinematics::getInverse()
{
    for (int i = 0; i < 6; i++)
    {
        cout << "Q" << i + 1 << " = " << Q[i] << endl;
    }
}