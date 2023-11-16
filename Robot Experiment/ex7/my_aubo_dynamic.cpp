#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "AuboRobotMetaType.h"
#include "serviceinterface.h"

#define ROBOT_ADDR "192.168.1.107"
#define ROBOT_PORT 8899
#define DoF 6
#define M_PI 3.14159265358979323846

using namespace std;

double g = 9.8;

double jointPosRealTime[DoF];
double jointPosOffset[DoF] = {0, -1.5708, 0, -1.5708, 0, 0};
double jointPosSign[DoF] = {1, 1, -1, 1, 1, 1}; //坐标轴方向校正
double jointCurRealTime[DoF];
double torqueRealTime[DoF];
double gravityQRealtime[DoF];
double endForceCalculate[DoF];

//机械臂参数
int gearRatio[DoF] = {121, 121, 121, 101, 101, 101};
double torquePara[DoF] = {114.65, 114.65, 114.65, 140.996, 140.996, 140.996};
int torqueDirection[DoF] = {1, -1, 1, -1, -1, -1};

double d2 = 0.1405;
double a2 = 0.408;
double a3 = 0.376;
double d4 = 0.019;
double d5 = -0.1025;

double m1 = 5.05;
double m2 = 11.28;
double m3 = 2.88;
double m4 = 1.62;
double m5 = 1.62;
double m6 = 0.4;

double x2 = 0.204;
double x3 = 0.2948;
double z3 = -0.1025;
double y4 = 0.025;
double z6 = -(0.094 - 0.0071);

void gravityQ(double q[DoF], double Gq[DoF])
{
    double q1 = q[0];
    double q2 = q[1];
    double q3 = q[2];
    double q4 = q[3];
    double q5 = q[4];
    double q6 = q[5];

    Gq[0] = 0;
    Gq[1] = g * m5 * (a3 * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + a2 * cos(q2) - d5 * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)))) + g * m4 * (a3 * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + a2 * cos(q2) + y4 * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)))) + g * m3 * (x3 * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + a2 * cos(q2)) + g * m6 * (a3 * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + a2 * cos(q2) - d5 * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3))) - z6 * sin(q5) * (cos(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + sin(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)))) + g * m2 * x2 * cos(q2);
    Gq[2] = g * m5 * (a3 * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) - d5 * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)))) - g * m6 * (d5 * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3))) - a3 * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + z6 * sin(q5) * (cos(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + sin(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)))) + g * m4 * (a3 * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + y4 * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)))) + g * m3 * x3 * (cos(q2) * cos(q3) - sin(q2) * sin(q3));
    Gq[3] = g * m6 * (d5 * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3))) + z6 * sin(q5) * (cos(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + sin(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)))) + d5 * g * m5 * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3))) - g * m4 * y4 * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)));
    Gq[4] = -g * m6 * z6 * cos(q5) * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)));
    Gq[5] = 0;
}

void invTransJacobianMatrix(double q[DoF], double temp[DoF])
{
    double q1 = q[0];
    double q2 = q[1];
    double q3 = q[2];
    double q4 = q[3];
    double q5 = q[4];
    double q6 = q[5];

    double J[6][6], JT[6][6];
    // Jv
    // Jv1
    J[0][0] = (d4 - d2) * cos(q1) + d5 * sin(q1) * (cos(q4) * (cos(q2) * sin(q3) + sin(q2) * cos(q3)) - sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3))) - a3 * sin(q1) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) - a2 * sin(q1) * cos(q2);
    J[1][0] = (d4 - d2) * sin(q1) - d5 * cos(q1) * (cos(q4) * (cos(q2) * sin(q3) + sin(q2) * cos(q3)) + sin(q4) * (sin(q2) * sin(q3) - cos(q2) * cos(q3))) - a3 * cos(q1) * (sin(q2) * sin(q3) - cos(q2) * cos(q3)) + a2 * cos(q1) * cos(q2);
    J[2][0] = 0;
    // Jv2
    J[0][1] = -d5 * cos(q1) * (cos(q4) * (-sin(q2) * sin(q3) + cos(q2) * cos(q3)) - sin(q4) * (-sin(q2) * cos(q3) - cos(q2) * sin(q3))) + a3 * cos(q1) * (-sin(q2) * cos(q3) - cos(q2) * sin(q3)) - a2 * cos(q1) * sin(q2);
    J[1][1] = -d5 * sin(q1) * (cos(q4) * (-sin(q2) * sin(q3) + cos(q2) * cos(q3)) + sin(q4) * (cos(q2) * sin(q3) + sin(q2) * cos(q3))) - a3 * sin(q1) * (cos(q2) * sin(q3) + sin(q2) * cos(q3)) - a2 * sin(q1) * sin(q2);
    J[2][1] = -a3 * (-sin(q2) * sin(q3) + cos(q2) * cos(q3)) - a2 * cos(q2) - d5 * (cos(q4) * (-sin(q2) * cos(q3) - cos(q2) * sin(q3)) + sin(q4) * (-sin(q2) * sin(q3) + cos(q2) * cos(q3)));
    // Jv3
    J[0][2] = -d5 * cos(q1) * (cos(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) - sin(q4) * (-cos(q2) * sin(q3) - sin(q2) * cos(q3))) + a3 * cos(q1) * (-cos(q2) * sin(q3) - sin(q2) * cos(q3));
    J[1][2] = -d5 * sin(q1) * (cos(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) + sin(q4) * (sin(q2) * cos(q3) + cos(q2) * sin(q3))) - a3 * sin(q1) * (sin(q2) * cos(q3) + cos(q2) * sin(q3));
    J[2][2] = -a3 * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) - d5 * (cos(q4) * (-cos(q2) * sin(q3) -
                                                                               sin(q2) * cos(q3)) +
                                                                    sin(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)));
    // Jv4
    J[0][3] = -d5 * cos(q1) * (-cos(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) - sin(q4) * (cos(q2) * sin(q3) + sin(q2) * cos(q3)));
    J[1][3] = -d5 * sin(q1) * (cos(q4) * (sin(q2) * sin(q3) - cos(q2) * cos(q3)) - sin(q4) * (cos(q2) * sin(q3) + sin(q2) * cos(q3)));
    J[2][3] = -d5 * (cos(q4) * (cos(q2) * sin(q3) + sin(q2) * cos(q3)) - sin(q4) * (cos(q2) * cos(q3) -
                                                                                    sin(q2) * sin(q3)));
    // Jv5
    J[0][4] = 0;
    J[1][4] = 0;
    J[2][4] = 0;
    // Jv6
    J[0][5] = 0;
    J[1][5] = 0;
    J[2][5] = 0;

    // Jw
    // Jw1
    J[3][0] = 0;
    J[4][0] = 0;
    J[5][0] = 1;
    // Jw2
    J[3][1] = -sin(q1);
    J[4][1] = cos(q1);
    J[5][1] = 1;
    // Jw3
    J[3][2] = -sin(q1);
    J[4][2] = cos(q1);
    J[5][2] = 1;
    // Jw4
    J[3][3] = sin(q1);
    J[4][3] = -cos(q1);
    J[5][3] = 1;
    // Jw5
    J[3][4] = sin(q4) * (cos(q1) * cos(q2) * cos(q3) - cos(q1) * sin(q2) * sin(q3)) -
              cos(q4) * (cos(q1) * cos(q2) * sin(q3) + cos(q1) * cos(q3) * sin(q2));
    J[4][4] = -cos(q4) * (cos(q2) * sin(q1) * sin(q3) + cos(q3) * sin(q1) * sin(q2)) -
              sin(q4) * (sin(q1) * sin(q2) * sin(q3) - cos(q2) * cos(q3) * sin(q1));
    J[5][4] = -cos(q4) * (cos(q2) * cos(q3) - sin(q2) * sin(q3)) - sin(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2));
    // Jw6
    J[3][5] = cos(q5) * sin(q1) - sin(q5) * (cos(q4) * (cos(q1) * cos(q2) * cos(q3) - cos(q1) * sin(q2) * sin(q3)) +
                                             sin(q4) * (cos(q1) * cos(q2) * sin(q3) + cos(q1) * cos(q3) * sin(q2)));
    J[4][5] = sin(q5) * (cos(q4) * (sin(q1) * sin(q2) * sin(q3) - cos(q2) * cos(q3) * sin(q1)) -
                         sin(q4) * (cos(q2) * sin(q1) * sin(q3) + cos(q3) * sin(q1) * sin(q2))) -
              cos(q1) * cos(q5);
    J[5][5] = sin(q5) * (cos(q4) * (cos(q2) * sin(q3) + cos(q3) * sin(q2)) - sin(q4) * (cos(q2) * cos(q3) -
                                                                                        sin(q2) * sin(q3)));
    
    // transpose
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            JT[i][j] = J[j][i];
        }
    }
    
    // inverse
    int *is, *js, i, j, k;
    int n = 6;
    double d, p;
    
    is = new int[n];
    
    js = new int[n];
    for (k = 0; k < n; k++)
    {
        d = 0.0;
        for (i = k; i < n; i++)
            for (j = k; j < n; j++)
            {
                p = fabs(JT[i][j]);
                if (p > d)
                {
                    d = p;
                    is[k] = i;
                    js[k] = j;
                }
            }
        if (d + 1.0 == 1.0)
        {
            delete[] is, js;
            std::cout << "\nA为奇异矩阵!没有逆矩阵。" << std::endl;
            std::exit(1);
        }
        if (is[k] != k) //全选主元
            for (j = 0; j < n; j++)
            {
                p = JT[k][j];
                JT[k][j] = JT[is[k]][j];
                JT[is[k]][j] = p;
            }
        if (js[k] != k)
            for (i = 0; i < n; i++)
            {
                p = JT[i][k];
                JT[i][k] = JT[i][js[k]];
                JT[i][js[k]] = p;
            }
        JT[k][k] = 1.0 / JT[k][k];
        for (j = 0; j < n; j++)
            if (j != k)
                JT[k][j] = JT[k][j] * JT[k][k];
        for (i = 0; i < n; i++)
            if (i != k)
                for (j = 0; j < n; j++)
                    if (j != k)
                        JT[i][j] = JT[i][j] - JT[i][k] * JT[k][j];
        for (i = 0; i < n; i++)
            if (i != k)
                JT[i][k] = -JT[i][k] * JT[k][k];
    }
    for (k = n - 1; k >= 0; k--)
    {
        if (js[k] != k)
            for (j = 0; j < n; j++)
            {
                p = JT[k][j];
                JT[k][j] = JT[js[k]][j];
                JT[js[k]][j] = p;
            }
        if (is[k] != k)
            for (i = 0; i < n; i++)
            {
                p = JT[i][k];
                JT[i][k] = JT[i][is[k]];
                JT[i][is[k]] = p;
            }
    }
    delete[] is, js;

    // JTI*temp=F
    for (int i = 0; i < 6; i++)
    {
        endForceCalculate[i] = 0;
        for (int j = 0; j < 6; j++)
        {
            endForceCalculate[i] += JT[i][j] * temp[j];
        }
    }
}

void realTimeJointStatusCallback(const aubo_robot_namespace::JointStatus *jointStatusPtr, int size, void *arg)
{
    (void)arg;
    for (int i = 0; i < size; i++)
    { /*
std::cout<<"Joint_ID:"<<i<<" | " ;
std::cout<<"Joint_Current:"<<jointStatusPtr[i].jointCurrentI<<" | ";
std::cout<<"Joint_Velc:"<<jointStatusPtr[i].jointSpeedMoto<<" | ";
std::cout<<"Joint_Pos:"<<jointStatusPtr[i].jointPosJ<<" "<<" ~
"<<jointStatusPtr[i].jointPosJ*180.0/3.1415926 << " | ";
std::cout<<"Joint_Speed:" <<jointStatusPtr[i].jointCurVol<<" | ";
std::cout<<"Joint_Temp:" <<jointStatusPtr[i].jointCurTemp<<" | ";
std::cout<<"Joint_Tag_Current:" <<jointStatusPtr[i].jointTagCurrentI<<" | ";
std::cout<<"Joint_Tag_Speed:" <<jointStatusPtr[i].jointTagSpeedMoto<<" | ";
std::cout<<"Joint_Tag_Pos:" <<jointStatusPtr[i].jointTagPosJ<<" | ";
std::cout<<"Joint_Err:" <<jointStatusPtr[i].jointErrorNum<<std::endl<<std::endl;
*/
        jointCurRealTime[i] = jointStatusPtr[i].jointCurrentI;
        jointPosRealTime[i] = (jointStatusPtr[i].jointPosJ + jointPosOffset[i]) * jointPosSign[i];
    }
}

int main(int argc, char **argv)
{
    ServiceInterface my_robot;
    int res = -1;

    //登录
    cout << "正在登录..." << endl;
    res = my_robot.robotServiceLogin(ROBOT_ADDR, ROBOT_PORT, "AUBO", "123456");
    if (!res)
        cout << "登录成功: " << res << endl
             << endl;
    else
        cout << "登录失败: " << res << endl
             << endl;
    //利用SDK读取关节电流，并转换成力矩
    for (int i = 0; i < 20; i++)
    {
        my_robot.robotServiceRegisterRealTimeJointStatusCallback(realTimeJointStatusCallback, NULL);
    }
    /*for(int i = 0; i < 6; i++)
{
cout << "Joint_ID: " << i+1 << " | " ;
cout << "Joint_Current: " << jointCurRealTime[i] << " | " << endl;
}
cout << endl;*/
    /*for(int i = 0; i < 6; i++)
{
cout << "Joint_ID: " << i+1 << " | " ;
cout << "Joint_Pos: " << jointPosRealTime[i] << " | " << endl;
}
cout << endl;*/
    for (int i = 0; i < 6; i++)
    {
        torqueRealTime[i] = torqueDirection[i] * jointCurRealTime[i] / torquePara[i];
    }
    /*for(int i = 0; i < 6; i++)
{
cout << "Joint_ID: "<< i+1 << " | " ;
cout << "Joint_TQ: " << torqueRealTime[i] << " | " << endl;
}
cout << endl;*/
    //计算重力项G(q)，与关节力矩进行对比
    gravityQ(jointPosRealTime, gravityQRealtime);

    for (int i = 0; i < 6; i++)
    {
        cout << "Joint_ID: " << i + 1 << " | ";
        cout << "Joint_T: " << torqueRealTime[i] << " | "
             << "Joint_Gq: " << gravityQRealtime[i] << " | "
             << endl;
    }
    cout << endl;
    //计算理论外力大小

    double temp[6];
    for (int i = 0; i < 6; i++)
    {
        temp[i] = torqueRealTime[i] - gravityQRealtime[i];
    }

    invTransJacobianMatrix(jointPosRealTime, temp);
    
    for (int i = 0; i < 6; i++)
    {
        cout << "End_Force: " << endForceCalculate[i] << " | " << endl;
    }
    cout << endl;
    
    //登出
    cout << "正在登出..." << endl;
    res = my_robot.robotServiceLogout();
    if (!res)
        cout << "登出成功: " << res << endl
             << endl;
    else
        cout << "登出失败: " << res << endl
             << endl;
    return 0;
}