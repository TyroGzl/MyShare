#include <iostream>
#include "serviceinterface.h"


#define ROBOT_ADDR "192.168.1.107"
#define ROBOT_PORT 8899
#define M_PI 3.14159265358979323846

using namespace std;


double q[6] = {0.3, 0.3, 0.3, 0.3, 0.3, 0.3};


void printRoadPoint(const aubo_robot_namespace::wayPoint_S *wayPoint)
{
    std::cout << "pos.x=" << wayPoint->cartPos.position.x << std::endl;
    std::cout << "pos.y=" << wayPoint->cartPos.position.y << std::endl;
    std::cout << "pos.z=" << wayPoint->cartPos.position.z << std::endl;

    std::cout << "ori.w=" << wayPoint->orientation.w << std::endl;
    std::cout << "ori.x=" << wayPoint->orientation.x << std::endl;
    std::cout << "ori.y=" << wayPoint->orientation.y << std::endl;
    std::cout << "ori.z=" << wayPoint->orientation.z << std::endl;

    std::cout << "joint_1=" << wayPoint->jointpos[0] * 180.0 / M_PI << std::endl;
    std::cout << "joint_2=" << wayPoint->jointpos[1] * 180.0 / M_PI << std::endl;
    std::cout << "joint_3=" << wayPoint->jointpos[2] * 180.0 / M_PI << std::endl;
    std::cout << "joint_4=" << wayPoint->jointpos[3] * 180.0 / M_PI << std::endl;
    std::cout << "joint_5=" << wayPoint->jointpos[4] * 180.0 / M_PI << std::endl;
    std::cout << "joint_6=" << wayPoint->jointpos[5] * 180.0 / M_PI << std::endl;
}


int main(int argc, char **argv)
{
    ServiceInterface my_robot;
    int res = -1;
    aubo_robot_namespace::wayPoint_S wayPoint;
    aubo_robot_namespace::wayPoint_S wayPoint2;

    //登录
    cout << "正在登录 ..." << endl;
    res = my_robot.robotServiceLogin(ROBOT_ADDR, ROBOT_PORT, "AUBO", "123456");
    if (!res)
        cout << "登录成功 : " << res << endl
             << endl;
    else
        cout << "登录失败 : " << res << endl
             << endl;
    
    //正解
    cout << "计算正解 ..." << endl;
    res = my_robot.robotServiceRobotFk(q, 6, wayPoint);
    printRoadPoint(&wayPoint);
    
    //逆解
    cout << "计算逆解 ..." << endl;
    double startPointJointAngle[6] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    res = my_robot.robotServiceRobotIk(startPointJointAngle, wayPoint.cartPos.position, wayPoint.orientation, wayPoint2);
    printRoadPoint(&wayPoint2);
    
    //关节运动 关节运动
    my_robot.robotServiceJointMove(q, true);
    
    //登出
    cout << "正在登出 ..." << endl;
    res = my_robot.robotServiceLogout();
    if (!res)
        cout << "登出成功 : " << res << endl
             << endl;
    else
        cout << "登出失败 : " << res << endl
             << endl;
    
    //等待退出
    cout << "please enter to exit" << endl;
    getchar();
    return 0;
}