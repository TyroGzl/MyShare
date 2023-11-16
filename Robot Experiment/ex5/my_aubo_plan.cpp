#include <iostream>
#include <math.h>
#include "serviceinterface.h"

#define ROBOT_ADDR "192.168.1.107"
#define ROBOT_PORT 8899
#define M_PI 3.14159265358979323846
#define DEG2RAD(x) x *M_PI / 180

using namespace std;

aubo_robot_namespace::Pos lineStartPos = {-0.4, -0.3, 0.4};
aubo_robot_namespace::Pos lineEndPos = {-0.4, 0.2, 0.2};
aubo_robot_namespace::Pos homePos = {-0.4, -0.1215, 0.5476};
aubo_robot_namespace::Pos arcPos1 = {-0.4, -0.0215, 0.4476};
aubo_robot_namespace::Pos arcPos2 = {-0.4, -0.1215, 0.3476};

int nPoints = 5;

double startPointJointAngle[6] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
aubo_robot_namespace::JointVelcAccParam maxJointV = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
aubo_robot_namespace::JointVelcAccParam maxJointA = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2};

double maxEndV = 0.2;
double maxEndA = 0.2;

int main(int argc, char **argv)
{
    ServiceInterface my_robot;
    int res = -1;
    aubo_robot_namespace::wayPoint_S wayPoint;
    aubo_robot_namespace::wayPoint_S wayPoint2;
    //登录
    cout << "正在登录..." << endl;
    res = my_robot.robotServiceLogin(ROBOT_ADDR, ROBOT_PORT, "AUBO", "123456");
    if (!res)
        cout << "登录成功: " << res << endl
             << endl;
    else
        cout << "登录失败: " << res << endl
             << endl;

    //设置最大速度、加速度
    my_robot.robotServiceSetGlobalMoveJointMaxVelc(maxJointV);
    my_robot.robotServiceSetGlobalMoveJointMaxAcc(maxJointA);
    my_robot.robotServiceSetGlobalMoveEndMaxLineVelc(maxEndV);
    my_robot.robotServiceSetGlobalMoveEndMaxLineAcc(maxEndA);

    //获取初始姿态
    aubo_robot_namespace::Rpy rpy;
    rpy.rx = DEG2RAD(180);
    rpy.ry = DEG2RAD(0);
    rpy.rz = DEG2RAD(-90);
    aubo_robot_namespace::Ori orientation;
    my_robot.RPYToQuaternion(rpy, orientation);

    int flag_main = 1;
    int flag_switch = 0;
    aubo_robot_namespace::wayPoint_S nextWayPoint;
    aubo_robot_namespace::Pos nextPos = lineStartPos;
    aubo_robot_namespace::Pos deltaPos;
    while (flag_main)
    {
        cout << "================================================" << endl;
        cout << "1: 关节运动测试（JointMove）" << endl;
        cout << "2: 直线运动测试（LineMove）" << endl;
        cout << "3: 直线插值测试（JointMove）" << endl;
        cout << "4: 直线插值测试（TrackMove）" << endl;
        cout << "5: 圆弧运动测试（TrackMove）" << endl;
        cout << "6: 圆弧插值测试（JointMove）" << endl;
        cout << "7: 圆弧插值测试（TrackMove）" << endl;
        cout << "0: 退出" << endl;
        cout << "================================================" << endl;
        cout << "Please input:";
        cin >> flag_switch;
        cout << endl;

        switch (flag_switch)
        {
        case 1:
            // ------------------------------------------------------------------------------------ //
            // ----------------------- 关节运动测试（JointMove）----------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "关节运动测试（JointMove）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            my_robot.robotServiceRobotIk(startPointJointAngle, lineStartPos, orientation, nextWayPoint);
            my_robot.robotServiceJointMove(nextWayPoint, true);
            cout << "按回车键运动至终点..." << endl;
            getchar();
            my_robot.robotServiceRobotIk(startPointJointAngle, lineEndPos, orientation, nextWayPoint);
            my_robot.robotServiceJointMove(nextWayPoint, true);
            break;
        case 2:
            // ------------------------------------------------------------------------------------ //
            // ----------------------- 直线运动测试（LineMove）----------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "直线运动测试（LineMove）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            my_robot.robotServiceRobotIk(startPointJointAngle, lineStartPos, orientation, nextWayPoint);
            my_robot.robotServiceJointMove(nextWayPoint, true);
            cout << "按回车键运动至终点..." << endl;
            getchar();
            my_robot.robotServiceRobotIk(startPointJointAngle, lineEndPos, orientation, nextWayPoint);
            my_robot.robotServiceLineMove(nextWayPoint, true);
            break;
        case 3:
            // ------------------------------------------------------------------------------------ //
            // ----------------------- 直线插值测试（JointMove）----------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "直线插值测试（JointMove）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            my_robot.robotServiceRobotIk(startPointJointAngle, lineStartPos, orientation, nextWayPoint);
            my_robot.robotServiceJointMove(nextWayPoint, true);
            cout << "按回车键运动至终点..." << endl;
            getchar();
            nextPos = lineStartPos;
            deltaPos.x = (lineEndPos.x - lineStartPos.x) / nPoints;
            deltaPos.y = (lineEndPos.y - lineStartPos.y) / nPoints;
            deltaPos.z = (lineEndPos.z - lineStartPos.z) / nPoints;
            for (int i = 0; i < nPoints; i++)
            {
                nextPos.x += deltaPos.x;
                nextPos.y += deltaPos.y;
                nextPos.z += deltaPos.z;
                my_robot.robotServiceRobotIk(startPointJointAngle, nextPos, orientation, nextWayPoint);
                my_robot.robotServiceJointMove(nextWayPoint, true);
            }
            break;
        case 4:
            // ------------------------------------------------------------------------------------ //
            // ---------------------- 直线插值测试（TrackMove）----------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "直线插值测试（TrackMove）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            my_robot.robotServiceRobotIk(startPointJointAngle, lineStartPos, orientation, nextWayPoint);
            my_robot.robotServiceJointMove(nextWayPoint, true);
            cout << "按回车键运动至终点..." << endl;
            getchar();
            nextPos = lineStartPos;
            deltaPos.x = (lineEndPos.x - lineStartPos.x) / nPoints;
            deltaPos.y = (lineEndPos.y - lineStartPos.y) / nPoints;
            deltaPos.z = (lineEndPos.z - lineStartPos.z) / nPoints;
            my_robot.robotServiceClearGlobalWayPointVector();
            my_robot.robotServiceAddGlobalWayPoint(nextWayPoint);
            for (int i = 0; i < nPoints; i++)
            {
                nextPos.x += deltaPos.x;
                nextPos.y += deltaPos.y;
                nextPos.z += deltaPos.z;
                my_robot.robotServiceRobotIk(startPointJointAngle, nextPos, orientation, nextWayPoint);
                my_robot.robotServiceAddGlobalWayPoint(nextWayPoint);
            }
            my_robot.robotServiceTrackMove(aubo_robot_namespace::CARTESIAN_CUBICSPLINE, true);
            //my_robot.robotServiceTrackMove(aubo_robot_namespace::JIONT_CUBICSPLINE, true);
            break;
        case 5:
            // ------------------------------------------------------------------------------------ //
            // ---------------------- 圆弧运动测试（TrackMove）----------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "圆弧运动测试（TrackMove）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            my_robot.robotServiceRobotIk(startPointJointAngle, homePos, orientation, nextWayPoint);
            my_robot.robotServiceJointMove(nextWayPoint, true);
            cout << "按回车键运动至终点..." << endl;
            getchar();
            my_robot.robotServiceClearGlobalWayPointVector();
            my_robot.robotServiceAddGlobalWayPoint(nextWayPoint);
            my_robot.robotServiceRobotIk(startPointJointAngle, arcPos1, orientation, nextWayPoint);
            my_robot.robotServiceAddGlobalWayPoint(nextWayPoint);
            my_robot.robotServiceRobotIk(startPointJointAngle, arcPos2, orientation, nextWayPoint);
            my_robot.robotServiceAddGlobalWayPoint(nextWayPoint);
            my_robot.robotServiceSetGlobalCircularLoopTimes(0);
            my_robot.robotServiceTrackMove(aubo_robot_namespace::ARC_CIR, true);
            break;
        case 6:
            // ------------------------------------------------------------------------------------ //
            // ----------------------- 圆弧插值测试（JointMove）----------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "圆弧插值测试（JointMove）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            my_robot.robotServiceRobotIk(startPointJointAngle, homePos, orientation, nextWayPoint);
            my_robot.robotServiceJointMove(nextWayPoint, true);
            cout << "按回车键运动至终点..." << endl;
            getchar();
            for (int i = 0; i < nPoints; i++)
            {
                double theta = (i + 1) * M_PI / nPoints;
                double r = (homePos.z - arcPos2.z) / 2;
                nextPos.x = homePos.x;
                nextPos.y = homePos.y + r * sin(theta);
                nextPos.z = homePos.z - r + r * sin(theta + M_PI / 2);
                my_robot.robotServiceRobotIk(startPointJointAngle, nextPos, orientation, nextWayPoint);
                my_robot.robotServiceJointMove(nextWayPoint, true);
            }
            break;
        case 7:
            // ------------------------------------------------------------------------------------ //
            // ---------------------- 圆弧插值测试（TrackMove）----------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "圆弧插值测试（TrackMove）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            my_robot.robotServiceRobotIk(startPointJointAngle, homePos, orientation, nextWayPoint);
            my_robot.robotServiceJointMove(nextWayPoint, true);
            cout << "按回车键运动至终点..." << endl;
            getchar();
            my_robot.robotServiceClearGlobalWayPointVector();
            my_robot.robotServiceAddGlobalWayPoint(nextWayPoint);
            for (int i = 0; i < nPoints; i++)
            {
                double theta = (i + 1) * M_PI / nPoints;
                double r = (homePos.z - arcPos2.z) / 2;
                nextPos.x = homePos.x;
                nextPos.y = homePos.y + r * sin(theta);
                nextPos.z = homePos.z - r + r * sin(theta + M_PI / 2);
                my_robot.robotServiceRobotIk(startPointJointAngle, nextPos, orientation, nextWayPoint);
                my_robot.robotServiceAddGlobalWayPoint(nextWayPoint);
            }
            my_robot.robotServiceTrackMove(aubo_robot_namespace::CARTESIAN_CUBICSPLINE, true);
            //my_robot.robotServiceTrackMove(aubo_robot_namespace::JIONT_CUBICSPLINE, true);
            break;
        case 0:
            flag_main = 0;
            break;
        }
        cout << endl;
    }
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