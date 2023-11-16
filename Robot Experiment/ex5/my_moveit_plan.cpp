#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <tf/LinearMath/Quaternion.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#define M_PI 3.14159265358979323846
#define DEG2RAD(x) x *M_PI / 180
using namespace std;
double lineStartPos[3] = {-0.4, -0.3, 0.4};
double lineEndPos[3] = {-0.4, 0.2, 0.2};
double homePos[3] = {-0.4, -0.1215, 0.5476};
double arcPos1[3] = {-0.4, -0.0215, 0.4476};
double arcPos2[3] = {-0.4, -0.1215, 0.3476};
int nPoints = 5;
double maxV = 0.2;
double maxA = 0.2;
/*
double scale = 2;
int n_joints = plan.trajectory_.joint_trajectory.joint_names.size();
for (int i = 0; i < plan.trajectory_.joint_trajectory.points.size(); i++)
{
plan.trajectory_.joint_trajectory.points[i].time_from_start *= 1 / scale;
for (int j = 0; j < n_joints; j++)
{
plan.trajectory_.joint_trajectory.points[i].velocities[j] *= scale;
plan.trajectory_.joint_trajectory.points[i].accelerations[j] *= scale * scale;
}
}
*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo");
    ros::NodeHandle nh;
    ros::Publisher plan_positions_pub = nh.advertise<sensor_msgs::JointState>("/robot_state", 100);
    sensor_msgs::JointState pub_robot_state;
    ros::Time init_time(0.0);
    // Start a thread
    ros::AsyncSpinner spinner(1);
    spinner.start();
    // Define the planning group name
    static const std::string PLANNING_GROUP = "manipulator_i5";
    // Create a planning group interface object and set up a planning group
    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);
    move_group.setPoseReferenceFrame("base_link");
    // Create a planning scene interface object
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    // Create a robot model information object
    const robot_state::JointModelGroup *joint_model_group =
        move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);
    // 创建RobotState对象
    moveit::core::RobotStatePtr robot_state = move_group.getCurrentState();
    /*
moveit::planning_interface::MoveGroupInterface::Plan joinedPlan;
robot_trajectory::RobotTrajectory rt(move_group.getCurrentState()->getRobotModel(), "manipulator_i5");
trajectory_processing::IterativeParabolicTimeParameterization iptp;
*/
    //设置最大速度、加速度
    move_group.setMaxVelocityScalingFactor(maxV);
    move_group.setMaxAccelerationScalingFactor(maxA);
    //获取初始姿态
    tf::Quaternion orientation;
    orientation.setRPY(DEG2RAD(180), DEG2RAD(0), DEG2RAD(-90));
    geometry_msgs::Pose next_pose;
    next_pose.orientation.x = orientation.x();
    next_pose.orientation.y = orientation.y();
    next_pose.orientation.z = orientation.z();
    next_pose.orientation.w = orientation.w();
    double deltaPos[3] = {0};
    for (int i = 0; i < 3; i++)
        deltaPos[i] = (lineEndPos[i] - lineStartPos[i]) / nPoints;
    std::vector<geometry_msgs::Pose> waypoints;
    moveit_msgs::RobotTrajectory trajectory;
    double fraction;
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    int flag_main = 1;
    int flag_switch = 0;
    while (flag_main)
    {
        cout << "================================================" << endl;
        cout << "1: 关节运动测试" << endl;
        cout << "2: 直线插值测试" << endl;
        cout << "3: 直线插值测试（computeCartesianPath）" << endl;
        cout << "4: 圆弧插值测试" << endl;
        cout << "5: 圆弧插值测试（computeCartesianPath）" << endl;
        cout << "0: 退出" << endl;
        cout << "================================================" << endl;
        cout << "Please input:";
        cin >> flag_switch;
        cout << endl;
        switch (flag_switch)
        {
        case 1:
            // ------------------------------------------------------------------------------------ //
            // ------------------------------- 关节运动测试 -------------------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "关节运动测试（JointMove）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            //move_group.setPositionTarget (lineStartPos[0], lineStartPos[1], lineStartPos[2]);
            //move_group.move();
            next_pose.position.x = lineStartPos[0];
            next_pose.position.y = lineStartPos[1];
            next_pose.position.z = lineStartPos[2];
            move_group.setPoseTarget(next_pose);
            move_group.move();
            cout << "按回车键运动至终点..." << endl;
            getchar();
            //move_group.setPositionTarget (lineStartPos[0], lineStartPos[1], lineStartPos[2]);
            //move_group.move();
            next_pose.position.x = lineEndPos[0];
            next_pose.position.y = lineEndPos[1];
            next_pose.position.z = lineEndPos[2];
            move_group.setPoseTarget(next_pose);
            move_group.move();
            break;
        case 2:
            // ------------------------------------------------------------------------------------ //
            // ------------------------------- 直线插值测试 -------------------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "直线插值测试：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            next_pose.position.x = lineStartPos[0];
            next_pose.position.y = lineStartPos[1];
            next_pose.position.z = lineStartPos[2];
            move_group.setPoseTarget(next_pose);
            move_group.move();
            cout << "按回车键运动至终点..." << endl;
            getchar();
            for (int i = 0; i < nPoints; i++)
            {
                next_pose.position.x += deltaPos[0];
                next_pose.position.y += deltaPos[1];
                next_pose.position.z += deltaPos[2];
                move_group.setPoseTarget(next_pose);
                move_group.move();
                //move_group.asyncMove();
            }
            break;
        case 3:
            // ------------------------------------------------------------------------------------ //
            // ---------------- 直线插值测试（computeCartesianPath）---------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "直线插值测试（computeCartesianPath）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            next_pose.position.x = lineStartPos[0];
            next_pose.position.y = lineStartPos[1];
            next_pose.position.z = lineStartPos[2];
            move_group.setPoseTarget(next_pose);
            move_group.move();
            cout << "按回车键运动至终点..." << endl;
            getchar();
            waypoints.clear();
            for (int i = 0; i < nPoints; i++)
            {
                next_pose.position.x += deltaPos[0];
                next_pose.position.y += deltaPos[1];
                next_pose.position.z += deltaPos[2];
                waypoints.push_back(next_pose);
            }
            fraction = move_group.computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
            ROS_INFO("Visualizing plan 4 (cartesian path) (%.2f%% acheived)", fraction * 100.0);
            plan.trajectory_ = trajectory;
            pub_robot_state.header = plan.trajectory_.joint_trajectory.header;
            for (int i = 0; i < plan.trajectory_.joint_trajectory.points.size(); i++)
            {
                pub_robot_state.header.stamp = init_time + plan.trajectory_.joint_trajectory.points[i].time_from_start;
                pub_robot_state.position = plan.trajectory_.joint_trajectory.points[i].positions;
                pub_robot_state.velocity = plan.trajectory_.joint_trajectory.points[i].velocities;
                pub_robot_state.effort = plan.trajectory_.joint_trajectory.points[i].accelerations;
                plan_positions_pub.publish(pub_robot_state);
            }
            move_group.execute(plan);
            break;
        case 4:
            // ------------------------------------------------------------------------------------ //
            // ------------------------------- 圆弧插值测试 -------------------------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "圆弧插值测试：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            next_pose.position.x = homePos[0];
            next_pose.position.y = homePos[1];
            next_pose.position.z = homePos[2];
            move_group.setPoseTarget(next_pose);
            move_group.move();
            cout << "按回车键运动至终点..." << endl;
            getchar();
            for (int i = 0; i < nPoints; i++)
            {
                double theta = (i + 1) * M_PI / nPoints;
                double r = (homePos[2] - arcPos2[2]) / 2;
                next_pose.position.x = homePos[0];
                next_pose.position.y = homePos[1] + r * sin(theta);
                next_pose.position.z = homePos[2] - r + r * sin(theta + M_PI / 2);
                move_group.setPoseTarget(next_pose);
                move_group.move();
            }
            break;
        case 5:
            // ------------------------------------------------------------------------------------ //
            // ---------------- 圆弧插值测试（computeCartesianPath）---------------- //
            // ------------------------------------------------------------------------------------ //
            cout << "圆弧插值测试（computeCartesianPath）：" << endl;
            cout << "按回车键运动至起点..." << endl;
            getchar();
            next_pose.position.x = homePos[0];
            next_pose.position.y = homePos[1];
            next_pose.position.z = homePos[2];
            move_group.setPoseTarget(next_pose);
            move_group.move();
            cout << "按回车键运动至终点..." << endl;
            getchar();
            waypoints.clear();
            for (int i = 0; i < nPoints; i++)
            {
                double theta = (i + 1) * M_PI / nPoints;
                double r = (homePos[2] - arcPos2[2]) / 2;
                next_pose.position.x = homePos[0];
                next_pose.position.y = homePos[1] + r * sin(theta);
                next_pose.position.z = homePos[2] - r + r * sin(theta + M_PI / 2);
                waypoints.push_back(next_pose);
            }
            fraction = move_group.computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
            ROS_INFO("Visualizing plan 4 (cartesian path) (%.2f%% acheived)", fraction * 100.0);
            plan.trajectory_ = trajectory;
            pub_robot_state.header = plan.trajectory_.joint_trajectory.header;
            for (int i = 0; i < plan.trajectory_.joint_trajectory.points.size(); i++)
            {
                pub_robot_state.header.stamp = init_time + plan.trajectory_.joint_trajectory.points[i].time_from_start;
                pub_robot_state.position = plan.trajectory_.joint_trajectory.points[i].positions;
                pub_robot_state.velocity = plan.trajectory_.joint_trajectory.points[i].velocities;
                pub_robot_state.effort = plan.trajectory_.joint_trajectory.points[i].accelerations;
                plan_positions_pub.publish(pub_robot_state);
            }
            move_group.execute(plan);
            /*
//采用时间最优算法对轨迹重规划
rt.setRobotTrajectoryMsg(*move_group.getCurrentState(), trajectory);
iptp.computeTimeStamps(rt, maxV, maxA);
rt.getRobotTrajectoryMsg(joinedPlan.trajectory_);
if (!move_group.execute(joinedPlan))
{
ROS_ERROR("Failed to execute plan");
return false;
}
*/
            break;
        case 0:
            flag_main = 0;
            break;
        }
        cout << endl;
    }
    // END_TUTORIAL
    ros::shutdown();
    return 0;
}
