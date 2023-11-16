#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64.h>
#include <math.h>
#include <gazebo_msgs/ApplyJointEffort.h>

#define ARM_DOF 6
#define M_PI 3.14159265358979323846
#define DEG2RAD(x) x *M_PI / 180
#define M_G 9.8

using namespace std;

float joint_states[ARM_DOF];
float l1 = 0.408, l2 = 0.4785, m1 = 11.9, m2 = 6.94, p1 = 0.304, p2 = 0.417; //0.415;
float pos1 = 0, vel1 = 0, acc1 = 0, pos2 = 0, vel2 = 0, acc2 = 0;
float effort1 = 0, effort2 = 0;
float t = 0;
float delta_t = 0.01;
float theta1_0 = -2.787, theta2_0 = -0.972;
float theta1_t = DEG2RAD(50), theta2_t = DEG2RAD(-50);

void myGetTrajectary()
{
    t = t + delta_t;
    if (t < 2)
    {
        acc1 = theta1_t / 4;
        acc2 = theta2_t / 4;
    }
    else
    {
        acc1 = -theta1_t / 4;
        acc2 = -theta2_t / 4;
    }
    vel1 += acc1 * delta_t;
    vel2 += acc2 * delta_t;
    pos1 += vel1 * delta_t + 0.5 * acc1 * delta_t * delta_t;
    pos2 += vel2 * delta_t + 0.5 * acc2 * delta_t * delta_t;
    //cout << "pos1: " << pos1 << " , vel1: " << vel1 << " , acc1: " << acc1 << endl;
    //cout << "pos2: " << pos2 << " , vel2: " << vel2 << " , acc2: " << acc2 << endl;
}

void myGetEffort()
{
    float theta1 = M_PI + theta1_0 + pos1;
    float theta2 = -theta2_0 + pos2;
    effort1 = (m1 * p1 * p1 + m2 * p2 * p2 + m2 * l1 * l1 + 2 * m2 * l1 * p2 * cos(theta2)) * acc1 +
              (m2 * p2 * p2 + m2 * l1 * p2 * cos(theta2)) * acc2 - 2 * m2 * l1 * p2 * sin(theta2) * vel1 * vel2 - m2 * l1 * p2 * sin(theta2) * vel2 * vel2 +
              (m1 * p1 + m2 * l1) * M_G * sin(theta1) + m2 * M_G * p2 * sin(theta1 + theta2);
    effort2 = (m2 * p2 * p2 + m2 * l1 * p2 * cos(theta2)) * acc1 + m2 * p2 * p2 * acc2 + m2 * l1 * p2 * sin(theta2) * vel1 * vel1 +
              m2 * M_G * p2 * sin(theta1 + theta2);
    effort2 = -effort2;
    cout << "effort1: " << effort1 << " , effort2: " << effort2 << endl;
}

void myGetEffort2()
{
    float theta1 = M_PI + theta1_0 + theta1_t - pos1;
    float theta2 = -theta2_0 + theta2_t + DEG2RAD(-0) - pos2;
    effort1 = (m1 * p1 * p1 + m2 * p2 * p2 + m2 * l1 * l1 + 2 * m2 * l1 * p2 * cos(theta2)) * acc1 +
              (m2 * p2 * p2 + m2 * l1 * p2 * cos(theta2)) * acc2 - 2 * m2 * l1 * p2 * sin(theta2) * vel1 * vel2 - m2 * l1 * p2 * sin(theta2) * vel2 * vel2 +
              (m1 * p1 + m2 * l1) * M_G * sin(theta1) + m2 * M_G * p2 * sin(theta1 + theta2);
    effort2 = (m2 * p2 * p2 + m2 * l1 * p2 * cos(theta2)) * acc1 + m2 * p2 * p2 * acc2 + m2 * l1 * p2 * sin(theta2) * vel1 * vel1 +
              m2 * M_G * p2 * sin(theta1 + theta2);
    effort2 = -effort2;
    cout << "effort1: " << effort1 << " , effort2: " << effort2 << endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_gazebo_dynamic");
    ros::NodeHandle nh;

    memset(joint_states, 0.0, ARM_DOF);

    std::string robot_name;
    ros::param::get("/robot_name", robot_name);

    std::string shoulder_joint_command_name = "/shoulder_joint_effort_controller/command";
    std::string upperArm_joint_command_name = "/upperArm_joint_effort_controller/command";
    std::string foreArm_joint_command_name = "/foreArm_joint_effort_controller/command";
    std::string wrist1_joint_command_name = "/wrist1_joint_effort_controller/command";
    std::string wrist2_joint_command_name = "/wrist2_joint_effort_controller/command";
    std::string wrist3_joint_command_name = "/wrist3_joint_effort_controller/command";

    std::string shoulder_command_topic = robot_name + shoulder_joint_command_name;
    std::string upperArm_command_topic = robot_name + upperArm_joint_command_name;
    std::string foreArm_command_topic = robot_name + foreArm_joint_command_name;
    std::string wrist1_command_topic = robot_name + wrist1_joint_command_name;
    std::string wrist2_command_topic = robot_name + wrist2_joint_command_name;
    std::string wrist3_command_topic = robot_name + wrist3_joint_command_name;

    // ros::Publisher pub_gazebo_shoulder_joint =
    //     nh.advertise<std_msgs::Float64>(shoulder_command_topic, 1000);
    // ros::Publisher pub_gazebo_upperArm_joint =
    //     nh.advertise<std_msgs::Float64>(upperArm_command_topic, 1000);
    // ros::Publisher pub_gazebo_foreArm_joint = nh.advertise<std_msgs::Float64>(foreArm_command_topic, 1000);
    // ros::Publisher pub_gazebo_wrist1_joint = nh.advertise<std_msgs::Float64>(wrist1_command_topic, 1000);
    // ros::Publisher pub_gazebo_wrist2_joint = nh.advertise<std_msgs::Float64>(wrist2_command_topic, 1000);
    // ros::Publisher pub_gazebo_wrist3_joint = nh.advertise<std_msgs::Float64>(wrist3_command_topic, 1000);

    // std_msgs::Float64 shoulder_joint;
    // shoulder_joint.data = 0.0;

    // std_msgs::Float64 upperArm_joint;
    // upperArm_joint.data = 0.0;

    // std_msgs::Float64 foreArm_joint;
    // foreArm_joint.data = 0.0;

    // std_msgs::Float64 wrist1_joint;
    // wrist1_joint.data = 0.0;

    // std_msgs::Float64 wrist2_joint;
    // wrist2_joint.data = 0.0;

    // std_msgs::Float64 wrist3_joint;
    // wrist3_joint.data = 0.0;

    ros::ServiceClient client = nh.serviceClient<gazebo_msgs::ApplyJointEffort>("/gazebo/apply_joint_effort");

    gazebo_msgs::ApplyJointEffort joint_effort1;
    joint_effort1.request.joint_name = "shoulder_joint";
    joint_effort1.request.effort = 0.0;
    joint_effort1.request.start_time.fromSec(0);
    joint_effort1.request.duration.fromSec(10);
    client.call(joint_effort1);

    gazebo_msgs::ApplyJointEffort joint_effort2;
    joint_effort2.request.joint_name = "upperArm_joint";
    joint_effort2.request.effort = 0.0;
    joint_effort2.request.start_time.fromSec(0);
    joint_effort2.request.duration.fromSec(10);
    client.call(joint_effort2);

    gazebo_msgs::ApplyJointEffort foreArm_effort3;
    joint_effort3.request.joint_name = "foreArm_joint";
    joint_effort3.request.effort = 0.0;
    joint_effort3.request.start_time.fromSec(0);
    joint_effort3.request.duration.fromSec(10);
    client.call(joint_effort3);

    gazebo_msgs::ApplyJointEffort joint_effort4;
    joint_effort4.request.joint_name = "wrist1_joint";
    joint_effort4.request.effort = 0.0;
    joint_effort4.request.start_time.fromSec(0);
    joint_effort4.request.duration.fromSec(10);
    client.call(joint_effort4);

    gazebo_msgs::ApplyJointEffort joint_effort5;
    joint_effort5.request.joint_name = "wrist2_joint";
    joint_effort5.request.effort = 0.0;
    joint_effort5.request.start_time.fromSec(0);
    joint_effort5.request.duration.fromSec(10);
    client.call(joint_effort5);

    gazebo_msgs::ApplyJointEffort joint_effort6;
    joint_effort6.request.joint_name = "wrist3_joint";
    joint_effort6.request.effort = 0.0;
    joint_effort6.request.start_time.fromSec(0);
    joint_effort6.request.duration.fromSec(10);
    client.call(joint_effort6);

    while (t <= 4)
    {
        myGetTrajectary();
        myGetEffort();
        
        joint_effort2.request.effort = effort1;
        joint_effort3.request.effort = effort2;
        
        client.call(joint_effort1);
        client.call(joint_effort2);
        client.call(joint_effort3);
        client.call(joint_effort4);
        client.call(joint_effort5);
        client.call(joint_effort6);
        
        usleep(1000 * 1000 * delta_t);
    }
    // cout << "Press any key to go back." << endl;
    // getchar();
    
    // while (t <= 4)
    // {
    //     myGetTrajectary();
    //     myGetEffort();
    //     upperArm_joint.data = effort1;
    //     foreArm_joint.data = effort2;
    //     pub_gazebo_shoulder_joint.publish(shoulder_joint);
    //     pub_gazebo_upperArm_joint.publish(upperArm_joint);
    //     pub_gazebo_foreArm_joint.publish(foreArm_joint);
    //     pub_gazebo_wrist1_joint.publish(wrist1_joint);
    //     pub_gazebo_wrist2_joint.publish(wrist2_joint);
    //     pub_gazebo_wrist3_joint.publish(wrist3_joint);
    //     usleep(1000 * 1000 * delta_t);
    // }
    // cout << "Press any key to go back." << endl;
    // getchar();

    // t = 0, pos1 = 0, vel1 = 0, acc1 = 0, pos2 = 0, vel2 = 0, acc2 = 0;
    // while (t <= 4)
    // {
    //     myGetTrajectary();
    //     myGetEffort2();
    //     upperArm_joint.data = effort1;
    //     foreArm_joint.data = effort2;
    //     pub_gazebo_shoulder_joint.publish(shoulder_joint);
    //     pub_gazebo_upperArm_joint.publish(upperArm_joint);
    //     pub_gazebo_foreArm_joint.publish(foreArm_joint);
    //     pub_gazebo_wrist1_joint.publish(wrist1_joint);
    //     pub_gazebo_wrist2_joint.publish(wrist2_joint);
    //     pub_gazebo_wrist3_joint.publish(wrist3_joint);
    //     usleep(1000 * 1000 * delta_t);
    // }
    cout << "Press any key to exit." << endl;
    getchar();

    return 0;
}