#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <tf/LinearMath/Quaternion.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_aubo_demo");
    ros::NodeHandle node_handle;
    //Start a thread
    ros::AsyncSpinner spinner(1);
    spinner.start();

    //Define the planning group name
    static const std::string PLANNING GROUP = "manipulator i5";

    //Create a planning group interface object and set up a planning group
    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);
    move_group.setPoseReferenceFrame("base_link");

    // Create a planning scene interface object
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    //Create a robot model information object
    const robot state::JointModelGroup *joint_model_group = move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

    //Create an object of the visualization class
    namespace rvt = rviz_visual_tools;
    moveit_visual_tools::MoveItVisualTools visual_tools("base link");
    visual_tools.deleteAllMarkers();

    //Load remote control tool
    visual_tools.loadRemoteControl();
    // Create text
    Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 1.2;
    visual_tools.publishText(text_pose, "AUBO Demo", rvt::RED, rvt::XLARGE);

    //Text visualization takes effect
    visual_tools.trigger();

    // Get the coordinate system of the basic information
    ROS_INFO_NAMED("tutorial", "Planning frame:%s", move_group.getPlanningFrame().c_str());

    //Get the end of the basic information
    ROS_INFO_NAMED("tutorial", "End effector link:%s", move_group.getEndEffectorLink().c_str());

    //Visual terminal prompt（blocking）
    visual_tools.prompt("1、按'next'键回到零位姿态");

    //********************************************************************************************
    std::vector<double> home_position;
    home_position.push_back(0);
    home_position.push_back(0);
    home_position.push_back(0);
    home_position.push_back(0);
    home_position.push_back(0);
    home_position.push_back(0);
    move_group.setJointValueTarget(home_position);
    move_group.move();

    visual_tools.prompt("2、按'next'键沿轴2转20度");

    //********************************************************************************************
    moveit::core::RobotStatePtr current_state = move_group.getCurrentState();

    //Get the joint value and model information of the current group
    std::vector<double> joint_group_positions;
    current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);

    //Modify the value of joint1
    joint_group_positions[1] = 0.348888; //radians
    move_group.setJointValueTarget(joint_group_positions);
    move_group.move();

    visual_tools.prompt("3、按'next'键回到零位姿态");

    //********************************************************************************************
    move_group.setJointValueTarget(home_position);
    move_group.move();

    visual_tools.prompt("4、按'next'键退出");

    //********************************************************************************************
    //END TUTORIAL
    ros::shutdown();
    return 0;
}