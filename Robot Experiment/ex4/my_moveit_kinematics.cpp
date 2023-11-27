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
    ros::init(argc, argv, "demo");
    ros::NodeHandle node_handle;

    // Start a thread
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // Define the planning group name
    static const std::string PLANNING_GROUP = "my_group";

    // Create a planning group interface object and set up a planning group
    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);
    move_group.setPoseReferenceFrame("base_link");

    // Create a planning scene interface object
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // Create a robot model information object
    const robot_state::JointModelGroup *joint_model_group =
        move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);
    const std::vector<std::string> &joint_names = joint_model_group->getVariableNames();

    // 创建 RobotState对象
    moveit::core::RobotStatePtr robot_state = move_group.getCurrentState();

    // 正解
    ROS_INFO("Forward kinematics: ");
    std::vector<double> input_joint_values = {0.3, 0.3, 0.3, 0.3, 0.3, 0.3};
    robot_state->setJointGroupPositions(joint_model_group, input_joint_values);
    const Eigen::Isometry3d &end_effector_state = robot_state->getGlobalLinkTransform("wrist3_Link");
    /* Print end-effector pose. Remember that this is in the model frame */
    ROS_INFO_STREAM("Translation: \n"
                    << end_effector_state.translation() << "\n");
    ROS_INFO_STREAM("Rotation: \n"
                    << end_effector_state.rotation() << "\n");
    getchar();
    
    // 逆解
    ROS_INFO("Backward kinematics: ");
    double timeout = 0.1;
    bool found_ik = robot_state->setFromIK(joint_model_group, end_effector_state, timeout);
    if (found_ik)
    {
        std::vector<double> output_joint_values;
        robot_state->copyJointGroupPositions(joint_model_group, output_joint_values);
        for (std::size_t i = 0; i < joint_names.size(); ++i)
            ROS_INFO("Joint %s: %f", joint_names[i].c_str(), output_joint_values[i]);
    }
    else
    {
        ROS_INFO("Did not find IK solution");
    }
    getchar();
    
    // 运动至目标点（设置关节） 运动至目标点（设置关节） 运动至目标点（设置关节） 运动至目标点（设置关节） 运动至目标点（设置关节）
    ROS_INFO("Move to target (joint): ");
    move_group.setJointValueTarget(input_joint_values);
    move_group.move();
    getchar();
    
    // 回零
    ROS_INFO("Homing: ");
    std::vector<double> home_position = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    move_group.setJointValueTarget(home_position);
    move_group.move();
    getchar();
    ROS_INFO("Quite.");
    
    // END_TUTORIAL
    ros::shutdown();
    return 0;
}