my_exp8.c
#include <webots/robot.h>
#include <webots/motor.h>
#include <stdio.h>
#include <math.h>
#define TIME_STEP 16
#define NUM_MOTORS 12
#define L1 0.06
#define L2 0.3
#define L3 0.4
    double realtime_pose[NUM_MOTORS];
double degree2rad(double angle);
void forwardKinematics(double angle_1, double angle_2, double angle_3, double *x, double *y, double *z);
void inverseKinematics(double x, double y, double z, double *angle_1, double *angle_2, double *angle_3);
int main(int argc, char **argv)
{
    /* necessary to initialize webots stuff */
    wb_robot_init();
    const char *MOTOR_NAMES[NUM_MOTORS] = {
        "motor_LF_1", "motor_LF_2", "motor_LF_3",
        "motor_RB_1", "motor_RB_2", "motor_RB_3",
        "motor_RF_1", "motor_RF_2", "motor_RF_3",
        "motor_LB_1", "motor_LB_2", "motor_LB_3"};
    WbDeviceTag motors[NUM_MOTORS];
    int i, j;
    const double StandPose[NUM_MOTORS] = {
        degree2rad(0.0), degree2rad(45.0), degree2rad(-135.0),  // LF
        degree2rad(0.0), degree2rad(45.0), degree2rad(-135.0),  // RB
        degree2rad(0.0), degree2rad(45.0), degree2rad(-135.0),  // RF
        degree2rad(0.0), degree2rad(45.0), degree2rad(-135.0)}; // LB
    double x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4;
    double angle_1, angle_2, angle_3; // not in Webots coordinate
    /* declare WbDeviceTag variables for storing robot devices */
    for (i = 0; i < NUM_MOTORS; i++)
    {
        motors[i] = wb_robot_get_device(MOTOR_NAMES[i]);
        if (!motors[i])
            printf("could not find motor: %s\n", MOTOR_NAMES[i]);
    }
    //
    // initialize to stand pose
    //
    angle_1 = StandPose[0], angle_2 = StandPose[1], angle_3 = StandPose[2];
    forwardKinematics(angle_1, angle_2, angle_3, &x2, &y2, &z2);
    forwardKinematics(angle_1, angle_2, angle_3, &x3, &y3, &z3);
    angle_1 = StandPose[3], angle_2 = StandPose[4], angle_3 = StandPose[5];
    forwardKinematics(angle_1, angle_2, angle_3, &x1, &y1, &z1);
    forwardKinematics(angle_1, angle_2, angle_3, &x4, &y4, &z4);
    // printf("%f, %f, %f\n", x1, y1, z1); printf("%f, %f, %f\n", x2, y2, z2);
    // printf("%f, %f, %f\n", x3, y3, z3); printf("%f, %f, %f\n", x4, y4, z4);
    // LF
    inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
    // RB
    inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
    // RF
    inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
    // LB
    inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
    for (i = 0; i < NUM_MOTORS; i++)
    {
        wb_motor_set_position(motors[i], realtime_pose[i]);
        // printf("%f\n", realtime_pose[i] * 180 / M_PI);
    }
    wb_robot_step(TIME_STEP);
    //
    // get ready to walk
    //
    // LB up 5*0.02
    for (j = 0; j < 5; j++)
    {
        y4 += 0.02;
        inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // LB forward 5*0.03
    for (j = 0; j < 5; j++)
    {
        z4 += 0.03;
        inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // LB down 5*0.02
    for (j = 0; j < 5; j++)
    {
        y4 -= 0.02;
        inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // RF up
    for (j = 0; j < 5; j++)
    {
        y3 += 0.02;
        inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // RF forward
    for (j = 0; j < 5; j++)
    {
        z3 += 0.01;
        inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // RF down
    for (j = 0; j < 5; j++)
    {
        y3 -= 0.02;
        inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // LF up
    for (j = 0; j < 5; j++)
    {
        y1 += 0.02;
        inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // LF back
    for (j = 0; j < 5; j++)
    {
        z1 -= 0.03;
        inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // LF down
    for (j = 0; j < 5; j++)
    {
        y1 -= 0.02;
        inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // RB up
    for (j = 0; j < 5; j++)
    {
        y2 += 0.02;
        inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // RB back
    for (j = 0; j < 5; j++)
    {
        z2 -= 0.01;
        inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    // RB down
    for (j = 0; j < 5; j++)
    {
        y2 -= 0.02;
        inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
        for (i = 0; i < NUM_MOTORS; i++)
            wb_motor_set_position(motors[i], realtime_pose[i]);
        wb_robot_step(TIME_STEP);
    }
    //
    // walk parameters
    //
    double delta_f, delta_b, delta_u, delta_d;
    int num = 10;            // interpolate 10 points
    delta_f = 0.3 / num / 2; // swing 0.3m
    delta_b = 0.1 / num / 2; // hold 0.1m
    delta_u = 0.1 / num / 2; // up 0.1m
    delta_d = 0.1 / num / 2; // down 0.1m
    /* main loop
     * Perform simulation steps of TIME_STEP milliseconds
     * and leave the loop when the simulation is over
     */
    while (wb_robot_step(TIME_STEP) != -1)
    {
        // LF swing
        for (j = 0; j < num; j++)
        {
            y1 += delta_u;
            z1 += delta_f;
            z2 -= delta_b;
            z3 -= delta_b;
            z4 -= delta_b;
            inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
            inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
            inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
            inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
            for (i = 0; i < NUM_MOTORS; i++)
                wb_motor_set_position(motors[i], realtime_pose[i]);
            wb_robot_step(TIME_STEP);
        }
        for (j = 0; j < num; j++)
        {
            y1 -= delta_d;
            z1 += delta_f;
            z2 -= delta_b;
            z3 -= delta_b;
            z4 -= delta_b;
            inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
            inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
            inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
            inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
            for (i = 0; i < NUM_MOTORS; i++)
                wb_motor_set_position(motors[i], realtime_pose[i]);
            wb_robot_step(TIME_STEP);
        }
        // RB swing
        for (j = 0; j < num; j++)
        {
            y2 += delta_u;
            z2 += delta_f;
            z1 -= delta_b;
            z3 -= delta_b;
            z4 -= delta_b;
            inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
            inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
            inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
            inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
            for (i = 0; i < NUM_MOTORS; i++)
                wb_motor_set_position(motors[i], realtime_pose[i]);
            wb_robot_step(TIME_STEP);
        }
        for (j = 0; j < num; j++)
        {
            y2 -= delta_d;
            z2 += delta_f;
            z1 -= delta_b;
            z3 -= delta_b;
            z4 -= delta_b;
            inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
            inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
            inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
            inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
            for (i = 0; i < NUM_MOTORS; i++)
                wb_motor_set_position(motors[i], realtime_pose[i]);
            wb_robot_step(TIME_STEP);
        }
        // RF swing
        for (j = 0; j < num; j++)
        {
            y3 += delta_u;
            z3 += delta_f;
            z1 -= delta_b;
            z2 -= delta_b;
            z4 -= delta_b;
            inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
            inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
            inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
            inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
            for (i = 0; i < NUM_MOTORS; i++)
                wb_motor_set_position(motors[i], realtime_pose[i]);
            wb_robot_step(TIME_STEP);
        }
        for (j = 0; j < num; j++)
        {
            y3 -= delta_d;
            z3 += delta_f;
            z1 -= delta_b;
            z2 -= delta_b;
            z4 -= delta_b;
            inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
            inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
            inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
            inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
            for (i = 0; i < NUM_MOTORS; i++)
                wb_motor_set_position(motors[i], realtime_pose[i]);
            wb_robot_step(TIME_STEP);
        }
        // LB swing
        for (j = 0; j < num; j++)
        {
            y4 += delta_u;
            z4 += delta_f;
            z1 -= delta_b;
            z2 -= delta_b;
            z3 -= delta_b;
            inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
            inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
            inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
            inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
            for (i = 0; i < NUM_MOTORS; i++)
                wb_motor_set_position(motors[i], realtime_pose[i]);
            wb_robot_step(TIME_STEP);
        }
        for (j = 0; j < num; j++)
        {
            y4 -= delta_d;
            z4 += delta_f;
            z1 -= delta_b;
            z2 -= delta_b;
            z3 -= delta_b;
            inverseKinematics(x1, y1, z1, &realtime_pose[0], &realtime_pose[1], &realtime_pose[2]);
            inverseKinematics(x3, y3, z3, &realtime_pose[6], &realtime_pose[7], &realtime_pose[8]);
            inverseKinematics(x4, y4, z4, &realtime_pose[9], &realtime_pose[10], &realtime_pose[11]);
            inverseKinematics(x2, y2, z2, &realtime_pose[3], &realtime_pose[4], &realtime_pose[5]);
            for (i = 0; i < NUM_MOTORS; i++)
                wb_motor_set_position(motors[i], realtime_pose[i]);
            wb_robot_step(TIME_STEP);
        }
    };
    /* This is necessary to cleanup webots resources */
    wb_robot_cleanup();
    return 0;
}
double degree2rad(double angle)
{
    return angle * M_PI / 180.0;
}
void forwardKinematics(double angle_1, double angle_2, double angle_3, double *x, double *y, double *z)
{
    // left leg coordinate | right leg coordinate
    // x: left | x: right
    // y: up | y: up
    // z: forward | z: forward
    angle_1 = angle_1;
    angle_2 = angle_2;
    angle_3 = angle_3 + M_PI;
    *x = (L1 + L2 * cos(angle_2) - L3 * cos(angle_2) * cos(angle_3) + L3 * sin(angle_2) * sin(angle_3)) *
         cos(angle_1);
    *y = L2 * sin(angle_2) - L3 * sin(angle_2) * cos(angle_3) - L3 * cos(angle_2) * sin(angle_3);
    *z = (L1 + L2 * cos(angle_2) - L3 * cos(angle_2) * cos(angle_3) + L3 * sin(angle_2) * sin(angle_3)) *
         sin(angle_1);
}
void inverseKinematics(double x, double y, double z, double *angle_1, double *angle_2, double *angle_3)
{
    // left leg coordinate | right leg coordinate
    // x: left | x: right
    // y: up | y: up
    // z: forward | z: forward
    double x2, d, temp_angle_1, temp_angle_2, temp_angle_3;
    x2 = sqrt(x * x + z * z) - L1;
    d = sqrt(x2 * x2 + y * y);
    temp_angle_1 = acos((L2 * L2 + d * d - L3 * L3) / (2 * L2 * d));
    temp_angle_2 = acos((L2 * L2 + L3 * L3 - d * d) / (2 * L2 * L3));
    temp_angle_3 = atan(y / x2);
    *angle_1 = atan(z / x);
    *angle_2 = temp_angle_1 + temp_angle_3;
    *angle_3 = -(M_PI - temp_angle_2);
}