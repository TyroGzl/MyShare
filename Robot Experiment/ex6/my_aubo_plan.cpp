#include <iostream>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/Point32.h>
using namespace std;
sensor_msgs::ImagePtr msg_color(new sensor_msgs::Image);
sensor_msgs::ImagePtr msg_depth(new sensor_msgs::Image);
sensor_msgs::PointCloud2 msg_point;
float target[3] = {0};
geometry_msgs::Point32 target_point;
void color_Callback(const sensor_msgs::ImageConstPtr &color_msg)
{
    cout << "Get color image" << endl;
    msg_color->header = color_msg->header;
    msg_color->height = color_msg->height;
    msg_color->width = color_msg->width;
    msg_color->encoding = color_msg->encoding;
    msg_color->is_bigendian = color_msg->is_bigendian;
    msg_color->step = color_msg->step;
    msg_color->data = color_msg->data;
    for (int row = 0; row < msg_color->height; row++)
    {
        int idx1 = row * msg_color->step;
        for (int col = 0; col < msg_color->width; col++)
        {
            int idx2 = col * 3;
            int r = (int)(msg_color->data[idx1 + idx2]);
            int g = (int)(msg_color->data[idx1 + idx2 + 1]);
            int b = (int)(msg_color->data[idx1 + idx2 + 2]);
            //float ratio1 = (float)g / (float)r;
            //float ratio2 = (float)g / (float)b;
            float ratio1 = (float)b / (float)g;
            float ratio2 = (float)b / (float)r;
            //if (ratio1 > 1 && ratio2 > 4 && g > 50)
            if (ratio1 > 1 && ratio2 > 4 && b > 50)
            {
                msg_color->data[idx1 + idx2] = 255;
                msg_color->data[idx1 + idx2 + 1] = 255;
                msg_color->data[idx1 + idx2 + 2] = 0;
            }
        }
    }
}
void depth_Callback(const sensor_msgs::ImageConstPtr &depth_msg)
{
    cout << "Get depth image" << endl;
    msg_depth->header = depth_msg->header;
    msg_depth->height = depth_msg->height;
    msg_depth->width = depth_msg->width;
    msg_depth->encoding = depth_msg->encoding;
    msg_depth->is_bigendian = depth_msg->is_bigendian;
    msg_depth->step = depth_msg->step;
    msg_depth->data = depth_msg->data;
}
void point_Callback(const sensor_msgs::PointCloud2 &point_msg)
{
    cout << "Get point cloud" << endl;
    int n_points_in = point_msg.width;
    int n_points_out = 0;
    int r, g, b;
    float ratio1, ratio2;
    vector<float> point_data;
    vector<uint8_t> color_data;
    sensor_msgs::PointCloud2ConstIterator<float> iter_x_in(point_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_rgb_in(point_msg, "rgb");
    float sum[3] = {0};
    for (size_t i = 0; i < n_points_in; ++i, ++iter_x_in, ++iter_rgb_in)
    {
        b = (int)iter_rgb_in[0];
        g = (int)iter_rgb_in[1];
        r = (int)iter_rgb_in[2];
        //ratio1 = (float)g / (float)r;
        //ratio2 = (float)g / (float)b;
        ratio1 = (float)b / (float)g;
        ratio2 = (float)b / (float)r;
        point_data.push_back(iter_x_in[0]);
        point_data.push_back(iter_x_in[1]);
        point_data.push_back(iter_x_in[2]);
        //if (ratio1 > 1 && ratio2 > 4 && g > 50)
        if (ratio1 > 1 && ratio2 > 4 && b > 50)
        {
            color_data.push_back(0);
            color_data.push_back(255);
            color_data.push_back(255);
            sum[0] += iter_x_in[0];
            sum[1] += iter_x_in[1];
            sum[2] += iter_x_in[2];
            n_points_out++;
        }
        else
        {
            color_data.push_back(iter_rgb_in[0]);
            color_data.push_back(iter_rgb_in[1]);
            color_data.push_back(iter_rgb_in[2]);
        }
    }
    sensor_msgs::PointCloud2Modifier modifier(msg_point);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(n_points_in);
    sensor_msgs::PointCloud2Iterator<float> iter_x_out(msg_point, "x");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_rgb_out(msg_point, "rgb");
    for (size_t i = 0; i < n_points_in; ++i, ++iter_x_out, ++iter_rgb_out)
    {
        size_t idx = i * 3;
        for (size_t j = 0; j < 3; ++j)
        {
            iter_x_out[j] = point_data[idx + j];
            iter_rgb_out[j] = color_data[idx + j];
        }
    }
    msg_point.header = point_msg.header;
    msg_point.height = 1;
    msg_point.width = n_points_in;
    msg_point.is_bigendian = point_msg.is_bigendian;
    for (size_t i = 0; i < 3; ++i)
    {
        target[i] = sum[i] / n_points_out;
    }
    target_point.x = target[0];
    target_point.y = target[1];
    target_point.z = target[2];
    //cout << "target:" << target[0] << ", " << target[1] << ", " << target[2] << endl;
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_camera");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub_color = it.subscribe("/camera/color/image_raw", 1, color_Callback);
    //image_transport::Subscriber sub_depth = it.subscribe("/camera/depth/image_rect_raw", 1, depth_Callback);
    ros::Subscriber sub_point = nh.subscribe("/camera/depth/color/points", 1, point_Callback);
    
    image_transport::Publisher pub_color = it.advertise("my_pub/color", 1);
    //image_transport::Publisher pub_depth = it.advertise("my_pub/depth", 1);
    ros::Publisher pub_point = nh.advertise<sensor_msgs::PointCloud2>("my_pub/point", 1);
    ros::Publisher pub_point32 = nh.advertise<geometry_msgs::Point32>("my_pub/point32", 1);
   
    double sample_rate = 30.0;      // HZ
    ros::Rate naptime(sample_rate); // use to regulate loop rate

    while (ros::ok())
    {
        pub_color.publish(msg_color);
        //pub_depth.publish(msg_depth);
        pub_point.publish(msg_point);
        if (target_point.x < -0.5 || target_point.x > 0.5 ||
            target_point.y < -0.5 || target_point.y > 0.5 ||
            target_point.z < 0.3 || target_point.z > 0.8)
            cout << "Wrong point." << endl;
        else
            pub_point32.publish(target_point);
            
        ros::spinOnce(); //allow data update from callback;
        naptime.sleep(); // wait for remainder of specified period;
    }
    ros::shutdown();
    return 0;
}