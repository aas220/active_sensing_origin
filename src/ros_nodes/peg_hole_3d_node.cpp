//
// Created by tipakorng on 8/9/17.
//

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_publisher");
    ros::NodeHandle nh;
    ros::Rate r(1);

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    uint32_t shape = visualization_msgs::Marker::CUBE;

    tf::TransformBroadcaster tfbr;

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(0, 0, 0));
    tf::Quaternion q;
    q.setRPY(0, 0, 0);
    transform.setRotation(q);

    while (ros::ok())
    {
        tfbr.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "my_frame"));

        visualization_msgs::Marker marker;
        marker.header.frame_id = "my_frame";
        marker.header.stamp = ros::Time::now();
        marker.ns = "basic_shapes";
        marker.id = 0;
        marker.type = shape;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = 0;
        marker.pose.position.y = 0;
        marker.pose.position.z = 0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 0.0;

        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;

        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0;

        marker.lifetime = ros::Duration();

        // Publish marker.
        while (marker_pub.getNumSubscribers() < 1)
        {
            if (!ros::ok())
            {
                return 0;
            }

            ROS_WARN_ONCE("Please create a subscriber to the maker node.");
        }

        marker_pub.publish(marker);

        // Cycle between different shapes
        switch (shape)
        {
            case visualization_msgs::Marker::CUBE:
                shape = visualization_msgs::Marker::SPHERE;
                break;
            case visualization_msgs::Marker::SPHERE:
                shape = visualization_msgs::Marker::ARROW;
                break;
            case visualization_msgs::Marker::ARROW:
                shape = visualization_msgs::Marker::CYLINDER;
                break;
            case visualization_msgs::Marker::CYLINDER:
                shape = visualization_msgs::Marker::CUBE;
                break;
        }

        r.sleep();
    }
}
