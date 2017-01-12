// Common
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// ROS headers
#include <ros/ros.h>
#include <ros/console.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

// Messages
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf2_msgs/TFMessage.h>

// Pangolin
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/utils/timer.h>

// Eigen
#include <Eigen/Dense>

// CUDA
#include <cuda_runtime.h>
#include <vector_types.h>

// DART
#include "geometry/plane_fitting.h"
#include "img_proc/img_ops.h"
#include "optimization/priors.h"
#include "tracker.h"
#include "util/dart_io.h"
#include "util/gl_dart.h"
#include "util/image_io.h"
#include "util/ostream_operators.h"
#include "util/string_format.h"
#include "visualization/color_ramps.h"
#include "visualization/data_association_viz.h"
#include "visualization/gradient_viz.h"
#include "optimization/kernels/intersection.h"

// New rendering stuff
#include "render/renderer.h"

// Read CSV stuff
#include "util/csv_util.h"

// Utility functions
#include <helperfuncs.h>
#include <threadPool.h>

// OpenCV stuff
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>

// Boost
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>

// PCL
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

// Learn physics models
#include "learn_physics_models/LinkStates.h"

// Gazebo messages
#include <gazebo_msgs/ContactState.h>
#include <gazebo_msgs/ContactsState.h>

#define EIGEN_DONT_ALIGN
#define M_TO_MM 1e3
#define M_TO_TENTH_MM 1e4 // Represent data as one tenth of a mm
#define M_TO_HUNDREDTH_MM 1e5 // Represent data as one hundredth of a mm

using namespace std;
namespace po = boost::program_options;

const static int panelWidth = 180;

// -----------------------------------------
///
/// \brief colorize_depth - Colorize the given depth image (scaled to 0-255)
/// \param depth_image  - Input depth image to colorize
/// \return - Colorized output
///
cv::Mat colorize_depth(const cv::Mat &depth_image)
{
    // Find min and max of depth image values
    double d_min, d_max;
    cv::minMaxIdx(depth_image, &d_min, &d_max);

    // Scale depth map to [0,255] and colorize
    cv::Mat colorized_depth_image;
    depth_image.convertTo(colorized_depth_image,CV_8UC1, 255/(d_max-d_min), -d_min);
    cv::applyColorMap(colorized_depth_image, colorized_depth_image, cv::COLORMAP_JET);

    // Return colorized image
    return colorized_depth_image;
}

// -----------------------------------------
///
/// \brief convert_32FC3_to_16UC3 - Convert a given 3-channel float image and cast it to a 3-channel 16-bit unsigned char image
/// output = (ushort) (input * scale_factor)
/// \param img          - Input image (32FC3 format)
/// \param scale_factor - Scale factor to scale the image
/// \return             - Output scaled image (16UC3 format)
///
cv::Mat convert_32FC3_to_16UC3(const cv::Mat &img, const double scale_factor)
{
    // Scale each channel of the image by the scale factor and cast it into an unsigned short (16-bit char)
    cv::Mat conv_img(img.rows, img.cols, CV_16UC3, cv::Scalar(0));
    for(int r = 0; r < img.rows; r++)
    {
        for(int c = 0; c < img.cols; c++)
        {
            // Scale the float as well
            cv::Vec3f v = img.at<cv::Vec3f>(r,c);

            // Scale the data and save as ushort
            if (v[0] != 0 || v[1] != 0 || v[2] != 0)
            {
                conv_img.at<cv::Vec3w>(r,c) = cv::Vec3w((ushort)(v[0] * scale_factor),
                                                        (ushort)(v[1] * scale_factor),
                                                        (ushort)(v[2] * scale_factor));
            }
        }
    }

    return conv_img;
}

// -----------------------------------------
///
/// \brief compute_flow_between_frames - Compute flow between the point clouds at two timesteps
/// \param vmapwmeshid  - Vertex map with mesh id (each point has 4 vals - x,y,z,id)
/// \param t_fms_1      - Frame to Model transforms @ t1
/// \param t_fms_2      - Frame to Model transforms @ t2
/// \param modelView    - Model to View transform
/// \param width        - Width of vertex map (Default: 640)
/// \param height       - Height of vertex map (Default: 480)
/// \return 32FC3 CV Mat which computes flow between all pixels at time t2 & t1 (t2-t1)
///
cv::Mat compute_flow_between_frames(const float *vmapwmeshid,
                                    const std::vector<dart::SE3> &t_fms_1,
                                    const std::vector<dart::SE3> &t_fms_2,
                                    const Eigen::Matrix4f &modelView,
                                    const int width = 640, const int height = 480)
{
    // Compute the depth map based on the link transform and see if it matches what the depth map has
    cv::Mat flow_mat(height, width, CV_32FC3, cv::Scalar(0));
    int ct = 0;
    for(int i = 0; i < flow_mat.rows; i++)
    {
        for(int j = 0; j < flow_mat.cols; j++, ct+=4) // Increment count by 4 each time (x,y,z,id)
        {
            // Get the mesh vertex in local co-ordinate frame
            // Point(0,1,2) is the local co-ordinate of the mesh vertex corresponding to the pixel
            // Point(3) is the ID of the mesh from which the vertex originated
            const float *point = &(vmapwmeshid[ct]);
            if (point[3] == 0) // Background - zero flow
                continue;

            // Get vertex point (for that pixel) and the mesh to which it belongs
            float3 localVertex = make_float3(point[0], point[1], point[2]);
            int meshid = (int) std::round(point[3]) - 1; // Reduce by 1 to get the ID

            // Transform the vertex from the local co-ordinate frame to the model @ t1 & t2
            float3 modelVertex1 = t_fms_1[meshid] * localVertex;
            float3 modelVertex2 = t_fms_2[meshid] * localVertex;

            // Get the flow vector in the robot model's frame of reference
            // This is just the difference between the vertex positions = (mV_t2 - mV_t1)
            // This is transformed to the camera frame of reference
            // Note: The flow is a vector, so the 4th homogeneous co-ordinate is zero
            Eigen::Vector4f modelFlow(modelVertex2.x - modelVertex1.x,
                                      modelVertex2.y - modelVertex1.y,
                                      modelVertex2.z - modelVertex1.z,
                                      0.0); // 4th co-ordinate = 0 so that we don't add any translation when transforming
            Eigen::Vector4f cameraFlow = modelView * modelFlow;

            // Update the flow value
            flow_mat.at<cv::Vec3f>(i,j) = cv::Vec3f(cameraFlow(0), cameraFlow(1), cameraFlow(2));
        }
    }

    return flow_mat;
}

// -----------------------------------------
///
/// \brief writeArray - Write an array to an ostream
/// \param out  - Ostream
/// \param arr  - Array of elements
/// \param n    - Number of elements to write to ostream
///
template <typename T>
void writeArray(std::ostream &out, const T *arr, const std::size_t &n)
{
    for(std::size_t i = 0; i < n; i++)
        out << arr[i] << " ";
    out << endl;
}

// -----------------------------------------
///
/// \brief writeVector - Write a vector to an ostream
/// \param out  - Ostream
/// \param vec  - Vector of elements
///
template <typename T>
void writeVector(std::ostream &out, const std::vector<T> &vec)
{
    writeArray(out, vec.data(), vec.size());
}

// ------------------------------------------
///
/// \brief renderPangolinFrame - Render a pangolin frame
/// \param tracker  - Dart tracker instance
/// \param camState - OpenGL render state
///
void renderPangolinFrame(dart::Tracker &tracker, const pangolin::OpenGlRenderState &camState,
                         const pangolin::View &camDisp)
{
    glShadeModel (GL_SMOOTH);
    float4 lightPosition = make_float4(normalize(make_float3(-0.4405,-0.5357,-0.619)),0);
    GLfloat light_ambient[] = { 0.3, 0.3, 0.3, 1.0 };
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_POSITION, (float*)&lightPosition);

    camDisp.ActivateScissorAndClear(camState);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHTING);

    camDisp.ActivateAndScissor(camState);

    glPushMatrix();

    // draw axis
    glLineWidth(10);
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(1.0, 0, 0);
    glEnd();
    glColor3f(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0, 1, 0);
    glEnd();
    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0, 1);
    glEnd();

    glColor4ub(0xff,0xff,0xff,0xff);

    glEnable(GL_COLOR_MATERIAL);

    for (int m=0; m<tracker.getNumModels(); ++m) {
        tracker.updatePose(m);
        tracker.getModel(m).render();
        tracker.getModel(m).renderSkeleton();
    }

    /////////////////
    glPopMatrix();
    glDisable(GL_LIGHTING);
    glColor4ub(255,255,255,255);

    /// Finish frame
    pangolin::FinishFrame();
}

/// ====================  PCL STUFF ====================== ///

// Ugly global variables for visualizer
pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>); // Point cloud for display
pcl::PointCloud<pcl::Normal>::Ptr flow_cloud(new pcl::PointCloud<pcl::Normal>); // Flow cloud for display

// Flags
boost::mutex point_cloud_mutex;
bool point_cloud_update = false;
bool pcl_viewer_terminate = false; // Flag for exit

// -----------------------------------------
///
/// \brief pclViewerSpinner - Spin a PCL viewer forever (never exits)
/// NOTE: PCL VISUALIZER is a single thread thing :(
/// You can't call functions like addPointCloud on a PCL Visualizer created in a separate thread.
///
void pclViewerThread()
{
    // Create a viewer
    boost::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    pcl_viewer->setBackgroundColor(0, 0, 0); // Background is black
    pcl_viewer->addCoordinateSystem(0.05); // Add a co-ordinate frame with scale
    pcl_viewer->initCameraParameters(); // Initialize camera parameters
    pcl_viewer->setCameraPosition(0,0,0.1,0,0,1,0,-1,0); // Set camera viewpoint properly (Z is the direction that the camera is facing)

    // Spin viewer forever while updating point cloud intermittently
    while (!pcl_viewer->wasStopped())
    {
        // Spin to check callbacks
        pcl_viewer->spinOnce(100);

        // Get lock on the boolean update and check if cloud was updated
        boost::mutex::scoped_lock update_lock(point_cloud_mutex);
        if(point_cloud_update)
        {
            // Show point cloud and flow dirns
            if(!pcl_viewer->updatePointCloud(point_cloud, "cloud"))
            {
                pcl_viewer->addPointCloud(point_cloud, "cloud");
                pcl_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
                pcl_viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (point_cloud, flow_cloud, 10, 1, "flow_vectors");
                pcl_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "flow_vectors");
            }
            else // Already added flow vector/point cloud
            {
                pcl_viewer->removePointCloud("flow_vectors");
                pcl_viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (point_cloud, flow_cloud, 10, 1, "flow_vectors");
                pcl_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "flow_vectors");
            }

            // Update done
            point_cloud_update = false;
        }
        update_lock.unlock();

        // If terminate flag set, exit
        if(pcl_viewer_terminate) break;

        // Sleep the thread briefly
        boost::this_thread::sleep (boost::posix_time::microseconds(10000));
    }
}

template <typename T>
std::size_t find_closest_id(const std::vector<T> &vec, const T val)
{
    // Comparator that checks absolute distance between values
    auto i = std::min_element(vec.begin(), vec.end(), [=] (T x, T y)
    {
        return std::abs(x - val) < std::abs(y - val);
    });

    // Returns closest value
    return std::distance(vec.begin(), i);
}

// -----------------------------------------
///
/// \brief get_link_transform - Get pose of link from learn_physics_models::LinkStates message given name
/// \param msg          - ConstPtr to learn_physics_models::LinkStates message
/// \param link_name    - Name of link
/// \param transform    - [Output] Pose of the link
/// \return True if link name is found in the message, false otherwise
///
bool get_link_transform(learn_physics_models::LinkStates::ConstPtr msg,
                        const std::string &link_name, tf::Transform &tfm)
{
    for(std::size_t j = 0; j < msg->name.size(); ++j)
    {
        if(msg->name[j].find(link_name) != std::string::npos)
        {
            // Copy over position
            tfm.setOrigin(tf::Vector3(msg->pose[j].position.x,
                                      msg->pose[j].position.y,
                                      msg->pose[j].position.z));

            // Copy over orientation
            tfm.setRotation(tf::Quaternion(msg->pose[j].orientation.x,
                                           msg->pose[j].orientation.y,
                                           msg->pose[j].orientation.z,
                                           msg->pose[j].orientation.w));

            // Success
            return true;
        }
    }

    // Link name not found
    //ROS_WARN("Link name [%s] not found in message",link_name.c_str());
    return false;
}

// ==== Read OBB data ==== //
struct OBB
{
    Eigen::Vector3f center;
    Eigen::Vector3f halfextents;
    Eigen::Matrix3f rotationmatrix;
};

int get_number_of_lines(const std::string &filename)
{
    // Open the file
    std::ifstream file(filename);
    std::string line;
    int number_of_lines = 0;
    if(file.is_open())
    {
        while(!file.eof()){
            getline(file,line);
            number_of_lines++;
        }
        file.close();
    }
    return number_of_lines;
}

// Read the data from a file
std::vector<OBB> read_obb_data(const std::string &filename)
{
    // Open the file
    std::ifstream obbfile(filename);
    if (!obbfile.is_open())
    {
        cout << "Cannot open file.\n";
        return std::vector<OBB>();
    }

    // Get number of lines
    int num_lines = get_number_of_lines(filename);
    int num_obbs  = std::floor(num_lines*(1.0/13));

    // Read OBBs from data
    std::vector<OBB> obbs;
    for(int k=0; k < num_obbs; k++)
    {
        // Create OBB
        OBB obb;

        // Get center
        obbfile >> obb.center(0);
        obbfile >> obb.center(1);
        obbfile >> obb.center(2);

        // Get halfextents
        obbfile >> obb.halfextents(0);
        obbfile >> obb.halfextents(1);
        obbfile >> obb.halfextents(2);

        // Get rotation matrix
        obbfile >> obb.rotationmatrix(0,0);
        obbfile >> obb.rotationmatrix(0,1);
        obbfile >> obb.rotationmatrix(0,2);
        obbfile >> obb.rotationmatrix(1,0);
        obbfile >> obb.rotationmatrix(1,1);
        obbfile >> obb.rotationmatrix(1,2);
        obbfile >> obb.rotationmatrix(2,0);
        obbfile >> obb.rotationmatrix(2,1);
        obbfile >> obb.rotationmatrix(2,2);

        // Add to vector
        obbs.push_back(obb);
    }

    // Close file
    obbfile.close();

    // Return obbs
    return obbs;
}

void write_cube_stl()
{
    std::string cube_stl = "solid cube\n"
                            "facet normal 0 0 -1\n"
                              "outer loop\n"
                                "vertex 0.5 0.5 -0.5\n"
                                "vertex 0.5 -0.5 -0.5\n"
                                "vertex -0.5 -0.5 -0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal 0 0 -1\n"
                              "outer loop\n"
                                "vertex -0.5 -0.5 -0.5\n"
                                "vertex -0.5 0.5 -0.5\n"
                                "vertex 0.5 0.5 -0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal 0 0 1\n"
                              "outer loop\n"
                                "vertex 0.5 0.5 0.5\n"
                                "vertex -0.5 0.5 0.5\n"
                                "vertex -0.5 -0.5 0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal 0 0 1\n"
                              "outer loop\n"
                                "vertex -0.5 -0.5 0.5\n"
                                "vertex 0.5 -0.5 0.5\n"
                                "vertex 0.5 0.5 0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal 1 0 0\n"
                              "outer loop\n"
                                "vertex 0.5 0.5 -0.5\n"
                                "vertex 0.5 0.5 0.5\n"
                                "vertex 0.5 -0.5 0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal 1 0 0\n"
                              "outer loop\n"
                                "vertex 0.5 -0.5 0.5\n"
                                "vertex 0.5 -0.5 -0.5\n"
                                "vertex 0.5 0.5 -0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal 0 -1 0\n"
                              "outer loop\n"
                                "vertex 0.5 -0.5 -0.5\n"
                                "vertex 0.5 -0.5 0.5\n"
                               " vertex -0.5 -0.5 0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal 0 -1 0\n"
                              "outer loop\n"
                                "vertex -0.5 -0.5 0.5\n"
                                "vertex -0.5 -0.5 -0.5\n"
                                "vertex 0.5 -0.5 -0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal -1 0 0\n"
                              "outer loop\n"
                                "vertex -0.5 -0.5 -0.5\n"
                                "vertex -0.5 -0.5 0.5\n"
                                "vertex -0.5 0.5 0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal -1 0 0\n"
                              "outer loop\n"
                                "vertex -0.5 0.5 0.5\n"
                                "vertex -0.5 0.5 -0.5\n"
                                "vertex -0.5 -0.5 -0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal 0 1 0\n"
                              "outer loop\n"
                                "vertex 0.5 0.5 0.5\n"
                                "vertex 0.5 0.5 -0.5\n"
                                "vertex -0.5 0.5 -0.5\n"
                              "endloop\n"
                            "endfacet\n"
                            "facet normal 0 1 0\n"
                              "outer loop\n"
                                "vertex -0.5 0.5 -0.5\n"
                                "vertex -0.5 0.5 0.5\n"
                                "vertex 0.5 0.5 0.5\n"
                              "endloop\n"
                            "endfacet\n"
                          "endsolid cube";
    std::ofstream file("/tmp/cube.stl");
    file << cube_stl;
    file.close();
}

dart::SE3 SE3fromEigen(const Eigen::Vector3f &pos, const Eigen::Matrix3f &rot)
{
    Eigen::AngleAxisf aa(rot);
    return dart::SE3Fromse3(dart::se3(pos(0), pos(1), pos(2), 0, 0, 0)) *
           dart::SE3Fromse3(dart::se3(0, 0, 0,
                                      aa.angle() * aa.axis()(0),
                                      aa.angle() * aa.axis()(1),
                                      aa.angle() * aa.axis()(2))); // dart::se3 rotates the translations, so we need to do this 2-step thing
}

///
/// \brief rosmsg_compare_header_stamp - Compares two ros messages using their header timestamps.
/// If used with std::sort, this sorts in an ascending order (lower timestamps first)
/// \param l - Ros message (must have a Header)
/// \param r - Ros message (must have a Header)
/// \return - True if l occured earlier than r
///
template <typename T1, typename T2>
bool rosmsg_compare_header_stamp(const T1 &l,
                                 const T2 &r)
{
    return l.header.stamp < r.header.stamp; // This sorts in an ascending manner
}

// -----------------------------------------
///
/// \brief write_link_state_to_stream - Write the passed in state vars to an output stream
/// \param out      - Output stream
/// \param name     - Name of link
/// \param pose     - Pose of link
/// \param twist    - Twist of link
/// \param accel    - Accel of link
/// \param wrench   - Wrench of link
///
void write_link_state_to_stream(std::ostream &out, const std::string &name, const geometry_msgs::Pose &pose,
                                const geometry_msgs::Twist &twist, const geometry_msgs::Accel &accel,
                                const geometry_msgs::Wrench &wrench)
{
    // Write name
    out << name << std::endl;

    // Write pose
    out << pose.position.x << " " << pose.position.y << " " << pose.position.z << " "
        << pose.orientation.x << " "  << pose.orientation.y << " " << pose.orientation.z << " "
        << pose.orientation.w << std::endl;

    // Write twist
    out << twist.linear.x << " " << twist.linear.y << " " << twist.linear.z << " "
        << twist.angular.x << " "  << twist.angular.y << " " << twist.angular.z << std::endl;

    // Write accel
    out << accel.linear.x << " " << accel.linear.y << " " << accel.linear.z << " "
        << accel.angular.x << " "  << accel.angular.y << " " << accel.angular.z << std::endl;

    // Write wrench
    out << wrench.force.x << " " << wrench.force.y << " " << wrench.force.z << " "
        << wrench.torque.x << " "  << wrench.torque.y << " " << wrench.torque.z << std::endl;

    // Empty line for de-lineation
    out << std::endl;
}

// -----------------------------------------
////
/// \brief save_link_states_as_text - Save a LinkStates message to disk as text
/// \param msg          - learn_physics_models::LinkStates message
/// \param file_path    - Path to save file
/// \return - False if file can't be opened, True on Success
///
bool save_link_states(const learn_physics_models::LinkStates::ConstPtr msg,
                      const std::string &file_path)
{
    // Open the file
    std::ofstream out(file_path);
    if(!out.is_open())
    {
        ROS_ERROR("Cannot open file [%s] for writing", file_path.c_str());
        return false;
    }

    //  Write the data
    out << std::setprecision(8); // 8 digit floating point precision
    for(std::size_t j = 0; j < msg->name.size(); ++j)
    {
        write_link_state_to_stream(out, msg->name[j], msg->pose[j], msg->twist[j],
                              msg->accel[j], msg->wrench[j]);
    }

    // Close it now
    out.close();
    return true;
}

// -----------------------------------------
///
/// \brief tf_transform_to_geometry_msgs_pose - Convert tf::Transform to geometry_msgs::Pose
/// \param tfm   - TF Transform
/// \param pose  - [Output] Geometry msgs pose
///
geometry_msgs::Pose tf_transform_to_geometry_msgs_pose(const tf::Transform &tfm)
{
    // Init
    geometry_msgs::Pose pose;

    // Get position
    pose.position.x = tfm.getOrigin().x();
    pose.position.y = tfm.getOrigin().y();
    pose.position.z = tfm.getOrigin().z();

    // Get orientation
    pose.orientation.x = tfm.getRotation().x();
    pose.orientation.y = tfm.getRotation().y();
    pose.orientation.z = tfm.getRotation().z();
    pose.orientation.w = tfm.getRotation().w();

    return pose;
}

/// =================== POST PROCESS TO GET EXPERIMENT states ================= ////

///
/// \brief get_experiment_event_times - Post-process the experiment data to get the time for various events:
/// 1,2 are present for all dataset, 3,4,5,6 may or may not be present
/// 1) ANY_MOTION        : First motion of any item in the scene (includes baxter, tabletop objects etc)
/// 2) BAXTER_MOTION     : First motion of any baxter link in the scene
/// 3) BOX_MOTION        : First motion of any object in the scene (happens only because of baxter hitting objects)
/// 4) TARGET_BOX_MOTION : First motion of target object in the scene (happens because of baxter hitting target object and other objects hitting target object)
/// 5) MULTI_BOX_COLL    : First contact between multiple objects (Assumes that objects are initialized out of contact)
/// 6) BOX_FALL          : First time any object falls off table -> check when it reaches edge of table
/// 7) ALL_STATIC        : Point at which all items become static. Works by thresholding on position change, very approximate (Assumes motion is not fully rotational)
/// 8) BAXTER_STATIC     : Point at which all baxter links become static. Works same way as above
/// 9) BOXES_STATIC      : Point at which all objects become static. Works by thresholding on position change, very approximate (Assumes motion is not fully rotational)
/// 10) TARGET_BOX_STATIC : Point at which target object becomes static. Works same way as above
/// \param contact_states   - Vector of contact states of objects in the scene
/// \param link_states      - Link states of the objects in the scene
/// \param table_obb        - Bounding box of the table in the scene
/// \return - A map that maps between various significant events during the experiment to the time they occur at
///
std::map<std::string, double> get_experiment_event_times(const std::vector<gazebo_msgs::ContactState> &contact_states,
                                                         std::vector<learn_physics_models::LinkStates::ConstPtr> link_states,
                                                         const OBB &table_obb,
                                                         const std::string &obj_common_str,
                                                         const std::string &target_link_name)
{
    // Init the map
    std::map<std::string, double> map_events_to_times;

    // Get table min and max x,y,z values
    Eigen::Vector3f a = table_obb.center + table_obb.rotationmatrix * table_obb.halfextents;
    Eigen::Vector3f b = table_obb.center - table_obb.rotationmatrix * table_obb.halfextents;
    Eigen::Vector3f table_min(std::min(a(0), b(0)), std::min(a(1), b(1)), std::min(a(2), b(2)));
    Eigen::Vector3f table_max(std::max(a(0), b(0)), std::max(a(1), b(1)), std::max(a(2), b(2)));

    // Print out approx start and end times for the bag
    double start_time = std::min(contact_states.front().header.stamp.toSec(), link_states.front()->header.stamp.toSec());
    double end_time   = std::max(contact_states.back().header.stamp.toSec(), link_states.back()->header.stamp.toSec());
    ROS_INFO("(Approx) Bag start time: %f, end time: %f, duration: %f", start_time, end_time, end_time-start_time);

    // 1) Any motion - First motion of any item in the scene (including baxter etc)
    for(std::size_t i = 0; i < link_states.size()-1; ++i)
    {
        // Check for small changes in the pose of the object. This works very well
        // This is very approximate -> no rotation is checked here. Assumption is that there is no pure torque
        bool any_motion = false;
        for(std::size_t j = 0; j < link_states[i]->name.size(); ++j)
        {
            double dist = tf::Vector3(link_states[i]->pose[j].position.x - link_states[i+1]->pose[j].position.x,
                                      link_states[i]->pose[j].position.y - link_states[i+1]->pose[j].position.y,
                                      link_states[i]->pose[j].position.z - link_states[i+1]->pose[j].position.z).length();
            if(dist > 1e-3) // 2e-3 is somewhat small => note that this is sampled @ 50 Hz, so 1e-5 = ~1cm/s
            {
                any_motion = true; // Objects move by more than a mm
                ROS_INFO("First motion of any object [%s] at: %f", link_states[i]->name[j].c_str(),
                         link_states[i]->header.stamp.toSec());
                break;
            }
        }

        // If the objects are static, save this and exit
        if(any_motion)
        {
            map_events_to_times["ANY_MOTION"] = link_states[i]->header.stamp.toSec();
            break; // Exit
        }
    }

    // 2) Baxter motion - First motion of any baxter link in the scene
    for(std::size_t i = 0; i < link_states.size()-1; ++i)
    {
        // This can only be after first item motion
        if(link_states[i]->header.stamp.toSec() < map_events_to_times["ANY_MOTION"])
        {
            continue;
        }

        // Check for small changes in the pose of the object. This works very well
        // This is very approximate -> no rotation is checked here. Assumption is that there is no pure torque
        bool baxter_motion = false;
        for(std::size_t j = 0; j < link_states[i]->name.size(); ++j)
        {
            if(link_states[i]->name[j].find("baxter") != std::string::npos) // Check all objects
            {
                double dist = tf::Vector3(link_states[i]->pose[j].position.x - link_states[i+1]->pose[j].position.x,
                                          link_states[i]->pose[j].position.y - link_states[i+1]->pose[j].position.y,
                                          link_states[i]->pose[j].position.z - link_states[i+1]->pose[j].position.z).length();
                if(dist > 1e-3) // 2e-3 is somewhat small => note that this is sampled @ 50 Hz, so 1e-5 = ~1cm/s
                {
                    baxter_motion = true; // Objects move by more than a mm
                    ROS_INFO("First motion of baxter link [%s] at: %f", link_states[i]->name[j].c_str(),
                             link_states[i]->header.stamp.toSec());
                    break;
                }
            }
        }

        // If the objects are static, save this and exit
        if(baxter_motion)
        {
            map_events_to_times["BAXTER_MOTION"] = link_states[i]->header.stamp.toSec();
            break; // Exit
        }
    }

    // 3) First motion of any object in scene (after force has been applied to the force_target
    for(std::size_t i = 0; i < contact_states.size(); ++i)
    {
        // Check for any collision between baxter and one of the objects
        if( ((contact_states[i].collision1_name.find("baxter") != std::string::npos) &&
             (contact_states[i].collision2_name.find(obj_common_str) != std::string::npos)) ||
            ((contact_states[i].collision1_name.find(obj_common_str) != std::string::npos) &&
             (contact_states[i].collision2_name.find("baxter") != std::string::npos)) )
        {
            map_events_to_times["BOX_MOTION"] = contact_states[i].header.stamp.toSec();
            ROS_INFO("First collision between [%s] and an object [%s] at: %f",
                     contact_states[i].collision1_name.c_str(), contact_states[i].collision2_name.c_str(),
                     contact_states[i].header.stamp.toSec());
            break; // Exit loop
        }
    }

    // 4) First motion of target object in scene (this happens when the bullet hits it or it is directly moved)
    // This can ONLY be at or after the motion of any object in the scene
    for(std::size_t i = 0; i < contact_states.size(); ++i)
    {
        // This can only be after first object motion
        if(contact_states[i].header.stamp.toSec() < map_events_to_times["BOX_MOTION"])
        {
            continue;
        }

        // Check for any collision between an object and the target object
        if ( (contact_states[i].collision1_name.find(target_link_name) != std::string::npos) &&
            ((contact_states[i].collision2_name.find("baxter") != std::string::npos) ||
             (contact_states[i].collision2_name.find(obj_common_str) != std::string::npos)) )
        {
            ROS_INFO("Collision with [%s] causes target object [%s] to move at: %f", contact_states[i].collision2_name.c_str(),
                 target_link_name.c_str(), contact_states[i].header.stamp.toSec());
            map_events_to_times["TARGET_BOX_MOTION"] = contact_states[i].header.stamp.toSec();
            break;
        }
        else if ( (contact_states[i].collision2_name.find(target_link_name) != std::string::npos) &&
                  ((contact_states[i].collision1_name.find("baxter") != std::string::npos) ||
                   (contact_states[i].collision1_name.find(obj_common_str) != std::string::npos)) )
        {
            ROS_INFO("Collision with [%s] causes target object [%s] to move at: %f", contact_states[i].collision1_name.c_str(),
                 target_link_name.c_str(), contact_states[i].header.stamp.toSec());
            map_events_to_times["TARGET_BOX_MOTION"] = contact_states[i].header.stamp.toSec();
            break;
        }
    }

    // 5) First contact between multiple objects
    // Assumes that objects are initialized out of contact
    for(std::size_t i = 0; i < contact_states.size(); ++i)
    {
        // Check only states that are at or after the first object motion
        if(contact_states[i].header.stamp.toSec() < map_events_to_times["BOX_MOTION"])
        {
            // Assumption that objects are initialized out of contact
            continue;
        }

        // Check for any collision between two objects
        if((contact_states[i].collision1_name.find(obj_common_str) != std::string::npos) &&
           (contact_states[i].collision2_name.find(obj_common_str) != std::string::npos))
        {
            map_events_to_times["MULTI_BOX_COLL"] = contact_states[i].header.stamp.toSec();
            ROS_INFO("First collision between 2 objects ([%s] & [%s]) at: %f",
                     contact_states[i].collision1_name.c_str(), contact_states[i].collision2_name.c_str(),
                     contact_states[i].header.stamp.toSec());
            break; // Exit loop
        }
    }

    // 6) Find first time an object reaches an edge
    // Approximate this by looking at objects's COM. If this goes beyond the table bounding box, it is assumed to tilt/fall
    for(std::size_t i = 0; i < link_states.size(); ++i)
    {
        // Check only states that are at or after the first object motion
        if(link_states[i]->header.stamp.toSec() < map_events_to_times["BOX_MOTION"])
        {
            // Assumption that objects are initialized out of contact
            continue;
        }

        // Find the state where any of objects goes beyond the table's bounding object
        bool found = false;
        for(std::size_t j = 0; j < link_states[i]->name.size(); ++j)
        {
            if(link_states[i]->name[j].find(obj_common_str) != std::string::npos) // Check all objects
            {
                // If the center of the object leaves the confines of the table
                if(link_states[i]->pose[j].position.x <= table_min(0) ||
                   link_states[i]->pose[j].position.x >= table_max(0) ||
                   link_states[i]->pose[j].position.y <= table_min(1) ||
                   link_states[i]->pose[j].position.y >= table_max(1))
                {
                    map_events_to_times["BOX_FALL"] = link_states[i]->header.stamp.toSec();
                    ROS_INFO("Object [%s] falls off table at: %f",
                             link_states[i]->name[j].c_str(), link_states[i]->header.stamp.toSec());
                    found = true;
                    break; // Exit loop
                }
            }
        }

        if(found) // Found event
            break;
    }

    // 7) Point at which all objects become static
    for(std::size_t i = 0; i < link_states.size()-1; ++i)
    {
        // Check only states that are at or after the first object motion
        if(link_states[i]->header.stamp.toSec() < map_events_to_times["ANY_MOTION"])
        {
            // Objects first have to move
            continue;
        }

        // Check for small changes in the pose of the all links. This works very well
        // This is very approximate -> no rotation is checked here. Assumption is that there is no pure torque
        bool all_static = true;
        for(std::size_t j = 0; j < link_states[i]->name.size(); ++j)
        {
            double dist = tf::Vector3(link_states[i]->pose[j].position.x - link_states[i+1]->pose[j].position.x,
                                      link_states[i]->pose[j].position.y - link_states[i+1]->pose[j].position.y,
                                      link_states[i]->pose[j].position.z - link_states[i+1]->pose[j].position.z).length();
            if(dist > 1e-5) // 2e-3 is somewhat small => note that this is sampled @ 50 Hz, so 1e-5 = ~1cm/s
            {
                all_static = false;
                break;
            }
        }

        // If the objects are static, save this and exit
        if(all_static)
        {
            map_events_to_times["ALL_STATIC"] = link_states[i]->header.stamp.toSec();
            ROS_INFO("All items static at: %f", link_states[i]->header.stamp.toSec());
            break; // Exit
        }
    }

    // 8) Point at which all objects become static
    for(std::size_t i = 0; i < link_states.size()-1; ++i)
    {
        // Check only states that are at or after the first object motion
        if(link_states[i]->header.stamp.toSec() < map_events_to_times["BAXTER_MOTION"])
        {
            // Objects first have to move
            continue;
        }

        // Check for small changes in the pose of the object. This works very well
        // This is very approximate -> no rotation is checked here. Assumption is that there is no pure torque
        bool baxter_static = true;
        for(std::size_t j = 0; j < link_states[i]->name.size(); ++j)
        {
            if(link_states[i]->name[j].find("baxter") != std::string::npos) // Check all objects
            {
                double dist = tf::Vector3(link_states[i]->pose[j].position.x - link_states[i+1]->pose[j].position.x,
                                          link_states[i]->pose[j].position.y - link_states[i+1]->pose[j].position.y,
                                          link_states[i]->pose[j].position.z - link_states[i+1]->pose[j].position.z).length();
                if(dist > 1e-5) // 2e-3 is somewhat small => note that this is sampled @ 50 Hz, so 1e-5 = ~1cm/s
                {
                    baxter_static = false;
                    break;
                }
            }
        }

        // If the objects are static, save this and exit
        if(baxter_static)
        {
            map_events_to_times["BAXTER_STATIC"] = link_states[i]->header.stamp.toSec();
            ROS_INFO("Baxter static at: %f", link_states[i]->header.stamp.toSec());
            break; // Exit
        }
    }

    // 9) Point at which all objects become static
    for(std::size_t i = 0; i < link_states.size()-1; ++i)
    {
        // Check only states that are at or after the first object motion
        if(link_states[i]->header.stamp.toSec() < map_events_to_times["BOX_MOTION"])
        {
            // Objects first have to move
            continue;
        }

        // Check for small changes in the pose of the object. This works very well
        // This is very approximate -> no rotation is checked here. Assumption is that there is no pure torque
        bool objects_static = true;
        for(std::size_t j = 0; j < link_states[i]->name.size(); ++j)
        {
            if(link_states[i]->name[j].find(obj_common_str) != std::string::npos) // Check all objects
            {
                double dist = tf::Vector3(link_states[i]->pose[j].position.x - link_states[i+1]->pose[j].position.x,
                                          link_states[i]->pose[j].position.y - link_states[i+1]->pose[j].position.y,
                                          link_states[i]->pose[j].position.z - link_states[i+1]->pose[j].position.z).length();
                if(dist > 1e-5) // 2e-3 is somewhat small => note that this is sampled @ 50 Hz, so 1e-5 = ~1cm/s
                {
                    objects_static = false;
                    break;
                }
            }
        }

        // If the objects are static, save this and exit
        if(objects_static)
        {
            map_events_to_times["BOXES_STATIC"] = link_states[i]->header.stamp.toSec();
            ROS_INFO("All tabletop objects static at: %f", link_states[i]->header.stamp.toSec());
            break; // Exit
        }
    }

    // 10) Point at which the target object becomes static
    // Compute target static time
    for(std::size_t i = 0; i < link_states.size()-1; ++i)
    {
        // Check only states that are at or after the first object motion
        if(link_states[i]->header.stamp.toSec() < map_events_to_times["TARGET_BOX_MOTION"])
        {
            // Objects first have to move
            continue;
        }

        // Check for small changes in the pose of the object. This works very well
        // This is very approximate -> no rotation is checked here. Assumption is that there is no pure torque
        bool target_static = true;
        for(std::size_t j = 0; j < link_states[i]->name.size(); ++j)
        {
            if(link_states[i]->name[j].find(target_link_name) != std::string::npos) // Check all objects
            {
                double dist = tf::Vector3(link_states[i]->pose[j].position.x - link_states[i+1]->pose[j].position.x,
                                          link_states[i]->pose[j].position.y - link_states[i+1]->pose[j].position.y,
                                          link_states[i]->pose[j].position.z - link_states[i+1]->pose[j].position.z).length();
                if(dist > 1e-5) // 1e-5 is somewhat conservative => note that this is sampled @ 1000 Hz, so 1e-5 = ~1cm/s
                {
                    target_static = false;
                    break;
                }
            }
        }

        // If the objects are static, save this and exit
        if(target_static)
        {
            map_events_to_times["TARGET_BOX_STATIC"] = link_states[i]->header.stamp.toSec();
            ROS_INFO("Target object [%s] static at: %f", target_link_name.c_str(), link_states[i]->header.stamp.toSec());
            break; // Exit
        }
    }

    return map_events_to_times;
}

// -----------------------------------------
///
/// \brief save_object_data - Save data about all the events that happen in the experiment and their corresponding closest depth images
/// \param map_events_to_times  - Map of events to times (Look at: "get_experiment_event_times")
/// \param depth_image_times    - Timestamps for all the depth images
/// \param file_path            - Filename to write the data to
/// \return - True if success, false if not
///
bool save_event_time_data(const std::map<std::string, double> &map_events_to_times,
                          const std::vector<double> &depth_image_times,
                          const std::string &file_path)
{
    // Open file to write this data
    std::ofstream out(file_path);
    if(!out.is_open())
    {
        ROS_ERROR("Cannot open file [%s] for writing", file_path.c_str());
        return false;
    }

    //  Write the data - Get the corresponding depth images for each of the events
    out << std::setprecision(8); // 8 digit floating point precision
    for(std::map<std::string, double>::const_iterator it = map_events_to_times.begin();
        it != map_events_to_times.end(); ++it)
    {
        std::size_t k = find_closest_id(depth_image_times, it->second);
        if(k == depth_image_times.size()-1) // Can't use last image's index!!!
        {
            ROS_INFO("Corrected from using last image's depth frame");
            k--; // We do not save the final depth image (or any flow for it), so we can't use it's index for any event
        }
        out << it->first << " " << it->second << " " << k << " " << depth_image_times[k] << endl;
    }
    out.close(); // Close it now
    return true;
}

// -----------------------------------------
// Main
int main(int argc, char **argv)
{
    /// ===== Get parameters

    // Default parameters
    std::string pokeroot = "";       // Folder to load saved poke images from. Flow vectors will be saved in these folders.
    int startid = -1; int endid = -1;  // Start and end ids
    std::string modelfolder     = "";       // Model folder
    bool visualize              = false;     // Visualize the data in pangolin
    bool pcl_visualize          = false;    // Visualize the data in pcl
    bool save_fullres_data      = false;    // By default, do not save full res images
    int num_threads             = 5;        // By default, we use 5 threads
    std::string step_list_str   = "[1]";    // Steps (depth images) in future to look at computing flow (Default: 1 => t+1)

    //PARSE INPUT ARGUMENTS
    po::options_description desc("Allowed options",1024);
    desc.add_options()
        ("help", "produce help message")
        ("pokeroot",   po::value<std::string>(&pokeroot), "Top level folder containing all the poke datasets")
        ("startid",    po::value<int>(&startid), "Start folder id. Folder names are poke$id, poke($id+1) and so on")
        ("endid",      po::value<int>(&endid), "End folder id. Folder names are ... poke$id-1, poke$id")
        ("modelfolder", po::value<std::string>(&modelfolder), "Path to folder containing the models in the data")
        ("visualize",  po::value<bool>(&visualize), "Flag for visualizing trajectories. [Default: 1]")
        ("pclvisualize",  po::value<bool>(&pcl_visualize), "Flag for visualizing data using PCL. [Default: 1]")
        ("savefullres",po::value<bool>(&save_fullres_data), "Flag for saving full resolution image data. [Default: 0]")
        ("numthreads", po::value<int>(&num_threads), "Number of threads to use")
        ("steplist",   po::value<std::string>(&step_list_str), "Comma separated list of steps to compute flow. [Default: [1]]")
        ;

    // Parse input options
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (std::exception & e) {
        cout << desc << endl;
        cout << e.what() << endl;
        return false;
    }
    if (vm.count("help")) {
        cout << desc << endl;
        return false;
    }

    /// ===== Parse parameters and print details
    assert(!pokeroot.empty() && "Please pass in root folder --pokeroot <path_to_folder>");
    assert(!modelfolder.empty() && "Please pass in folder which has all the model xml files --modelfolder <path_to_folder>");
    cout << "Loading data from directory: " << pokeroot << endl;
    if(save_fullres_data) cout << "Saving full resolution images [480x640] in addition to default [240x320] images" << endl;
    cout << "Using " << num_threads << " threads to speed up processing" << endl;

    // Check start and end ids
    assert(startid >= 0 && endid >= 0 && startid <= endid);
    cout << "Start folder: poke" << startid << ", End folder: poke" << endid << endl;

    // Parse step list to get all the queried "step" lengths
    std::string step_list_substr = step_list_str.substr(1, step_list_str.size() - 2); // Remove braces []
    std::vector<std::string> words;
    boost::split(words, step_list_substr, boost::is_any_of(","), boost::token_compress_on); // Split on ","

    // Convert the strings to integers
    std::vector<int> step_list;
    std::transform(words.begin(), words.end(), std::back_inserter(step_list), string_to_int);
    cout << "Will compute flows for " << step_list.size() << " different step(s)" << endl;

    // Allow multi-threaded support for speeding up computation
    rgbd::threadGroup tg(num_threads);
    printf("Using %d threads to speed up computation \n",num_threads);

    // Write the cube stl
    write_cube_stl();

    /// ===== PCL viewer
    boost::shared_ptr<boost::thread> pcl_viewer_thread;
    if(visualize && pcl_visualize)
    {
        printf("Visualizing the point clouds and flow vectors using PCL \n");
        point_cloud_update = false; // Do not update point clouds yet
        pcl_viewer_thread.reset(new boost::thread(pclViewerThread));
    }

    /// ===== Set up a DART tracker with the baxter and object models

    // Setup OpenGL/CUDA/Pangolin stuff
    cudaGLSetGLDevice(0);
    cudaDeviceReset();
    const float totalwidth = 1920;
    const float totalheight = 1080;
    pangolin::CreateWindowAndBind("Main",totalwidth,totalheight);
    glewInit();

    // ==== Load baxter model
    dart::Tracker tracker;
    const std::string objectModelFile = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter_rosmesh_closedgripper.xml";
    tracker.addModel(objectModelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID = 0;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID).getReducedArticulatedDimensions() << endl;

    // Get the baxter pose and create a map between "frame name" and "pose dimension"
    dart::Pose &baxter_pose(tracker.getPose(baxterID));
    std::vector<std::string> model_joint_names;
    std::map<std::string, int> joint_name_to_pose_dim;
    for(int i = 0; i < baxter_pose.getReducedArticulatedDimensions(); i++)
    {
        model_joint_names.push_back(baxter_pose.getReducedName(i));
        joint_name_to_pose_dim[baxter_pose.getReducedName(i)] = i;
    }

    // ==== Create a default table XML - 1x1x1 cube at the origin
    dart::SE3 defaultPose(dart::SE3Fromse3(dart::se3(-0.5,0,0,0,0,0))); // default pose is behind camera
    std::string box_xml = "<?xml version='1.0' ?>\
                           <model version ='1'>\
                               <geom type='mesh' sx='1' sy='1' sz='1' tx='0' ty='0' tz='0' rx='0' ry='0' rz='0' red='128' green='128' blue='128' meshFile='cube.stl' />\
                           </model>";

    // Save OBB xml to file
    std::string tableModelFile = "/tmp/tabledartmodel.xml";
    {
        std::ofstream file(tableModelFile);
        file << box_xml;
        file.close();
    }

    // Load table and set it's pose based on the recorded data
    std::cout << "Loading Table model" << std::endl;
    tracker.addModel(tableModelFile, 0.01, 0.10, 64, -1, make_float3(0,0,0), 0, 1e5, false); // Add baxter model with SDF resolution = 1 cm
    int tableID = 1;
    tracker.getPose(tableID).setTransformModelToCamera(defaultPose); // Set default pose for table
    tracker.updatePose(tableID);

    // ==== Load other object models
    std::vector<std::string> dart_model_dirs = findAllDirectoriesInFolder(modelfolder);
    std::map<std::string, int> model_name_to_id;
    std::map<int, std::string> model_id_to_name;
    model_name_to_id["baxter"] = 0; model_id_to_name[0] = "baxter";
    model_name_to_id["table"] = 1; model_id_to_name[1] = "table";
    for(std::size_t i = 0; i < dart_model_dirs.size(); i++)
    {
        // Get object name
        std::vector<std::string> words;
        boost::split(words, dart_model_dirs[i], boost::is_any_of("/"), boost::token_compress_on); // Split on "/"
        std::string modelName = words.back();
        std::cout << "Loading model: " << modelName << std::endl;

        // Load DART xml
        std::string modelFile = dart_model_dirs[i] + "/dartmodel.xml"; // DART xml file
        tracker.addModel(modelFile, 0.01, 0.10, 64, -1, make_float3(0,0,0), 0, 1e5, false); // Add object model with SDF resolution = 1 cm
        int modelID = i+2;

        // Set model id based on name
        model_name_to_id[modelName] = modelID; // Map object name to dart id (already added two models before)
        model_id_to_name[modelID] = modelName; // Map dart id to object name

        // Update pose of model to default pose behind camera
        tracker.getPose(modelID).setTransformModelToCamera(defaultPose); // Set default pose for object
        tracker.updatePose(modelID);
        std::cout << std::endl;
    }

    /// ===== Pangolin initialization
    /// Pangolin mirrors the display, so we need to use TopLeft display for it. Our rendering needs BottomLeft

    /// ===== Setup camera parameters
    // Initialize camera parameters and projection matrix
    int glHeight  = 480;
    int glWidth   = 640;
    float glFLx   = 525;
    float glFLy   = 525;
    float glPPx   = 319.5;
    float glPPy   = 239.5;
    pangolin::OpenGlMatrixSpec glK_pangolin = pangolin::ProjectionMatrixRDF_TopLeft(glWidth,glHeight,glFLx,glFLy,glPPx,glPPy,0.01,1000);

    // Create the pangolin state
    pangolin::OpenGlRenderState camState(glK_pangolin);
    pangolin::View & camDisp = pangolin::Display("cam").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::CreatePanel("pose").SetBounds(0.0,1.0,pangolin::Attach::Pix(-panelWidth), pangolin::Attach::Pix(-2*panelWidth));
    pangolin::Display("multi")
            .SetBounds(1.0, 0.0, pangolin::Attach::Pix(2*panelWidth), pangolin::Attach::Pix(-2*panelWidth))
            .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(camDisp);

    // Get base TF w.r.t camera (fixed)
    tf::Transform base_to_depth_camera;
    base_to_depth_camera.setOrigin(tf::Vector3(0.005, 0.508, 0.126));
    base_to_depth_camera.setRotation(tf::Quaternion(0.630, -0.619, 0.336, 0.326));

    // Get the modelview matrix
    pangolin::OpenGlMatrix mat;
    base_to_depth_camera.getOpenGLMatrix(mat.m);
    Eigen::Matrix4f modelView(mat);

    /// ===== Create stuff for *our* rendering

    // Create a renderer
    pangolin::OpenGlMatrixSpec glK = pangolin::ProjectionMatrixRDF_BottomLeft(glWidth,glHeight,glFLx,glFLy,glPPx,glPPy,0.01,1000);
    l2s::Renderer<TYPELIST2(l2s::IntToType<l2s::RenderVertMapWMeshID>, l2s::IntToType<l2s::RenderDepth>)> renderer(640, 480, glK);
    renderer.setModelViewMatrix(modelView);

    ///////////////////////////////////////////////////////////////////////////////////////////
    /// ===== Iterate over all the poke directories to compute the results for each poke folder
    for(int jj = startid; jj <= endid; jj++)
    {
        // Get the poke folder
        std::string pokefolder = pokeroot + "/poke" + std::to_string(jj) + "/";
        std::cout << " Folder [" << jj+1 << "/" << endid+1 << "]: " << pokefolder << endl;

        /// ===== Read the table OBB data and update the table model's geometry scale
        std::vector<OBB> obbs = read_obb_data(pokefolder + "/" + "tableobbdata.txt");
        assert(obbs.size() == 1 && "Table OBB not read correctly");
        OBB tableOBB = obbs[0];
        Eigen::Vector3f tableExtents = 2 * tableOBB.halfextents;

        // Set the table's geometry scale to the extents
        for (int s = 0; s < tracker.getModel(tableID).getNumSdfs(); ++s)
        {
            // Get the frame number for the SDF and it's transform w.r.t robot base
            int f = tracker.getModel(tableID).getSdfFrameNumber(s);

            // Iterate over all the geometries for the model and get the mesh attributes for the data
            for(int g = 0; g < tracker.getModel(tableID).getFrameNumGeoms(f); ++g)
            {
                // Get the mesh index
                int gid = tracker.getModel(tableID).getFrameGeoms(f)[g];
                int mid = tracker.getModel(tableID).getMeshNumber(gid);
                if(mid == -1) continue; // Has no mesh

                // Set the geometry scale for all the meshes
                tracker.getModel(tableID).setGeometryScale(gid, make_float3(tableExtents(0), tableExtents(1), tableExtents(2))); // Set scale of the geometry
            }
        }

        /// ====== Setup mesh vertices for rendering depth & flow data
        // Pre-process to compute the mesh vertices and indices for all the robot parts
        std::vector<std::vector<float3> > meshVertices, transformedMeshVertices;
        std::vector<std::vector<float4> > meshVerticesWMeshID;
        std::vector<pangolin::GlBuffer> meshIndexBuffers;
        std::vector<std::vector<pangolin::GlBuffer *> > meshVertexAttributeBuffers;
        std::vector<int> meshFrameids, meshModelids;
        for (int m = 0; m < tracker.getNumModels(); ++m)
        {
            // Get the model
            const dart::SE3 modeltfm = tracker.getModel(m).getTransformModelToCamera();
            for (int s = 0; s < tracker.getModel(m).getNumSdfs(); ++s)
            {
                // Get the frame number for the SDF and it's transform w.r.t robot base
                int f = tracker.getModel(m).getSdfFrameNumber(s);
                const dart::SE3 tfm = modeltfm * tracker.getModel(m).getTransformFrameToModel(f);

                // Iterate over all the geometries for the model and get the mesh attributes for the data
                for(int g = 0; g < tracker.getModel(m).getFrameNumGeoms(f); ++g)
                {
                    // Get the mesh index
                    int gid = tracker.getModel(m).getFrameGeoms(f)[g];
                    int mid = tracker.getModel(m).getMeshNumber(gid);
                    //std::cout << "Model: " << m << " SDF: " << s << " Frame: " << f << " GID: " << gid << " MID: " << mid << std::endl;
                    if(mid == -1) continue; // Has no mesh

                    // Get the mesh
                    const float3 geomscale = tracker.getModel(m).getGeometryScale(gid); // Scale of the geometry
                    const dart::SE3 geomtfm = tracker.getModel(m).getGeometryTransform(gid); // Constant transform for the geometry from frame origin
                    const dart::Mesh mesh = tracker.getModel(m).getMesh(mid);
                    meshFrameids.push_back(f); // Index of the frame for that particular mesh
                    meshModelids.push_back(m); // ID of the model for that particular mesh

                    // Get their vertices and transform them using the given frame to model transform
                    meshVertices.push_back(std::vector<float3>(mesh.nVertices));
                    transformedMeshVertices.push_back(std::vector<float3>(mesh.nVertices));
                    meshVerticesWMeshID.push_back(std::vector<float4>(mesh.nVertices));
                    for(int i = 0; i < mesh.nVertices; ++i)
                    {
                        // Get mesh vertex in the proper frame of reference
                        // Every geometry can have a fixed transform to it's frame of reference and a scale factor defining its size
                        // To get the vertices in the proper frame of reference (with proper scaling), we have to apply these once
                        meshVertices.back()[i] = geomtfm * (geomscale * mesh.vertices[i]); // Scale each dimension of the vertex by the corresponding scale

                        // Apply transform based on the pose of the object
                        transformedMeshVertices.back()[i] = tfm * meshVertices.back()[i];

                        // Update the canonical vertex (in the correct frame of reference) with the mesh ID
                        meshVerticesWMeshID.back()[i] = make_float4(meshVertices.back()[i].x, meshVertices.back()[i].y,
                                                                    meshVertices.back()[i].z, meshVertices.size()); // Add +1 to mesh IDs (BG is zero)
                    }

                    // For each mesh, initialize memory for the transformed vertex buffers & the (canonical/fixed )mesh vertices with mesh ids
                    std::vector<pangolin::GlBuffer *> attributes;
                    attributes.push_back(new pangolin::GlBuffer(pangolin::GlBufferType::GlArrayBuffer, mesh.nVertices, GL_FLOAT, 3)); // deformed mesh vertices
                    attributes.push_back(new pangolin::GlBuffer(pangolin::GlBufferType::GlArrayBuffer, mesh.nVertices, GL_FLOAT, 4)); // canonical mesh vertices with mesh id
                    attributes[1]->Upload(meshVerticesWMeshID.back().data(), mesh.nVertices*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
                    meshVertexAttributeBuffers.push_back(attributes);

                    // For each mesh, save the faces - one time thing only
                    meshIndexBuffers.resize(meshIndexBuffers.size()+1);
                    meshIndexBuffers.back().Reinitialise(pangolin::GlBufferType::GlElementArrayBuffer, mesh.nFaces*3, GL_INT, 3, GL_DYNAMIC_DRAW);
                    meshIndexBuffers.back().Upload(mesh.faces, mesh.nFaces*sizeof(int3));
                }
            }
        }

        /// ===== Load bag file
        // Open first bag file
        rosbag::Bag bag(pokefolder + "/baxterdata.bag", rosbag::BagMode::Read);

        // Setup topic names
        std::vector<std::string> topic_names = {"/gazebo_aug/obj_contacts",
                                                "/gazebo_aug/link_states",
                                                "/tf"};
        rosbag::View view_data(bag, rosbag::TopicQuery(topic_names));

        // Declare vars
        std::vector<learn_physics_models::LinkStates::ConstPtr> linkstates_messages;
        std::vector<gazebo_msgs::ContactState> contactstates_messages;
        std::vector<tf2_msgs::TFMessage::ConstPtr> tf_messages;
        std::vector<double> tf_timestamps, linkstates_timestamps, contactstates_timestamps;

        // Get all the messages
        BOOST_FOREACH(rosbag::MessageInstance const m, view_data)
        {
            if(m.getTopic() == "/gazebo_aug/link_states")
            {
                learn_physics_models::LinkStates::ConstPtr msg = m.instantiate<learn_physics_models::LinkStates>();
                linkstates_messages.push_back(msg);
                linkstates_timestamps.push_back(msg->header.stamp.toSec());
            }
            else if (m.getTopic() == "/gazebo_aug/obj_contacts")
            {
                gazebo_msgs::ContactsState::ConstPtr css = m.instantiate<gazebo_msgs::ContactsState>();
                if(css != NULL)
                {
                    for(std::size_t j = 0; j < css->states.size(); j++)
                    {
                        contactstates_messages.push_back(css->states[j]);
                    }
                }
            }
            else if(m.getTopic() == "/tf")
            {
                tf2_msgs::TFMessage::ConstPtr msg = m.instantiate<tf2_msgs::TFMessage>();
                tf_messages.push_back(msg);
                tf_timestamps.push_back(msg->transforms[0].header.stamp.toSec()); // Assumes that there's atleast one message here
            }
        }

        // Sort the contact state messages according to time - they can be unsorted!
        std::sort(contactstates_messages.begin(), contactstates_messages.end(),
                  rosmsg_compare_header_stamp<gazebo_msgs::ContactState, gazebo_msgs::ContactState>);
        for(std::size_t k = 0; k < contactstates_messages.size(); k++)
            contactstates_timestamps.push_back(contactstates_messages[k].header.stamp.toSec());

        // Close the bag file
        bag.close();

        // Get the transform between the baxter's base link and the gazebo world frame
        // All link states are w.r.t gazebo world in bag file, but we need them to be w.r.t baxter's base frame
        // for the renderer. So transform by this fixed offset.
        assert(linkstates_messages.size() > 0 && "Did not find any link states in bag. Please check bag file");
        tf::Transform baxter_base_in_gworld;
        bool success = get_link_transform(linkstates_messages[0], "baxter::base", baxter_base_in_gworld);
        std::cout << "Baxter base to Gazebo world transform: " << baxter_base_in_gworld.getOrigin().x() << " " << baxter_base_in_gworld.getOrigin().y() << " "
                  << baxter_base_in_gworld.getOrigin().z() << " " << baxter_base_in_gworld.getRotation().x() << " " << baxter_base_in_gworld.getRotation().y() << " "
                  << baxter_base_in_gworld.getRotation().z() << " " << baxter_base_in_gworld.getRotation().w() << std::endl;
        assert(success && "Could not find transform between baxter::base and gazebo world frame. Cannot proceed");

        /// ===== Load the commanded joint data
        // Read the labels in the CSV file and get valid joint names (both recorded and from model)
        std::vector<std::string> recorded_commanded_joint_names = read_csv_labels(pokefolder + "/commandedpositions.csv");
        std::vector<std::string> valid_commanded_joint_names    = find_common_strings(model_joint_names, recorded_commanded_joint_names);
        cout << "Number of valid commanded DOF: " << valid_commanded_joint_names.size() << endl;

        // Read joint velocities (commanded) for all joints on the robot and the file
        std::vector<std::vector<float> > valid_commanded_joint_positions = read_csv_data<float>(pokefolder + "/commandedpositions.csv", valid_commanded_joint_names);
        cout << "Number of valid commanded joint data from recorded file: " << valid_commanded_joint_positions.size() << endl;

        // Read joint velocities (commanded) for all joints on the robot and the file
        std::vector<std::vector<float> > valid_commanded_joint_velocities = read_csv_data<float>(pokefolder + "/commandedvelocities.csv", valid_commanded_joint_names);
        assert(valid_commanded_joint_positions.size() == valid_commanded_joint_velocities.size() && "Commanded Position and velocity files do not have same number of rows.");

        // Read joint velocities (commanded) for all joints on the robot and the file
        std::vector<std::vector<float> > valid_commanded_joint_accelerations = read_csv_data<float>(pokefolder + "/commandedaccelerations.csv", valid_commanded_joint_names);
        assert(valid_commanded_joint_positions.size() == valid_commanded_joint_accelerations.size() && "Commanded Position and Commanded acceleration files do not have same number of rows.");

        // Get the timestamps from the files
        std::vector<double> commanded_joint_timestamps = read_csv_data<double>(pokefolder + "/commandedpositions.csv", "time");
        assert(valid_commanded_joint_positions.size() == commanded_joint_timestamps.size() && "Commanded Position and Timestamps do not have same number of rows.");

        /// ===== Load the joint data
        /// TODO: Use both arms' data

        // Read the labels in the CSV file and get valid joint names (both recorded and from model)
        std::vector<std::string> recorded_joint_names = read_csv_labels(pokefolder + "/positions.csv");
        std::vector<std::string> valid_joint_names    = find_common_strings(model_joint_names, recorded_joint_names);
        cout << "Number of valid DOF: " << valid_joint_names.size() << endl;

        // Read joint angles for all joints on the robot and the file
        std::vector<std::vector<float> > valid_joint_positions  = read_csv_data<float>(pokefolder + "/positions.csv", valid_joint_names);
        std::vector<std::vector<float> > valid_joint_positions_1  = read_csv_data<float>(pokefolder + "/positions.csv", valid_commanded_joint_names);
        cout << "Number of valid joint data from recorded file: " << valid_joint_positions.size() << endl;

        // Read joint velocities (recorded) for all joints on the robot and the file
        std::vector<std::vector<float> > valid_joint_velocities = read_csv_data<float>(pokefolder + "/velocities.csv", valid_joint_names);
        std::vector<std::vector<float> > valid_joint_velocities_1 = read_csv_data<float>(pokefolder + "/velocities.csv", valid_commanded_joint_names);
        assert(valid_joint_positions.size() == valid_joint_velocities.size() && "Position and Velocity files do not have same number of rows.");

        // Read joint efforts (recorded) for all joints on the robot and the file
        std::vector<std::vector<float> > valid_joint_efforts = read_csv_data<float>(pokefolder + "/efforts.csv", valid_joint_names);
        std::vector<std::vector<float> > valid_joint_efforts_1 = read_csv_data<float>(pokefolder + "/efforts.csv", valid_commanded_joint_names);
        assert(valid_joint_positions.size() == valid_joint_efforts.size() && "Position and Effort files do not have same number of rows.");

        // Get the timestamps from the files
        std::vector<double> joint_timestamps = read_csv_data<double>(pokefolder + "/positions.csv", "time");
        assert(valid_joint_positions.size() == joint_timestamps.size() && "Position and Timestamps do not have same number of rows.");

        /// ===== Load end effector data
        // Read endeff poses
        std::vector<std::string> endeffpose_names = read_csv_labels(pokefolder + "/endeffposes.csv");
        endeffpose_names.erase(endeffpose_names.begin()); // First name is the time
        std::vector<std::vector<float> > valid_endeff_poses = read_csv_data<float>(pokefolder + "/endeffposes.csv", endeffpose_names);
        cout << "Number of valid endeff poses from recorded file: " << valid_endeff_poses.size() << endl;

        // Read endeff twists
        std::vector<std::string> endefftwist_names = read_csv_labels(pokefolder + "/endefftwists.csv");
        endefftwist_names.erase(endefftwist_names.begin()); // First name is the time
        std::vector<std::vector<float> > valid_endeff_twists = read_csv_data<float>(pokefolder + "/endefftwists.csv", endefftwist_names);
        assert(valid_endeff_poses.size() == valid_endeff_twists.size() && "Endeff pose and twist files do not have same number of rows.");

        // Read endeff poses
        std::vector<std::string> endeffwrench_names = read_csv_labels(pokefolder + "/endeffwrenches.csv");
        endeffwrench_names.erase(endeffwrench_names.begin()); // First name is the time
        std::vector<std::vector<float> > valid_endeff_wrenches = read_csv_data<float>(pokefolder + "/endeffwrenches.csv", endeffwrench_names);
        assert(valid_endeff_poses.size() == valid_endeff_wrenches.size() && "Endeff pose and wrench files do not have same number of rows.");

        // Get the timestamps from the files
        std::vector<double> endeff_timestamps = read_csv_data<double>(pokefolder + "/endeffposes.csv", "time");
        assert(valid_endeff_poses.size() == endeff_timestamps.size() && "Endeff pose and timestamps do not have same number of rows.");

        /// ===== Iterate over all the datasets and save the depth/flow/control data

        // Create a folder to save the processed datasets
        std::string save_dir = pokefolder + "/";

        // Get the mesh IDs for each of the robot's commanded joints
        std::vector<int> jointMeshids, valid_joint_ids(valid_joint_names.size());
        for(std::size_t i = 0; i < valid_joint_names.size(); i++)
        {
            // Get ID of the valid joint
            std::string name = valid_joint_names[i];
            int jointid = -1;
            for(int j = 0; j < tracker.getModel(baxterID).getNumJoints(); j++)
            {
                if (tracker.getModel(baxterID).getJointName(j).compare(name) == 0)
                {
                    valid_joint_ids[i] = j;
                    jointid = j;
                    break;
                }
            }

            // Get mesh ID for that joint
            int f = tracker.getModel(baxterID).getJointFrame(jointid);
            for(int k = 0; k < meshFrameids.size(); k++)
            {
                if (f == meshFrameids[k])
                {
                    jointMeshids.push_back(k+1); // Note that "0" is reserved for background amongst mesh ids
                    break;
                }
            }
        }

        // Save the valid joint names in case it is needed later
        std::ofstream labelsfile(save_dir + "statelabels.txt");
        writeVector(labelsfile, valid_joint_names);
        writeVector(labelsfile, jointMeshids); // The mesh label that the joint takes. Can use this to index to "label" image
        labelsfile.close();

        // Save the camera pose
        std::ofstream cameradatafile(save_dir + "cameradata.txt");
        cameradatafile << "Model view matrix: " << endl;
        cameradatafile << modelView << endl;
        cameradatafile << "Camera parameters: " << endl;
        cameradatafile << Eigen::Matrix4f(glK) << endl;
        cameradatafile.close();

        /// === Initialize a counter for the dataset start/end and the various images
        std::vector<float *> depth_images, vmapwmeshid_images;

        // Start timer
        struct timespec tic, toc;
        clock_gettime(CLOCK_REALTIME, &tic);

        /// == Setup time values for depth images
        // Start at initial joint angle timestamp, move in steps of 1/30 (30 Hz) till we reach end
        double t = joint_timestamps[0];
        std::vector<double> depth_timestamps;
        while(t <= joint_timestamps.back())
        {
            depth_timestamps.push_back(t);
            t += (1.0/30);
        }

        // Some stats
        int numframes = depth_timestamps.size();
        double traj_time = joint_timestamps.back() - joint_timestamps.front();
        printf("Total Time: %f, Num images: %d \n", traj_time, numframes);

        // Find closest indices of all the data
        std::vector<int> depth_pos_indices(numframes), depth_commandedpos_indices(numframes);
        std::vector<int> depth_endeff_indices(numframes), depth_linkstates_indices(numframes);
        std::vector<int> depth_contactstates_indices(numframes), depth_tf_indices(numframes);
        for(std::size_t i = 0; i < depth_timestamps.size(); i++)
        {
            depth_pos_indices[i] = find_closest_id(joint_timestamps, depth_timestamps[i]);
            depth_commandedpos_indices[i] = find_closest_id(commanded_joint_timestamps, depth_timestamps[i]);
            depth_endeff_indices[i] = find_closest_id(endeff_timestamps, depth_timestamps[i]);
            depth_linkstates_indices[i] = find_closest_id(linkstates_timestamps, depth_timestamps[i]);
            depth_contactstates_indices[i] = find_closest_id(contactstates_timestamps, depth_timestamps[i]);
            depth_tf_indices[i] = find_closest_id(tf_timestamps, depth_timestamps[i]);
        }

        /// == Resize the image vectors if we need to
        for(int k = depth_images.size(); k < numframes; k++)
        {
            depth_images.push_back(new float[glWidth * glHeight]); // 1-channel float image
            vmapwmeshid_images.push_back(new float[glWidth * glHeight * 4]); // 4-channel float image
        }

        /// == Read meta data file, use it to post-process results to generate the event times
        // Meta data vars
        std::vector<std::string> object_names;
        int num_objects;
        double table_friction;
        std::string target_link_name;
        std::string obj_common_str = "obj_"; // Common string in all object names

        // Read the meta data file
        std::ifstream ifs(pokefolder + "/metadata.txt");
        ifs >> num_objects;
        ifs >> table_friction;
        if (num_objects > 1)
        {
            // Read names of all objects
            std::string line;
            std::getline(ifs, line);
            boost::split(object_names, line, boost::is_any_of(","), boost::token_compress_on); // Split on ","
            // Read target object name
            std::getline(ifs, target_link_name);
            target_link_name = target_link_name + "::link";
        }
        else // For single object, it is the target of the poke. Also the common string
        {
            // Get target link name
            assert(num_objects == 1);
            learn_physics_models::LinkStates::ConstPtr msg = linkstates_messages[depth_linkstates_indices[0]];
            for(std::size_t j = 0; j < msg->name.size(); ++j)
            {
                for(std::size_t m = 0; m < tracker.getNumModels(); m++)
                {
                    if(model_id_to_name[m] != "baxter" && model_id_to_name[m] != "table" &&
                            msg->name[j].find(model_id_to_name[m] + "::link") != std::string::npos)
                    {
                        target_link_name = msg->name[j];
                    }
                }
            }

            // Common string = object name now (As there is only one object)
            obj_common_str = target_link_name;
            object_names.push_back(target_link_name);
        }

        // == Post process the experiment state (search for all objects)
        std::cout << std::endl << "Metadata: " << std::endl;
        std::cout << "Num objects: " << num_objects << ", Friction: " << table_friction << std::endl;
        std::cout << "Object common string: " << obj_common_str << ", Target link name: "
                  << target_link_name << std::endl;
        std::map<std::string, double> map_events_to_times = get_experiment_event_times(contactstates_messages,
                                                                                       linkstates_messages,
                                                                                       tableOBB,
                                                                                       obj_common_str,
                                                                                       target_link_name);
        // Save this data to disk
        std::string events_file_path = pokefolder + "/events.txt";
        save_event_time_data(map_events_to_times, depth_timestamps, events_file_path);
        std::cout << std::endl;

        /// == Save the link states (closest to depth maps)
        for(int counter = 0; counter < numframes; counter++)
        {
            // Write state of gazebo links
            std::string state_file_name = pokefolder + "/linkstate" + std::to_string(counter) + ".txt";
            save_link_states(linkstates_messages[depth_linkstates_indices[counter]], state_file_name);

            // Append the pose of the depth camera link from TF to it
            std::ofstream ofs;
            ofs.open(state_file_name, std::ofstream::out | std::ofstream::app);
            write_link_state_to_stream(ofs, "kinect::link",
                                       tf_transform_to_geometry_msgs_pose(base_to_depth_camera.inverse()),
                                       geometry_msgs::Twist(), geometry_msgs::Accel(),
                                       geometry_msgs::Wrench());
            ofs.close(); // Close stream
        }

        /// == Append the number of depth images to the metadata
        if((get_number_of_lines(pokefolder + "/metadata.txt")-1) == 4) // Only if num lines = 4
        {
            std::ofstream ofs;
            ofs.open(pokefolder + "/metadata.txt", std::ofstream::out | std::ofstream::app);
            ofs << numframes << std::endl; // Write number of depth files to meta data
        }

        /// == Render the depth maps and the vertex data for all these steps first
        std::vector<std::vector<dart::SE3> > mesh_transforms(numframes);
        //bool selfcollision = false;
        for(int counter = 0; counter < numframes; counter++)
        {
            /// == Update model pose for all the models in the scene
            for(std::size_t m = 0; m < tracker.getNumModels(); m++)
            {
                // == Get the pose of the robot at that timestep and update the pose of the robot
                if (model_id_to_name[m] == "baxter")
                {
                    for(std::size_t k = 0; k < valid_joint_names.size(); k++)
                    {
                        if (joint_name_to_pose_dim.find(valid_joint_names[k]) != joint_name_to_pose_dim.end())
                        {
                            int pose_dim = joint_name_to_pose_dim[valid_joint_names[k]];
                            tracker.getPose(m).getReducedArticulation()[pose_dim] = valid_joint_positions[depth_pos_indices[counter]][k];
                        }
                    }
                }
                else
                {
                    /// == Get transform of object
                    tf::Transform tfm; // Name of model
                    learn_physics_models::LinkStates::ConstPtr msg = linkstates_messages[depth_linkstates_indices[counter]];
                    bool success = get_link_transform(msg, model_id_to_name[m] + "::link", tfm);

                    // If pose exists, use that or set default pose
                    if (!success) // can't find in recorded message
                    {
                        tracker.getPose(m).setTransformModelToCamera(defaultPose); // Set default pose for object
                    }
                    else
                    {
                        // Get pose from bag file
                        // Bag file pose is object in gazebo_world (T_gw_o)
                        // We need object in baxter base (T_bb_o) := T_bb_gw * T_gw_o
                        tf::Transform tfm_inbase = baxter_base_in_gworld.inverseTimes(tfm);
                        tf::Vector3 t = tfm_inbase.getOrigin();
                        tf::Quaternion q = tfm_inbase.getRotation();
                        dart::SE3 pose = dart::SE3Fromse3(dart::se3(t.x(), t.y(), t.z(), 0, 0, 0))*
                                         dart::SE3Fromse3(dart::se3(0, 0, 0,
                                                                    q.getAngle()*q.getAxis().x(),
                                                                    q.getAngle()*q.getAxis().y(),
                                                                    q.getAngle()*q.getAxis().z())); // default pose is behind camera
                        tracker.getPose(m).setTransformModelToCamera(pose);
                    }
                }

                // == Update the pose so that the FK is computed properly
                tracker.updatePose(m);
            }

            /// == Update mesh vertices based on new pose
            for (int i = 0; i < meshVertices.size(); i++)
            {
                // Get the SE3 transform for the frame
                int m = meshModelids[i];
                int f = meshFrameids[i];
                const dart::SE3 modeltfm = tracker.getModel(m).getTransformModelToCamera();
                const dart::SE3 tfm = modeltfm * tracker.getModel(m).getTransformFrameToModel(f); // model_to_cam * frame_to_model
                mesh_transforms[counter].push_back(dart::SE3(tfm.r0, tfm.r1, tfm.r2)); // Create a copy of the SE3 and save it

                // Transform the canonical vertices based on the new transform
                for(int j = 0; j < meshVertices[i].size(); ++j)
                {
                    transformedMeshVertices[i][j] = tfm * meshVertices[i][j];
                }

                // Upload to pangolin vertex buffer
                meshVertexAttributeBuffers[i][0]->Upload(transformedMeshVertices[i].data(), transformedMeshVertices[i].size()*sizeof(float3));
                meshVertexAttributeBuffers[i][1]->Upload(meshVerticesWMeshID[i].data(), meshVerticesWMeshID[i].size()*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
            }

            /// == Render a depth image
            glEnable(GL_DEPTH_TEST);
            renderer.renderMeshes<l2s::RenderDepth>(meshVertexAttributeBuffers, meshIndexBuffers);

            // Get a float image
            renderer.texture<l2s::RenderDepth>().Download(depth_images[counter]);

            /// == Render a vertex map with the mesh ids
            renderer.renderMeshes<l2s::RenderVertMapWMeshID>(meshVertexAttributeBuffers, meshIndexBuffers);

            // Get the vertex image with mesh id
            renderer.texture<l2s::RenderVertMapWMeshID>().Download(vmapwmeshid_images[counter], GL_RGBA, GL_FLOAT);
        }

        /// == Save the depth and pixel label images to disk
        std::vector<cv::Mat> depth_mat_f(numframes), labels_mat_c(numframes);
        for(int counter = 0; counter < numframes; counter++)
        {
            // == Convert depth float to an opencv matrix
            depth_mat_f[counter] = cv::Mat(glHeight, glWidth, CV_32FC1, depth_images[counter]);

            // == Create a new matrix for the mask
            labels_mat_c[counter] = cv::Mat(glHeight, glWidth, CV_8UC1);
            int ct = 0;
            for(int r = 0; r < glHeight; r++)
            {
                for(int c = 0; c < glWidth; c++, ct+=4)
                {
                    float *point = &(vmapwmeshid_images[counter][ct]);
                    labels_mat_c[counter].at<char>(r,c) = (char) std::round(point[3]);
                }
            }

            // == Save full res data
            if(save_fullres_data)
            {
                // Convert the depth data from "m" to "0.1 mm" range and store it as a 16-bit unsigned short
                // 2^16 = 65536  ;  1/10 mm = 1/10 * 1e-3 m = 1e-4 m  ;  65536 * 1e-4 = 6.5536
                // We can represent depth from 0 to +6.5536 m using this representation (enough for our data)
                cv::Mat depth_mat;
                depth_mat_f[counter].convertTo(depth_mat, CV_16UC1, M_TO_TENTH_MM); // 0.1 mm resolution and round off to *nearest* unsigned short
                cv::imwrite(save_dir + "depth" + std::to_string(counter) + ".png", depth_mat); // Save depth image

                // Save mask image
                cv::imwrite(save_dir + "labels" + std::to_string(counter) + ".png", labels_mat_c); // Save depth image
            }

            // == Subsample depth image by NN interpolation (don't do cubic/bilinear interp) and save it
            cv::Mat depth_mat_f_sub(std::round(0.5 * glHeight), std::round(0.5 * glWidth), CV_32FC1);
            cv::resize(depth_mat_f[counter], depth_mat_f_sub, cv::Size(), 0.5, 0.5, CV_INTER_NN); // Do NN interpolation

            // Scale the depth image as a 16-bit single channel unsigned char image
            cv::Mat depth_mat_sub;
            depth_mat_f_sub.convertTo(depth_mat_sub, CV_16UC1, M_TO_TENTH_MM); // Scale from m to 0.1 mm resolution and save as ushort
            cv::imwrite(save_dir + "depthsub" + std::to_string(counter) + ".png", depth_mat_sub); // Save depth image

            // == Subsample label image and save it
            cv::Mat labels_mat_c_sub(std::round(0.5 * glHeight), std::round(0.5 * glWidth), CV_8UC1);
            cv::resize(labels_mat_c[counter], labels_mat_c_sub, cv::Size(), 0.5, 0.5, CV_INTER_NN); // Do NN interpolation
            cv::imwrite(save_dir + "labelssub" + std::to_string(counter) + ".png", labels_mat_c_sub); // Save depth image
        }

        /// == Compute flows for each of the steps and save the data to disk
        // Iterate over the different requested "step" lengths and compute flow for each
        // Each of these is stored in their own folder - "flow_k" where "k" is the step-length
        for(std::size_t k = 0; k < step_list.size(); k++)
        {
            // Get step size and create flow dir
            int step = step_list[k];
            std::string flow_dir = save_dir + "/flow_" + std::to_string(step) + "/";
            createDirectory(flow_dir);
            printf("Computing flows for step length: %d \n", step);

            // Iterate over depth images to compute flow for each relevant one
            std::vector<cv::Mat> flow_mat_f(numframes); // used for visualizing data
            for(int counter = 0; counter < numframes-step; ++counter)
            {
                // Compute flow multi-threaded
                tg.addTask([&,counter]() // Lambda function/thread
                {
                    // Compute flow between frames @ t & t+step
                    flow_mat_f[counter] = compute_flow_between_frames(vmapwmeshid_images[counter],
                                                                      mesh_transforms[counter],
                                                                      mesh_transforms[counter+step],
                                                                      modelView, glWidth, glHeight);

                    // Convert the flow data from "m" to "0.1 mm" range and store it as a 16-bit unsigned short
                    // 2^16 = 65536. 2^16/2 ~ 32767 * 1e-4 (0.1mm) = +-3.2767 (since we have positive and negative values)
                    // We can represent motion ranges from -3.2767 m to +3.2767 m using this representation
                    // This amounts to ~300 cm/frame ~ 90m/s speed (which should be way more than enough for the motions we currently have)
                    if(save_fullres_data)
                    {
                        cv::Mat flow_mat = convert_32FC3_to_16UC3(flow_mat_f[counter], M_TO_TENTH_MM);
                        cv::imwrite(flow_dir + "flow" + std::to_string(counter) + ".png", flow_mat);
                    }

                    // Save subsampled image by NN interpolation
                    // KEY: Don't do cubic/bilinear interpolation as it leads to interpolation of flow values across boundaries which is incorrect
                    cv::Mat flow_mat_f_sub(round(0.5 * glHeight), round(0.5 * glWidth), CV_32FC3);
                    cv::resize(flow_mat_f[counter], flow_mat_f_sub, cv::Size(), 0.5, 0.5, CV_INTER_NN); // Do NN interpolation (otherwise flow gets interpolated, which is incorrect)

                    // Scale the flow image data as a 16-bit 3-channel ushort image & save it
                    cv::Mat flow_mat_sub = convert_32FC3_to_16UC3(flow_mat_f_sub, M_TO_TENTH_MM);
                    cv::imwrite(flow_dir + "flowsub" + std::to_string(counter) + ".png", flow_mat_sub);
                });
            }
            tg.wait();

            // In case we are asked to visualize the flow outputs and/or images do it here
            if(visualize)
            {
                // Iterate over all the depth images
                for(int counter = 0; counter < numframes-step; ++counter)
                {
                    // ==== Image stuff
                    cv::Mat depth(glHeight, glWidth, CV_32F, depth_images[counter]);
                    cv::imshow("Subsampled Colorized Depth", colorize_depth(depth));
                    cv::imshow("Labels", colorize_depth(labels_mat_c[counter]));
                    cv::imshow("Flow", colorize_depth(flow_mat_f[counter]));
                    cv::waitKey(2);      // wait for 2 ms

                    // ==== PCL visualization
                    if (pcl_visualize)
                    {
                        // == Flow stuff
                        // Get lock on the mutex
                        boost::mutex::scoped_lock update_lock(point_cloud_mutex);

                        // Update point cloud and flow vector
                        point_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>(glWidth, glHeight)); // Set point cloud for viewing
                        flow_cloud.reset(new pcl::PointCloud<pcl::Normal>(glWidth, glHeight)); // Update flow cloud
                        for (std::size_t r = 0; r < point_cloud->height; r++) // copy over from the flow matrix
                        {
                           for (std::size_t c = 0; c < point_cloud->width; c++)
                           {
                               float z = depth_mat_f[counter].at<float>(r,c);
                               float x = (c - glPPx)/glFLx; x *= z;
                               float y = (r - glPPy)/glFLy; y *= z;
                               point_cloud->at(c,r) = pcl::PointXYZ(x,y,z);
                               flow_cloud->at(c,r) = pcl::Normal(flow_mat_f[counter].at<cv::Vec3f>(r,c)[0],
                                                                 flow_mat_f[counter].at<cv::Vec3f>(r,c)[1],
                                                                 flow_mat_f[counter].at<cv::Vec3f>(r,c)[2]);
                           }
                        }

                        // Set flag for visualizer
                        point_cloud_update = true;
                        update_lock.unlock(); // Unlock mutex so that display happens
                    }

                    // ==== Update model pose for all the models in the scene
                    for(std::size_t m = 0; m < tracker.getNumModels(); m++)
                    {
                        // == Get the pose of the robot at that timestep and update the pose of the robot
                        if (model_id_to_name[m] == "baxter")
                        {
                            for(std::size_t k = 0; k < valid_joint_names.size(); k++)
                            {
                                if (joint_name_to_pose_dim.find(valid_joint_names[k]) != joint_name_to_pose_dim.end())
                                {
                                    int pose_dim = joint_name_to_pose_dim[valid_joint_names[k]];
                                    tracker.getPose(m).getReducedArticulation()[pose_dim] = valid_joint_positions[depth_pos_indices[counter]][k];
                                }
                            }
                        }
                        else
                        {
                            // == Get transform of object
                            tf::Transform tfm; // Name of model
                            learn_physics_models::LinkStates::ConstPtr msg = linkstates_messages[depth_linkstates_indices[counter]];
                            bool success = get_link_transform(msg, model_id_to_name[m] + "::link", tfm);

                            // If pose exists, use that or set default pose
                            if (!success) // can't find in recorded message
                            {
                                tracker.getPose(m).setTransformModelToCamera(defaultPose); // Set default pose for object
                            }
                            else
                            {
                                // Get pose from bag file
                                // Bag file pose is object in gazebo_world (T_gw_o)
                                // We need object in baxter base (T_bb_o) := T_bb_gw * T_gw_o
                                tf::Transform tfm_inbase = baxter_base_in_gworld.inverseTimes(tfm);
                                tf::Vector3 t = tfm_inbase.getOrigin();
                                tf::Quaternion q = tfm_inbase.getRotation();
                                dart::SE3 pose = dart::SE3Fromse3(dart::se3(t.x(), t.y(), t.z(), 0, 0, 0))*
                                                 dart::SE3Fromse3(dart::se3(0, 0, 0,
                                                                            q.getAngle()*q.getAxis().x(),
                                                                            q.getAngle()*q.getAxis().y(),
                                                                            q.getAngle()*q.getAxis().z())); // default pose is behind camera
                                tracker.getPose(m).setTransformModelToCamera(pose);
                            }
                        }
                    }

                    // ==== Render pangolin frame
                    renderPangolinFrame(tracker, camState, camDisp);
                }
            }
        }

        /// == Save the joint states, velocities and efforts (recorded - closest to depth maps)
        printf("Saving recorded joint states to disk \n");
        for(int counter = 0; counter < numframes; counter++)
        {
            // Create a new file per state
            std::ofstream statefile(pokefolder + "/state" + std::to_string(counter) + ".txt");

            // Write joint states, velocities and efforts
            writeVector(statefile, valid_joint_positions_1[depth_pos_indices[counter]]);
            writeVector(statefile, valid_joint_velocities_1[depth_pos_indices[counter]]);
            writeVector(statefile, valid_joint_efforts_1[depth_pos_indices[counter]]);

            // Write end eff pose, twist, wrenches
            writeVector(statefile, valid_endeff_poses[depth_endeff_indices[counter]]);
            writeVector(statefile, valid_endeff_twists[depth_endeff_indices[counter]]);
            writeVector(statefile, valid_endeff_wrenches[depth_endeff_indices[counter]]);

            // Close file
            statefile.close();
        }

        /// == Save the joint states, velocities and efforts (commanded - closest to depth maps)
        printf("Saving commanded joint states to disk \n");
        std::vector<float> temp1 = {0,0,0,0,0,0,0}; // pos,quat
        std::vector<float> temp2 = {0,0,0,0,0,0}; // 6D pose
        for(int counter = 0; counter < numframes; counter++)
        {
            // Create a new file per state
            std::ofstream statefile(pokefolder + "/commandedstate" + std::to_string(counter) + ".txt");

            // Write joint states, velocities and efforts

            writeVector(statefile, valid_commanded_joint_positions[depth_commandedpos_indices[counter]]);
            writeVector(statefile, valid_commanded_joint_velocities[depth_commandedpos_indices[counter]]);
            writeVector(statefile, valid_commanded_joint_accelerations[depth_commandedpos_indices[counter]]);

            // Write end eff pose, twist, wrenches
            writeVector(statefile, temp1); // Zero
            writeVector(statefile, temp2); // Zero
            writeVector(statefile, temp2); // Zero

            // Close file
            statefile.close();
        }

        // Time taken
        clock_gettime(CLOCK_REALTIME, &toc);
        timespec_sub(&toc, &tic);
        printf("Time taken for dataset: %f \n",timespec_double(&toc));
        cout << " ================================= " << endl << endl;

        // Free memory for depth images
        for(std::size_t i = 0; i < depth_images.size(); i++)
            free(depth_images[i]);

        // Free memory for the vertex map images
        for(std::size_t i = 0; i < vmapwmeshid_images.size(); i++)
            free(vmapwmeshid_images[i]);

        // Free memory for the rendering buffers
        for(std::size_t i = 0; i < meshVertexAttributeBuffers.size(); i++)
            for(size_t j = 0; j < meshVertexAttributeBuffers[i].size(); j++)
                free(meshVertexAttributeBuffers[i][j]);
    }

    return 0;
}

