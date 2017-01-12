// Common
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// ROS headers
#include <ros/ros.h>
#include <ros/console.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

// Messages
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf2_msgs/TFMessage.h>
#include <sensor_msgs/CameraInfo.h>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

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
#include "depth_sources/image_depth_source.h"

// Read CSV stuff
#include "util/csv_util.h"

// Utility functions
#include <helperfuncs.h>
#include <threadPool.h>

// OpenCV stuff
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>

// Boost
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>

// PCL
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

#define EIGEN_DONT_ALIGN
#define M_TO_MM 1e3
#define M_TO_TENTH_MM 1e4 // Represent data as one tenth of a mm
#define M_TO_HUNDREDTH_MM 1e5 // Represent data as one hundredth of a mm
#define pointColoringNone 0
#define pointColoringRGB 1
#define pointColoringErr 2
#define pointColoringDA 3

const int nPointColorings = 4;
const int panelWidth = 180;
using namespace std;
namespace po = boost::program_options;

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
cv::Mat compute_flow_between_frames(const cv::Mat &pointCloud,
                                    const cv::Mat &dataAssMap,
                                    const std::vector<std::vector<dart::SE3> > &t_fcs_1,
                                    const std::vector<std::vector<dart::SE3> > &t_fcs_2,
                                    const int width = 640, const int height = 480)
{
    // Compute the depth map based on the link transform and see if it matches what the depth map has
    cv::Mat flow_mat(height, width, CV_32FC3, cv::Scalar(0));
    for(int i = 0; i < flow_mat.rows; i++)
    {
        for(int j = 0; j < flow_mat.cols; j++)
        {
            // Get the point in the camera coordinate frame
            cv::Vec3f point = pointCloud.at<cv::Vec3f>(i,j);

            // Get the model and SDF corresponding to that point
            int model = dataAssMap.at<cv::Vec2i>(i,j)[0];
            int sdf = dataAssMap.at<cv::Vec2i>(i,j)[1];
            if (model == -1 || sdf == -1) // No data association, zero flow
                continue;

            // Get vertex point (for that pixel) and the mesh to which it belongs
            float3 camVertex = make_float3(point[0], point[1], point[2]);

            // Two steps:
            // a) Transform the point in the camera to the SDF's reference system @ time "t"
            // b) The SDF has moved to a new pose @ time "t+1". To get the new vertex in the camera, we need to
            //    take this point in the SDF reference system @ "t+1" to the camera's reference system.
            //    We use the inverse of the camera/frame transform for this.
            float3 sdfVertex    = t_fcs_1[model][sdf] * camVertex;
            float3 camVertex2   = dart::SE3Invert(t_fcs_2[model][sdf]) * sdfVertex; // The frame has moved now. So apply the new transform, but the inverse to move the point back to the camera

            // Get the flow vector in the camera's referece system
            flow_mat.at<cv::Vec3f>(i,j) = cv::Vec3f(camVertex2.x - camVertex.x,
                                                    camVertex2.y - camVertex.y,
                                                    camVertex2.z - camVertex.z);
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
                         const pangolin::View &camDisp, bool showVoxelized = false, float levelSet = 0)
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
        tracker.getModel(m).renderCoordinateFrames();
        tracker.getModel(m).renderSkeleton();
        if (showVoxelized) tracker.getModel(m).renderVoxels(levelSet);
    }

    /////////////////
    glPopMatrix();
    glDisable(GL_LIGHTING);
    glColor4ub(255,255,255,255);

    /// Finish frame
    pangolin::FinishFrame();
}

// Returns true if the input path is a directory or false otherwise.
bool doesFileExist(std::string path)
{
    boost::filesystem::path bPath(path);
    if (boost::filesystem::exists(bPath)) // In case it does not exist, create it
    {
        if(boost::filesystem::is_regular_file(bPath))
        {
            return true;
        }
        return false;
    }
    else
        return false;
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

// Read the data from a file
template <typename T>
std::vector<T> read_timestamps(const std::string &filename)
{
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        cout << "Cannot open file.\n";
        return std::vector<T>();
    }

    // Get number of lines
    int num_lines = get_number_of_lines(filename);
    std::vector<T> timestamps(num_lines);
    for(int k = 0; k < num_lines; k++)
    {
        file >> timestamps[k];
    }

    // Close file
    file.close();

    // Return timestamps
    return timestamps;
}

inline float3 make_float3(const tf::Vector3 &vec)
{
    return make_float3(vec.x(), vec.y(), vec.z());
}

inline float3 make_float3(const Eigen::Vector3f &vec)
{
    return make_float3(vec(0), vec(1), vec(2));
}

tf::Vector3 minvec(const tf::Vector3 &a, const tf::Vector3 &b)
{
    return tf::Vector3(std::min(a.x(), b.x()),
                       std::min(a.y(), b.y()),
                       std::min(a.z(), b.z()));
}

tf::Vector3 maxvec(const tf::Vector3 &a, const tf::Vector3 &b)
{
    return tf::Vector3(std::max(a.x(), b.x()),
                       std::max(a.y(), b.y()),
                       std::max(a.z(), b.z()));
}

Eigen::Vector3f minvec(const Eigen::Vector3f &a, const Eigen::Vector3f &b)
{
    return Eigen::Vector3f(std::min(a(0), b(0)),
                           std::min(a(1), b(1)),
                           std::min(a(2), b(2)));
}

Eigen::Vector3f maxvec(const Eigen::Vector3f &a, const Eigen::Vector3f &b)
{
    return Eigen::Vector3f(std::max(a(0), b(0)),
                           std::max(a(1), b(1)),
                           std::max(a(2), b(2)));
}

void transformOBBPose(const OBB &obb, const tf::Transform &tfm,
                      Eigen::Vector3f &t_pos, Eigen::Matrix3f &t_rot)
{
    // Get table rotation in camera frame of reference
    tf::Quaternion quat = tfm.getRotation();
    Eigen::Quaternionf eigquat(quat.getW(), quat.getX(), quat.getY(), quat.getZ());
    t_rot = eigquat.matrix() * obb.rotationmatrix;

    // Get table center and extents in camera frame of reference
    tf::Vector3 cent = tfm * tf::Vector3(obb.center[0], obb.center[1], obb.center[2]);
    t_pos = Eigen::Vector3f(cent.x(), cent.y(), cent.z());
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

cv::Mat normalizeImage(const cv::Mat &img)
{
    // Convert to float
    cv::Mat img1;
    img.convertTo(img1, CV_32FC1);

    // Normalize
    double min_val, max_val;
    cv::minMaxIdx(img1, &min_val, &max_val);
    return (img1 - min_val)/(max_val - min_val);
}

cv::Mat getChannel(const cv::Mat &img, const int c, const bool normalize=false)
{
    // Split to get channels
    std::vector<cv::Mat> channels(img.channels());
    cv::split(img, channels);
    if (normalize)
        return normalizeImage(channels[c]);
    else
        return channels[c];
}

namespace boost
{
    namespace serialization
    {
        template<class Archive>
        inline void serialize(Archive & ar, dart::SE3 & se3, const unsigned int version)
        {
            // Save r0
            ar & se3.r0.w;
            ar & se3.r0.x;
            ar & se3.r0.y;
            ar & se3.r0.z;

            // Save r1
            ar & se3.r1.w;
            ar & se3.r1.x;
            ar & se3.r1.y;
            ar & se3.r1.z;

            // Save r2
            ar & se3.r2.w;
            ar & se3.r2.x;
            ar & se3.r2.y;
            ar & se3.r2.z;
        }
    }
}


void serializeSE3(std::ofstream &file, const dart::SE3 &se3)
{
    // Save r0
    file << se3.r0.w << " ";
    file << se3.r0.x << " ";
    file << se3.r0.y << " ";
    file << se3.r0.z << " ";

    // Save r1
    file << se3.r1.w << " ";
    file << se3.r1.x << " ";
    file << se3.r1.y << " ";
    file << se3.r1.z << " ";

    // Save r2
    file << se3.r2.w << " ";
    file << se3.r2.x << " ";
    file << se3.r2.y << " ";
    file << se3.r2.z << " ";
}

dart::SE3 deserializeSE3(std::ifstream &file)
{
    // SE3
    dart::SE3 se3;

    // Get r0
    file >> se3.r0.w;
    file >> se3.r0.x;
    file >> se3.r0.y;
    file >> se3.r0.z;

    // Get r1
    file >> se3.r1.w;
    file >> se3.r1.x;
    file >> se3.r1.y;
    file >> se3.r1.z;

    // Get r2
    file >> se3.r2.w;
    file >> se3.r2.x;
    file >> se3.r2.y;
    file >> se3.r2.z;

    // Return
    return se3;
}

// -----------------------------------------
// Main
int main(int argc, char **argv)
{
    /// ===== Get parameters

    // Default parameters
    std::string pokefolder      = "";       // Folder to load saved poke images from. Flow vectors will be saved in these folders.
    std::string modelfolder     = "";       // Folder containing all dart models
    bool visualize              = true;     // Visualize the data in pangolin
    bool save_fullres_data      = true;    // By default, do not save full res images
    int num_threads             = 5;        // By default, we use 5 threads
    std::string step_list_str   = "[1]";    // Steps (depth images) in future to look at computing flow (Default: 1 => t+1)
    std::string tabletop_object = "cheezeit";       // Object on top of the table
    bool track_baxter           = false;    // By default, we only track the object using DART. Not the baxter model
    bool cache_sdfs             = false;    // By default, we do not cache SDFs
    bool use_stick              = false;    // Do not use stick

    //PARSE INPUT ARGUMENTS
    po::options_description desc("Allowed options",1024);
    desc.add_options()
        ("help", "produce help message")
        ("pokefolder", po::value<std::string>(&pokefolder), "Path to load data from. Will load/save data from all sub-directories")
        ("modelfolder", po::value<std::string>(&modelfolder), "Folder containing all dart models")
        ("visualize",  po::value<bool>(&visualize), "Flag for visualizing trajectories. [Default: 1]")
        ("savefullres",po::value<bool>(&save_fullres_data), "Flag for saving full resolution image data. [Default: 1]")
        ("numthreads", po::value<int>(&num_threads), "Number of threads to use")
        ("steplist",   po::value<std::string>(&step_list_str), "Comma separated list of steps to compute flow. [Default: [1]]")
        ("object",     po::value<std::string>(&tabletop_object), "Object present on top of the table [Default: '']")
        ("trackbaxter",    po::value<bool>(&track_baxter), "Track baxter model using DART")
        ("cachesdfs",      po::value<bool>(&cache_sdfs), "Cache SDFs and use the previously cached SDFs")
        ("usestick",   po::value<bool>(&use_stick), "Use baxter model with the stick" )
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
    assert(!pokefolder.empty() && "Please pass in folder with recorded data using --pokefolder <path_to_folder>");
    assert(!modelfolder.empty() && "Please pass in folder with dart models using --modelfolder <path_to_folder>");
    cout << "Loading data from directory: " << pokefolder << endl;
    cout << "Loading models from directory: " << modelfolder << endl;
    if(save_fullres_data) cout << "Saving full resolution images [480x640] in addition to default [240x320] images" << endl;
    cout << "Using " << num_threads << " threads to speed up processing" << endl;

    // Parse step list to get all the queried "step" lengths
    std::string step_list_substr = step_list_str.substr(1, step_list_str.size() - 2); // Remove braces []
    std::vector<std::string> words;
    boost::split(words, step_list_substr, boost::is_any_of(","), boost::token_compress_on); // Split on ","

    // Convert the strings to integers
    std::vector<int> step_list;
    std::transform(words.begin(), words.end(), std::back_inserter(step_list), string_to_int);
    cout << "Will compute flows for " << step_list.size() << " different step(s)" << endl;

    // Check which tabletop object has been used
    assert(!tabletop_object.compare("cheezeit") || !tabletop_object.compare("mustard") || !tabletop_object.compare("pringles")
           && "Tabletop object can be: cheezeit, mustard or pringles");
    printf("Tabletop object: %s",tabletop_object.c_str());

    // Baxter model
    int num_models = 1;
    if (track_baxter)
    {
        printf("Tracking the baxter model along with the table top object. \n");
        num_models++;
    }

    // Allow multi-threaded support for speeding up computation
    rgbd::threadGroup tg(num_threads);
    printf("Using %d threads to speed up computation \n",num_threads);

    // Cache SDFs
    if (cache_sdfs)
        printf("Caching sdfs at /tmp/ and reusing the previously cached SDFs from /tmp/ \n");

    /// ===== Get camera parameters
    // Open first bag file
    rosbag::Bag bag(pokefolder + "/baxterdata.bag", rosbag::BagMode::Read);

    // Get info messages
    sensor_msgs::CameraInfo::ConstPtr depth_camera_info;
    rosbag::View view_camera_info(bag, rosbag::TopicQuery("/camera/depth_registered/camera_info"));
    BOOST_FOREACH(rosbag::MessageInstance const m, view_camera_info)
    {
        if(m.getTopic() == "/camera/depth_registered/camera_info")
        {
            depth_camera_info = m.instantiate<sensor_msgs::CameraInfo>();
            break;
        }
    }

    // Close bag
    bag.close();

    // Get camera parameters
    int glHeight  = depth_camera_info->height;
    int glWidth   = depth_camera_info->width;
    float glFLx   = depth_camera_info->K[0];
    float glFLy   = depth_camera_info->K[4];
    float glPPx   = depth_camera_info->K[2];
    float glPPy   = depth_camera_info->K[5];
    std::string depth_camera_frame = depth_camera_info->header.frame_id;

    // Get base TF w.r.t camera (fixed)
    tf::Transform base_to_depth_camera;
    base_to_depth_camera.setOrigin(tf::Vector3(0.005, 0.508, 0.126));
    base_to_depth_camera.setRotation(tf::Quaternion(0.630, -0.619, 0.336, 0.326));
    tf::Quaternion q = base_to_depth_camera.getRotation();
    dart::SE3 base_to_depth_camera_se3 = dart::SE3Fromse3(dart::se3(0.005, 0.508, 0.126, 0, 0, 0))*
                                         dart::SE3Fromse3(dart::se3(0, 0, 0,
                                                                    q.getAngle()*q.getAxis().x(),
                                                                    q.getAngle()*q.getAxis().y(),
                                                                    q.getAngle()*q.getAxis().z())); // dart::se3 rotates the translations, so we need to do this 2-step thing

    /*
    // Get Camera TF w.r.t base
    tf::Transform depth_camera_to_base;
    rosbag::View view_tf(bag, rosbag::TopicQuery("/tf"));
    BOOST_FOREACH(rosbag::MessageInstance const m, view_tf)
    {
        // Get TF message
        if(m.getTopic() == "/tf")
        {
            tf2_msgs::TFMessage::ConstPtr tf_message = m.instantiate<tf2_msgs::TFMessage>();
            bool success = lookup_tf_transform(tf_message, depth_camera_frame, "/base", depth_camera_to_base);
            if (success)
            {
                cout << "Transform between camera and base.";
                cout << "Trans: (" << depth_camera_to_base.getOrigin().x() << " " << depth_camera_to_base.getOrigin().y()
                     << depth_camera_to_base.getOrigin().z() << ")" ;
                cout << "Rot: (" << depth_camera_to_base.getOrigin().x() << " " << depth_camera_to_base.getOrigin().y()
                     << depth_camera_to_base.getOrigin().z() << ")" << endl;
                break; // Exit if we get a successful transform
            }
        }
    }
    */

    /// ===== Pangolin initialization
    /// Pangolin mirrors the display, so we need to use TopLeft display for it. Our rendering needs BottomLeft

    // Setup OpenGL/CUDA/Pangolin stuff
    cudaGLSetGLDevice(0);
    cudaDeviceReset();
    const float totalwidth = 1920;
    const float totalheight = 1080;

    pangolin::CreateWindowAndBind("Main", totalwidth, totalheight);
    glewInit();
    glutInit(&argc, argv);

    // -=-=-=- pangolin window setup -=-=-=-
    pangolin::CreatePanel("lim").SetBounds(0.0,0.9,1.0,pangolin::Attach::Pix(-panelWidth)); //Attach::Pix(UI_WIDTH));
    pangolin::CreatePanel("ui").SetBounds(0.0,0.9, 0.0, pangolin::Attach::Pix(panelWidth));
    pangolin::CreatePanel("opt").SetBounds(0.0,0.9,pangolin::Attach::Pix(panelWidth), pangolin::Attach::Pix(2*panelWidth));
    pangolin::CreatePanel("data").SetBounds(0.9,1.0,0.0,1.0);

    // Initialize camera parameters and projection matrix
    pangolin::OpenGlMatrixSpec glK_pangolin = pangolin::ProjectionMatrixRDF_TopLeft(glWidth,glHeight,glFLx,glFLy,glPPx,glPPy,0.01,1000);

    // Create a model view matrix (based on the camera TF w.r.t base)
    //pangolin::OpenGlMatrix glM = pangolin::OpenGlMatrix::Translate(0,0,3) * pangolin::OpenGlMatrix::RotateX(-M_PI/2) * pangolin::OpenGlMatrix::RotateY(M_PI) * pangolin::OpenGlMatrix::RotateZ(M_PI/2) ;
    //pangolin::OpenGlMatrix glM = pangolin::OpenGlMatrix::Translate(camera_center[0],camera_center[1],camera_center[2]) *
    //        pangolin::OpenGlMatrix::RotateX(M_PI/8) * pangolin::OpenGlMatrix::RotateZ(-M_PI/2) * pangolin::OpenGlMatrix::RotateY(M_PI/2);
    //pangolin::OpenGlMatrix glM;
    //depth_camera_to_base.getOpenGLMatrix(glM.m);
    Eigen::Matrix4f modelView = Eigen::MatrixXf::Identity(4,4);

    // Create a display renderer
    pangolin::OpenGlRenderState camState(glK_pangolin);
    camState.SetModelViewMatrix(modelView);
    pangolin::View & camDisp = pangolin::Display("cam").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::CreatePanel("pose").SetBounds(0.0,0.9,pangolin::Attach::Pix(-panelWidth), pangolin::Attach::Pix(-2*panelWidth));
    pangolin::Display("multi")
            .SetBounds(0.9, 0.0, pangolin::Attach::Pix(2*panelWidth), pangolin::Attach::Pix(-2*panelWidth))
            .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(camDisp);

    /// ===== Pangolin options
    // Common options
    static pangolin::Var<bool> updateModel("lim.updateModel",false,false);
    pangolin::Var<bool> playVideo("ui.play",false,true);
    pangolin::Var<bool> restartVideo("ui.restartVideo",false,false,true);
    pangolin::Var<bool> stepVideo("ui.stepVideo",false,false,true);
    pangolin::Var<bool> stepVideoBack("ui.stepVideoBack",false,false,true);
    pangolin::Var<bool> doneTracking("ui.doneTracking",false,true); // Close dataset
    pangolin::Var<bool> saveTrackingResults("ui.saveTrackingResults",true,true); // By default, we save tracking results

    pangolin::Var<float> sigmaPixels("ui.sigma pixels",3.0,0.01,4);
    pangolin::Var<float> sigmaDepth("ui.sigma depth",0.1,0.001,1);
    pangolin::Var<bool>  showEstimatedPose("ui.show estimate",true,true);
    pangolin::Var<bool>  showVoxelized("ui.showVoxelized",false,true);
    pangolin::Var<float> levelSet("ui.level set",0.0,-10.0,10.0);
    pangolin::Var<bool>  showObsSdf("ui.showObsSdf",false,true);
    pangolin::Var<bool>  showTrackedPoints("ui.show points",true,true);
    pangolin::Var<int>   pointColoring("ui.pointColoring",nPointColorings-1,0,nPointColorings-1);
    pangolin::Var<bool>  showCollisionClouds("ui.showCollisionClouds",false,true);
    pangolin::Var<bool>  showIntersectionPotentials("ui.showIntersectionPotentials",false,true);
    pangolin::Var<float> planeOffset("ui.plane offset",-1.0, -2.0, 2.0);
    pangolin::Var<float> fps("ui.fps",0);

    // optimization options
    pangolin::Var<bool> continuousOptimization("opt.continuousOptimization",false,true);
    pangolin::Var<bool> iterateButton("opt.iterate",false,false);
    pangolin::Var<int>  itersPerFrame("opt.itersPerFrame",25,0,50);
    pangolin::Var<float> normalThreshold("opt.normalThreshold",-1.01,-1.01,1.0);
    pangolin::Var<float> distanceThreshold("opt.distanceThreshold",0.025,0.0,0.1);
    pangolin::Var<float> regularization("opt.regularization",0.5,0,10);
    pangolin::Var<float> regularizationScaled("opt.regularizationScaled",0.5,0,1);
    pangolin::Var<float> lambdaModToObs("opt.lambdaModToObs",0.03/*0.25*/,0,1);
    pangolin::Var<float> lambdaObsToMod("opt.lambdaObsToMod",1,0,1);
    pangolin::Var<float> lambdaCollision("opt.lambdaCollision",0,0,1);
    pangolin::Var<float> huberDelta("opt.huberDelta",0.01,0.001,0.05);
    pangolin::Var<float> lambdaIntersection("opt.lambdaIntersection",1.f,0,40);

    // Table plane subtraction
    static pangolin::Var<bool> showTablePlane("opt.showTablePlane",true,true);
    static pangolin::Var<bool> subtractTable("opt.subtractTable",true,true);
    static pangolin::Var<float> planeFitNormThresh("opt.planeNormThresh",-1,-1,1);//0.25
    static pangolin::Var<float> planeFitDistThresh("opt.planeDistThresh",0.015,0.0001,0.05);
    static pangolin::Var<bool> subtractBelowTable("opt.subtractBelowTable",true,true); // Remove any points below the table

    // Remove points too close to camera/far away from it
    static pangolin::Var<bool> thresholdPoints("opt.thresholdPoints",true,true);
    static pangolin::Var<float> zNear("opt.nearThreshold",0.1,0,1.0);
    static pangolin::Var<float> zFar("opt.farThreshold",1.5,1.0,10.0);

    // Initialize the models with the recorded data
    static pangolin::Var<bool> initObjPose("opt.initObjPose",true,false,true); // toggle button - initially true
    static pangolin::Var<bool> initBaxterPose("opt.initBaxterPose",true,false,true); // toggle button - initially true
    static pangolin::Var<bool> initBaxterJoints("opt.initBaxterJoints",true,true); // ON/OFF button - initially ON
    static pangolin::Var<int> baxterJointInitInterval("opt.baxterJointInitInterval",1,1,20); // Initialize once this many angles
    static pangolin::Var<bool> initFromSavedDataObj("opt.initFromSavedDataObj",true,true); // Initialize from saved tracking data
    static pangolin::Var<bool> initFromSavedDataBax("opt.initFromSavedDataBax",true,true); // Initialize from saved tracking data

    // SDF parameters
    const static int obsSdfSize = 64;
    const static float obsSdfResolution = 0.005;
    const static float defaultModelSdfResolution = 0.004;
    float defaultModelSdfPadding = 0.07;
    static pangolin::Var<float> modelSdfResolution("lim.modelSdfResolution",defaultModelSdfResolution,defaultModelSdfResolution/2,defaultModelSdfResolution*2);
    static pangolin::Var<float> modelSdfPadding("lim.modelSdfPadding",defaultModelSdfPadding,defaultModelSdfPadding/2,defaultModelSdfPadding*2);

    /// ====== Create a DART tracker and load objects
    dart::Tracker tracker;
    //std::string modelfolder = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/";

    // Load the object model
    tracker.addModel(modelfolder + "objects/" + tabletop_object + ".xml",
                     modelSdfResolution, modelSdfPadding, obsSdfSize,
                     -1, make_float3(0,0,0), 0, 1e5, cache_sdfs);
    int objectID = 0;

    // Load baxter model
    int baxterID = -1;
    std::vector<std::string> model_joint_names;
    std::map<std::string, int> joint_name_to_pose_dim;
    dart::SE3 baxter_base_se3;
    if (track_baxter)
    {
        // Baxter model
        std::string baxter_model_xml;
        if (use_stick)
            baxter_model_xml = modelfolder + "/baxter/baxter_rightarm_wstick_rosmesh.xml";
        else
            baxter_model_xml = modelfolder + "/baxter/baxter_rosmesh_closedgripper.xml";
        // Load baxter model
        tracker.addModel(baxter_model_xml,
                         modelSdfResolution, modelSdfPadding, obsSdfSize,
                         -1, make_float3(0,0,0), 0, 1e5, cache_sdfs); // Add baxter model with SDF resolution = 1 cm
        baxterID = 1;
        tracker.getModel(baxterID).setCameraToModelOptimizeFlag(false); // Do not move the baxter base

        /*
        // Setup intersection matrix (for self-collision check)
        {
            int * selfIntersectionMatrix = dart::loadSelfIntersectionMatrix(modelfolder + "/baxter/baxter_rightarm_wstick_intersection.txt",
                                                                            tracker.getModel(baxterID).getNumSdfs());
            tracker.setIntersectionPotentialMatrix(baxterID, selfIntersectionMatrix);
            delete [] selfIntersectionMatrix;
        }
        */

        // Get the baxter pose and create a map between "frame name" and "pose dimension"
        dart::Pose &baxter_pose(tracker.getPose(baxterID));
        for(int i = 0; i < baxter_pose.getReducedArticulatedDimensions(); i++)
        {
            model_joint_names.push_back(baxter_pose.getReducedName(i));
            joint_name_to_pose_dim[baxter_pose.getReducedName(i)] = i;
        }
        cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID).getReducedArticulatedDimensions() << endl;

        // Baxter pose w.r.t base when loaded
        baxter_base_se3 = tracker.getPose(baxterID).getTransformModelToCamera();
    }

    // Object related sliders (only for controlling pose)
    pangolin::Var<bool> sliderControlled("pose.sliderControlled",false,true);
    pangolin::Var<bool> saveObjectPose("pose.saveObjectPose",false,false,true);
    std::vector<pangolin::Var<float> **> poseVars;
    for (int m = 0; m < tracker.getNumModels(); ++m)
    {
        // Setup vars for the pose of the object
        const int dimensions = tracker.getModel(m).getPoseDimensionality();
        pangolin::Var<float> **vars = new pangolin::Var<float>*[dimensions];
        poseVars.push_back(vars);
        poseVars[m][0] = new pangolin::Var<float>(dart::stringFormat("pose.%d x",m),    -0.095,-0.5,0.5);
        poseVars[m][1] = new pangolin::Var<float>(dart::stringFormat("pose.%d y",m),     0.75,-0.5,0.5);
        poseVars[m][2] = new pangolin::Var<float>(dart::stringFormat("pose.%d z",m),     0.15,-3.0,3.0);
        poseVars[m][3] = new pangolin::Var<float>(dart::stringFormat("pose.%d wx",m),    1.571/*1.2*/,-M_PI,M_PI);
        poseVars[m][4] = new pangolin::Var<float>(dart::stringFormat("pose.%d wy",m),   -1.571/*-1.2*/,-M_PI,M_PI);
        poseVars[m][5] = new pangolin::Var<float>(dart::stringFormat("pose.%d wz",m),    0.53/*1.2*/,-M_PI,M_PI);

        // Set vars for variable joints
        const dart::Model &model = tracker.getModel(m);
        for (int i=6; i<dimensions; i++) {
            poseVars[m][i] = new pangolin::Var<float>(dart::stringFormat("pose.%d %s",m,model.getJointName(i-6).c_str()),0,model.getJointMin(i-6),model.getJointMax(i-6));
        }
    }

    // SDF colors
    dart::MirroredVector<const uchar3 *> allSdfColors(tracker.getNumModels());
    for (int m=0; m<tracker.getNumModels(); ++m) {
        allSdfColors.hostPtr()[m] = tracker.getModel(m).getDeviceSdfColors();
    }
    allSdfColors.syncHostToDevice();

    /// === Get extension of the recorded data files
    std::string extension = ".txt";
    if (doesFileExist(pokefolder + "/positions.csv"))
    {
        extension = ".csv";
    }

    /// ===== Load the joint angles, velocities and efforts for the given joints (along with timestamps)
    // Read the labels in the CSV file and get valid joint names (both recorded and from model)
    std::vector<std::string> recorded_joint_names = read_csv_labels(pokefolder + "/positions" + extension);
    std::vector<std::string> valid_joint_names    = find_common_strings(model_joint_names, recorded_joint_names);
    cout << "Number of valid DOF: " << valid_joint_names.size() << endl;

    // Read joint angles for all joints on the robot and the file
    std::vector<std::vector<double> > joint_positions  = read_csv_data<double>(pokefolder + "/positions" + extension, valid_joint_names);
    cout << "Number of valid joint data from recorded file: " << joint_positions.size() << endl;

    // Read joint velocities (recorded) for all joints on the robot and the file
    std::vector<std::vector<double> > joint_velocities = read_csv_data<double>(pokefolder + "/velocities" + extension, valid_joint_names);
    assert(joint_positions.size() == joint_velocities.size() && "Position and Velocity files do not have same number of rows.");

    // Read joint efforts (recorded) for all joints on the robot and the file
    std::vector<std::vector<double> > joint_efforts = read_csv_data<double>(pokefolder + "/efforts" + extension, valid_joint_names);
    assert(joint_positions.size() == joint_efforts.size() && "Position and Effort files do not have same number of rows.");

    // Get the timestamps from the files
    std::vector<double> joint_timestamps = read_csv_data<double>(pokefolder + "/positions" + extension, "time");
    assert(joint_positions.size() == joint_timestamps.size() && "Position and Timestamps do not have same number of entries.");

    /// ===== Load the commanded joint angles, velocities and efforts for the given joints (along with timestamps)

    // Read the labels in the CSV file and get valid joint names (both recorded and from model)
    std::vector<std::string> recorded_commanded_joint_names = read_csv_labels(pokefolder + "/commandedpositions" + extension);
    std::vector<std::string> valid_commanded_joint_names    = find_common_strings(model_joint_names, recorded_commanded_joint_names);
    cout << "Number of valid commanded DOF: " << valid_commanded_joint_names.size() << endl;

    // Read joint velocities (commanded) for all joints on the robot and the file
    std::vector<std::vector<double> > commanded_joint_positions = read_csv_data<double>(pokefolder + "/commandedpositions" + extension, valid_joint_names);
    cout << "Number of valid commanded joint data from recorded file: " << commanded_joint_positions.size() << endl;

    // Read joint velocities (commanded) for all joints on the robot and the file
    std::vector<std::vector<double> > commanded_joint_velocities = read_csv_data<double>(pokefolder + "/commandedvelocities" + extension, valid_joint_names);
    assert(commanded_joint_positions.size() == commanded_joint_velocities.size() && "Position and Commanded velocity files do not have same number of rows.");

    // Read joint velocities (commanded) for all joints on the robot and the file
    std::vector<std::vector<double> > commanded_joint_accelerations = read_csv_data<double>(pokefolder + "/commandedaccelerations" + extension, valid_joint_names);
    assert(commanded_joint_positions.size() == commanded_joint_accelerations.size() && "Position and Commanded acceleration files do not have same number of rows.");

    // Get the timestamps from the files
    std::vector<double> commanded_joint_timestamps = read_csv_data<double>(pokefolder + "/commandedpositions" + extension, "time");
    assert(commanded_joint_positions.size() == commanded_joint_timestamps.size() && "Position and Timestamps do not have same number of entries.");

    /// ===== Load the recorded end effector poses, twists and wrenches

    // Read end eff poses (no time)
    std::vector<std::string> endeffposelabels = read_csv_labels(pokefolder + "/endeffposes" + extension);
    endeffposelabels.erase(endeffposelabels.begin()); // Remove first label (time)
    std::vector<std::vector<double> > endeff_poses = read_csv_data<double>(pokefolder + "/endeffposes" + extension, endeffposelabels);
    cout << "Number of end effector data from recorded file: " << endeff_poses.size() << endl;

    // Read end eff twists (no time)
    std::vector<std::string> endefftwistlabels = read_csv_labels(pokefolder + "/endefftwists" + extension);
    endefftwistlabels.erase(endefftwistlabels.begin()); // Remove first label (time)
    std::vector<std::vector<double> > endeff_twists = read_csv_data<double>(pokefolder + "/endefftwists" + extension, endefftwistlabels);
    assert(endeff_poses.size() == endeff_twists.size() && "End eff pose and twist files do not have same number of rows.");

    // Read end eff wrenches (no time)
    std::vector<std::string> endeffwrenchlabels = read_csv_labels(pokefolder + "/endeffwrenches" + extension);
    endeffwrenchlabels.erase(endeffwrenchlabels.begin()); // Remove first label (time)
    std::vector<std::vector<double> > endeff_wrenches = read_csv_data<double>(pokefolder + "/endeffwrenches" + extension, endeffwrenchlabels);
    assert(endeff_poses.size() == endeff_wrenches.size() && "End eff pose and wrench files do not have same number of rows.");

    // Read timestamps
    std::vector<double> endefftimestamps = read_csv_data<double>(pokefolder + "/endeffposes" + extension, "time");
    assert(endeff_poses.size() == endefftimestamps.size() && "End eff pose and timestamps do not have same number of entries.");

    /// ===== Load the commanded end effector positions, twists and accelerations

    // Read end eff poses (no time)
    std::vector<std::string> commanded_endeffposelabels = read_csv_labels(pokefolder + "/commandedendeffposes" + extension);
    commanded_endeffposelabels.erase(commanded_endeffposelabels.begin()); // Remove first label (time)
    std::vector<std::vector<double> > commanded_endeff_poses = read_csv_data<double>(pokefolder + "/commandedendeffposes" + extension, commanded_endeffposelabels);
    cout << "Number of commanded end effector data from recorded file: " << commanded_endeff_poses.size() << endl;

    // Read end eff twists (no time)
    std::vector<std::string> commanded_endefftwistlabels = read_csv_labels(pokefolder + "/commandedendefftwists" + extension);
    commanded_endefftwistlabels.erase(commanded_endefftwistlabels.begin()); // Remove first label (time)
    std::vector<std::vector<double> > commanded_endeff_twists = read_csv_data<double>(pokefolder + "/commandedendefftwists" + extension, commanded_endefftwistlabels);
    assert(commanded_endeff_poses.size() == commanded_endeff_twists.size() && "End eff pose and twist files do not have same number of rows.");

    // Read end eff wrenches (no time)
    std::vector<std::string> commanded_endeffaccellabels = read_csv_labels(pokefolder + "/commandedendeffaccelerations" + extension);
    commanded_endeffaccellabels.erase(commanded_endeffaccellabels.begin()); // Remove first label (time)
    std::vector<std::vector<double> > commanded_endeff_accelerations = read_csv_data<double>(pokefolder + "/commandedendeffaccelerations" + extension, commanded_endeffaccellabels);
    assert(commanded_endeff_poses.size() == commanded_endeff_accelerations.size() && "End eff pose and wrench files do not have same number of rows.");

    // Read timestamps
    std::vector<double> commanded_endefftimestamps = read_csv_data<double>(pokefolder + "/commandedendeffposes" + extension, "time");
    assert(commanded_endeff_poses.size() == commanded_endefftimestamps.size() && "End eff pose and timestamps do not have same number of entries.");

    /// ===== Get timestamps for the depth maps
    std::vector<double> depth_timestamps = read_timestamps<double>(pokefolder + "/depthtimestamps.txt");

    // Find joint positions for each depth dataset
    std::vector<int> depth_pos_indices(depth_timestamps.size()), depth_commandedpos_indices(depth_timestamps.size());
    std::vector<int> depth_endeff_indices(depth_timestamps.size()), depth_commandedendeff_indices(depth_timestamps.size());
    for(std::size_t i = 0; i < depth_timestamps.size(); i++)
    {
        depth_pos_indices[i] = find_closest_id(joint_timestamps, depth_timestamps[i]);
        depth_commandedpos_indices[i] = find_closest_id(commanded_joint_timestamps, depth_timestamps[i]);
        depth_endeff_indices[i] = find_closest_id(endefftimestamps, depth_timestamps[i]);
        depth_commandedendeff_indices[i] = find_closest_id(commanded_endefftimestamps, depth_timestamps[i]);
    }

    /// ====== Read the Oriented Bounding Box data - for intializing the objects
    // Get box data
    std::vector<OBB> obbs = read_obb_data(pokefolder + "/obbdata.txt");
    OBB objectobb = obbs[0];

    // Get object pose in camera frame of reference
    Eigen::Vector3f cent_o;
    Eigen::Matrix3f rot_o;
    transformOBBPose(objectobb, base_to_depth_camera, cent_o, rot_o);

    /// ====== Table OBB for plane subtraction, removing points far away from camera and removing points below table
    // Get table OBB and convert to normal-intercept form
    std::vector<OBB> obbs_1 = read_obb_data(pokefolder + "/tableobbdata.txt");
    OBB tableobb = obbs_1[0]; // Get table OBB

    // Get table pose in camera frame of reference
    Eigen::Vector3f cent_t;
    Eigen::Matrix3f rot_t;
    transformOBBPose(tableobb, base_to_depth_camera, cent_t, rot_t);
    Eigen::Vector3f hext_t = tableobb.halfextents;

    /// ====== Initialize an image depth source
    dart::ImageDepthSource<ushort,uchar3> *depthSource = new dart::ImageDepthSource<ushort,uchar3>();
    depthSource->initialize(pokefolder+"/depth%d.png",make_float2(glFLx, glFLy), make_float2(glPPx, glPPy),
                           glWidth, glHeight, 1e-3, 0, false);
    int numframes = depthSource->getNumDepthFrames(); // number of depth frames
    tracker.addDepthSource(depthSource);
    tracker.setFrameOnGPU(0);

    // Load training frame data
    std::ifstream savedTrainingFrames(pokefolder + "/trainingframes.txt");
    int savedStartFrame=0, savedEndFrame=numframes-1;
    if (savedTrainingFrames.is_open())
    {
        savedTrainingFrames >> savedStartFrame;
        savedTrainingFrames >> savedEndFrame;
        savedTrainingFrames.close();
    }

    // Slider for moving around images
    static pangolin::Var<int> frameNumberSlider("data.frameNumber",0,0,numframes-1); // Initialize once this many angles
    static pangolin::Var<int> startFrame("data.startFrame",savedStartFrame,0,numframes-1); // Start of dataset used for NN
    static pangolin::Var<int> endFrame("data.endFrame",savedEndFrame,0,numframes-1); // End of dataset used for NN training

    // Color point cloud
    uchar3 * hDepthColor;
    cudaMallocHost(&hDepthColor,glWidth*glHeight*sizeof(uchar3));

    // set up VBO to display point cloud
    int bufferSize = glWidth*glHeight;
    GLuint pointCloudVbo,pointCloudColorVbo,pointCloudNormVbo;
    glGenBuffersARB(1,&pointCloudVbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudVbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB,bufferSize*sizeof(float4),tracker.getHostVertMap(),GL_DYNAMIC_DRAW_ARB);
    glGenBuffersARB(1,&pointCloudColorVbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB,bufferSize*sizeof(uchar3),hDepthColor,GL_DYNAMIC_DRAW_ARB);
    glGenBuffersARB(1,&pointCloudNormVbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudNormVbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB,bufferSize*sizeof(float4),tracker.getHostNormMap(),GL_DYNAMIC_DRAW_ARB);

    // Create stuff for the tracking
    std::vector<cv::Mat> dataAssMaps(numframes), pointClouds(numframes);
    std::vector<std::vector<std::vector<float> > > tracked_joint_angles(numframes);
    std::vector<std::vector<std::vector<dart::SE3> > > tracked_transforms(numframes);

    // Saved data from previous results
    std::vector<std::vector<std::vector<float> > > saved_tracked_joint_angles;
    std::vector<std::vector<std::vector<dart::SE3> > > saved_tracked_transforms;
    std::ifstream ifs(pokefolder + "/darttrackerdata.txt");
    if (ifs.is_open())
    {
        printf("Loading saved DART tracker data \n");
        boost::archive::text_iarchive ia(ifs);
        ia >> saved_tracked_joint_angles;
        //ia >> saved_tracked_transforms;
        ifs.close();
    }
    else
    {
        printf("Could not find any saved DART tracker data. Will track afresh \n");
    }

    /// ====== Tracking loop
    for (int pangolinFrame=1; !pangolin::ShouldQuit() && !pangolin::Pushed(doneTracking); ++pangolinFrame)
    {
        // ==== Get DART tracker options
        dart::OptimizationOptions & opts = tracker.getOptions();
        opts.focalLength = glFLx; // Assuming fx = fy
        opts.normThreshold = normalThreshold;
        //opts.distThreshold[0] = distanceThreshold;
        for (int m=0; m<tracker.getNumModels(); ++m)
        {
            opts.distThreshold[m] = distanceThreshold;
            opts.regularization[m] = regularization;
            opts.regularizationScaled[m] = regularizationScaled;
            opts.planeOffset[m] = planeOffset;
            opts.planeNormal[m] = make_float3(0,1,0);//make_float3(0,0,1);
        }
        opts.lambdaModToObs = lambdaModToObs;
        opts.lambdaObsToMod = lambdaObsToMod;
        opts.debugObsToModDA = true;//pointColoring == pointColoringDA;
        opts.debugObsToModErr = pointColoring == pointColoringErr;
        opts.numIterations = itersPerFrame;
        opts.huberDelta = huberDelta;

        //opts.lambdaIntersection[0] = lambdaCollision;
        //memset(opts.lambdaIntersection.data(),0,tracker.getNumModels()*tracker.getNumModels()*sizeof(float));
        //opts.lambdaIntersection[0 + 2*0] = lambdaIntersection; // left
        //opts.lambdaIntersection[1 + 2*1] = lambdaIntersection; // right

        // In case we are at the last frame, set play to false
        bool play = playVideo;
        if(depthSource->getFrame() == numframes-1)
            play = false;

        // ==== Set frame number based on slider
        if (frameNumberSlider != depthSource->getFrame())
            tracker.setFrameOnGPU(frameNumberSlider);

        // ==== Restart video (TODO)
        if (pangolin::Pushed(restartVideo))
        {
            frameNumberSlider = 0; // Reset frame number slider
            tracker.setFrameOnGPU(0); // Go back to start
            play = false; // Pause at the start
        }

        // ==== In case we reach the end of the video (or we are asked to restart),
        // we need to initialize all models to their default values
        //bool resetObjects = (depthSource->getFrame() == 0);

        // ==== Update SDF if resolution has been changed
        if (pangolin::Pushed(updateModel)) {
            for(int m = 0; m < tracker.getNumModels(); ++m)
                tracker.updateModel(m, modelSdfResolution, modelSdfPadding,
                                    obsSdfSize, obsSdfResolution, make_float3(0,0,0));
        }

        // ==== Other options
        if (pangolin::Pushed(stepVideoBack)) {
            tracker.stepBackward();
            frameNumberSlider = (frameNumberSlider > 0) ? (frameNumberSlider-1) : 0;
        }
        bool iteratePushed = pangolin::Pushed(iterateButton);
        int currframeid = depthSource->getFrame();

        // ==== Initialize the models with the recorded data
        if (pangolin::Pushed(initObjPose))
        {
            // Get recorded pose from disk
            std::ifstream recfile(pokefolder + "/dartrecobbinit.txt");
            if (recfile.is_open())
            {
                // Get SE3 from file
                dart::SE3 modelToCam;
                boost::archive::text_iarchive ia(recfile);
                ia >> modelToCam;
                recfile.close();

                // Initialize object with the OBB pose from file
                tracker.getPose(objectID).setTransformModelToCamera(modelToCam);
            }
            else
            {
                // Initialize object with the OBB pose from python
                tracker.getPose(objectID).setTransformModelToCamera(SE3fromEigen(cent_o, rot_o));
            }

            // Update pose
            tracker.updatePose(objectID);

            // Set slider values (translation)
            float3 t = dart::translationFromSE3(tracker.getPose(objectID).getTransformModelToCamera());
            *poseVars[objectID][0] =  t.x;
            *poseVars[objectID][1] =  t.y;
            *poseVars[objectID][2] =  t.z;

            // Set slider values (rotation)
            dart::se3 se3 = dart::se3FromSE3(tracker.getPose(objectID).getTransformModelToCamera()); // We can't use translation from this as it is not in the proper reference frame
            *poseVars[objectID][3] =  se3.p[3];
            *poseVars[objectID][4] =  se3.p[4];
            *poseVars[objectID][5] =  se3.p[5];
        }

        // Initialize baxter pose to default values
        if (track_baxter && (pangolin::Pushed(initBaxterPose)))
        {
            tracker.getPose(baxterID).setTransformModelToCamera(base_to_depth_camera_se3 * baxter_base_se3);
            tracker.updatePose(baxterID);

            // Set slider values (translation)
            float3 t = dart::translationFromSE3(tracker.getPose(baxterID).getTransformModelToCamera());
            *poseVars[baxterID][0] =  t.x;
            *poseVars[baxterID][1] =  t.y;
            *poseVars[baxterID][2] =  t.z;

            // Set slider values (rotation)
            dart::se3 se3 = dart::se3FromSE3(tracker.getPose(baxterID).getTransformModelToCamera()); // We can't use translation from this as it is not in the proper reference frame
            *poseVars[baxterID][3] =  se3.p[3];
            *poseVars[baxterID][4] =  se3.p[4];
            *poseVars[baxterID][5] =  se3.p[5];
        }

        // Initialize the baxter joints based on the closest joint angles (w.r.t the current depth frame)
        // Initialize only "x" number of iterations
        if (track_baxter && ((initBaxterJoints && ((currframeid % baxterJointInitInterval) == 0))))
        {
            // Get current baxter frame joint angles
            std::vector<double> curr_angles = joint_positions[depth_pos_indices[depthSource->getFrame()]];

            // Update the arm pose
            for (int i = 6; i<tracker.getPose(baxterID).getReducedDimensions(); ++i)
            {
                tracker.getPose(baxterID).getReducedArticulation()[i-6] = curr_angles[i-6];
            }
            tracker.updatePose(baxterID);

            // Set slider values for baxter joints
            for (int i = 6; i < tracker.getPose(baxterID).getReducedDimensions(); ++i)
            {
                *poseVars[baxterID][i] = curr_angles[i-6]; // Recorded joint angles closest to current depth map
            }
        }

        // update pose based on sliders
        if (sliderControlled)
        {
            for (int m=0; m<tracker.getNumModels(); ++m)
            {
                // Update joint angles
                for (int i = 6; i<tracker.getPose(m).getReducedDimensions(); ++i)
                {
                    tracker.getPose(m).getReducedArticulation()[i-6] = *poseVars[m][i];
                }

                // Update SE3 base pose
                tracker.getPose(m).setTransformModelToCamera(dart::SE3Fromse3(dart::se3(*poseVars[m][0],*poseVars[m][1],*poseVars[m][2],0,0,0))*
                        dart::SE3Fromse3(dart::se3(0,0,0,*poseVars[m][3],*poseVars[m][4],*poseVars[m][5])));
                tracker.updatePose(m);
            }
        }

        // Save initialized pose
        if (pangolin::Pushed(saveObjectPose))
        {
            std::ofstream file(pokefolder + "/dartrecobbinit.txt");
            boost::archive::text_oarchive oa(file);
            oa << tracker.getPose(objectID).getTransformModelToCamera();
            file.close();
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                                                                                      //
        // Process this frame                                                                                   //
        //                                                                                                      //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        {
            // get the latest depth map
            static pangolin::Var<bool> filteredNorms("ui.filteredNorms",true,true);
            static pangolin::Var<bool> filteredVerts("ui.filteredVerts",false,true);

            // Set stuff
            if (filteredNorms.GuiChanged()) {
                tracker.setFilteredNorms(filteredNorms);
            } else if (filteredVerts.GuiChanged()) {
                tracker.setFilteredVerts(filteredVerts);
            } else if (sigmaDepth.GuiChanged()) {
                tracker.setSigmaDepth(sigmaDepth);
            } else if (sigmaPixels.GuiChanged()) {
                tracker.setSigmaPixels(sigmaPixels);
            }

            /// === Filter points - table subtraction, box removal and removing far away points
            // Remove points below table
            if (subtractTable) {
                tracker.getPointCloudSource().eliminatePlane(make_float3(rot_t.col(2)), cent_t.dot(rot_t.col(2)),
                                                             planeFitDistThresh, planeFitNormThresh); // values!
            }

            // Remove points too far away from camera
            if (thresholdPoints){
                tracker.getPointCloudSource().cropBox(make_float3(-2,-2,zNear), make_float3(2,2,zFar));
            }

            // Use saved data if available to initialize the tracker
            if (initFromSavedDataObj && saved_tracked_joint_angles.size() > 0)
            {
                // Update SE3 base pose
                tracker.getPose(objectID).setTransformModelToCamera(dart::SE3Fromse3(dart::se3(saved_tracked_joint_angles[currframeid][objectID][0],
                                                                                               saved_tracked_joint_angles[currframeid][objectID][1],
                                                                                               saved_tracked_joint_angles[currframeid][objectID][2], 0, 0, 0))*
                                                                    dart::SE3Fromse3(dart::se3(0, 0, 0, saved_tracked_joint_angles[currframeid][objectID][3],
                                                                                               saved_tracked_joint_angles[currframeid][objectID][4],
                                                                                               saved_tracked_joint_angles[currframeid][objectID][5])));
            }

            // Use saved data if available to initialize the tracker
            if (initFromSavedDataBax && saved_tracked_joint_angles.size() > 0)
            {
                // Update joint angles
                for (int i = 6; i<tracker.getPose(baxterID).getReducedDimensions(); ++i)
                {
                    tracker.getPose(baxterID).getReducedArticulation()[i-6] = saved_tracked_joint_angles[currframeid][baxterID][i];
                }

                // Update SE3 base pose
                tracker.getPose(baxterID).setTransformModelToCamera(dart::SE3Fromse3(dart::se3(saved_tracked_joint_angles[currframeid][baxterID][0],
                                                                                        saved_tracked_joint_angles[currframeid][baxterID][1],
                                                                                        saved_tracked_joint_angles[currframeid][baxterID][2], 0, 0, 0))*
                                                                    dart::SE3Fromse3(dart::se3(0, 0, 0, saved_tracked_joint_angles[currframeid][baxterID][3],
                                                                                        saved_tracked_joint_angles[currframeid][baxterID][4],
                                                                                        saved_tracked_joint_angles[currframeid][baxterID][5])));
            }

            // Optimize model poses (initialized with the saved tracking data)
            if (continuousOptimization || iteratePushed )
            {
                tracker.optimizePoses();
            }

            // Update model poses
            for (int m=0; m<tracker.getNumModels(); ++m)
                tracker.updatePose(m);

            // Update pose/joint angles on GUI
            for (int m=0; m<tracker.getNumModels(); ++m)
            {
                // Update SE3 to base
                float3 t = dart::translationFromSE3(tracker.getPose(m).getTransformModelToCamera());
                *poseVars[m][0] =  t.x;
                *poseVars[m][1] =  t.y;
                *poseVars[m][2] =  t.z;

                // Set slider values (rotation)
                dart::se3 se3 = dart::se3FromSE3(tracker.getPose(m).getTransformModelToCamera()); // We can't use translation from this as it is not in the proper reference frame
                *poseVars[m][3] =  se3.p[3];
                *poseVars[m][4] =  se3.p[4];
                *poseVars[m][5] =  se3.p[5];

                // Update articulation values
                for (int i=6; i<tracker.getPose(m).getReducedDimensions(); ++i)
                {
                    *poseVars[m][i] = tracker.getPose(m).getReducedArticulation()[i-6];
                }
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                                                                                      //
        // Render this frame                                                                                    //
        //                                                                                                      //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Resizing
        if(pangolin::HasResized()) {
            pangolin::DisplayBase().ActivateScissorAndClear();
        }

        // Setup lighting
        glShadeModel (GL_SMOOTH);
        float4 lightPosition = make_float4(normalize(make_float3(-0.4405,-0.5357,-0.619)),0);
        GLfloat light_ambient[] = { 0.3, 0.3, 0.3, 1.0 };
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
        glLightfv(GL_LIGHT0, GL_POSITION, (float*)&lightPosition);

        // Enable OpenGL drawing
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

        // Color & material
        glColor4ub(0xff,0xff,0xff,0xff);
        glEnable(GL_COLOR_MATERIAL);

        // ==== Render models
        for (int m=0; m<tracker.getNumModels(); ++m)
        {
            // Do overall rendering for each model
            tracker.updatePose(m);
            tracker.getModel(m).render();
            tracker.getModel(m).renderCoordinateFrames();
            tracker.getModel(m).renderSkeleton();

            // Show obs sdf
            if(showObsSdf)
            {
                glPushMatrix();
                dart::glMultSE3(tracker.getModel(m).getTransformModelToCamera());
                tracker.getModel(m).syncObsSdfDeviceToHost();
                dart::Grid3D<float> * obsSdf = tracker.getModel(m).getObsSdf();
                tracker.getModel(m).renderSdf(*obsSdf,levelSet);
                glPopMatrix();
            }
        }

        // ==== Show voxelized

        if (showVoxelized) {
            glColor3f(0.2,0.3,1.0);
            for (int m=0; m<tracker.getNumModels(); ++m)
            {
                tracker.updatePose(m);
                tracker.getModel(m).renderVoxels(levelSet);
            }
        }

        // ==== Show table
        if (showTablePlane)
        {
            // Get the corners of the table
            std::vector<Eigen::Vector3f> pts = { cent_t - hext_t(1) * rot_t.col(0) - hext_t(0) * rot_t.col(1),
                                                 cent_t + hext_t(1) * rot_t.col(0) - hext_t(0) * rot_t.col(1),
                                                 cent_t + hext_t(1) * rot_t.col(0) + hext_t(0) * rot_t.col(1),
                                                 cent_t - hext_t(1) * rot_t.col(0) + hext_t(0) * rot_t.col(1) };
            // Display
            glColor3ub(120,100,100);
            glBegin(GL_QUADS);
            glNormal3f(rot_t(0,2), rot_t(1,2), rot_t(2,2));
            for (int i=0; i<pts.size(); ++i) {
                glVertex3f(pts[i](0), pts[i](1), pts[i](2));
            }
            glEnd();
        }

        // ==== Show tracked points
        glPointSize(1.0f);
        if (showTrackedPoints)
        {
            // Setup OpenGL stuff
            glPointSize(4.0f);
            glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudVbo);
            glBufferDataARB(GL_ARRAY_BUFFER_ARB,glWidth*glHeight*sizeof(float4),tracker.getHostVertMap(),GL_DYNAMIC_DRAW_ARB);
            glEnableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_NORMAL_ARRAY);
            glVertexPointer(4, GL_FLOAT, 0, 0);

            // Display based on type of point coloring
            switch (pointColoring)
            {
                case pointColoringNone:
                    glColor3f(0.25,0.25,0.25);
                    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudNormVbo);
                    glBufferDataARB(GL_ARRAY_BUFFER_ARB,glWidth*glHeight*sizeof(float4),tracker.getHostNormMap(),GL_DYNAMIC_DRAW_ARB);
                    glNormalPointer(GL_FLOAT, 4*sizeof(float), 0);
                    glEnableClientState(GL_NORMAL_ARRAY);
                    break;
                case pointColoringErr:
                {
                    static pangolin::Var<float> errorMin("ui.errorMinMod",0.0,0.0,0.5);
                    static pangolin::Var<float> errorMax("ui.errorMaxMod",0.000119,0.0,0.01);
                    dart::MirroredVector<uchar3> errColored(glWidth*glHeight);
                    float * dErr;
                    cudaMalloc(&dErr,glWidth*glHeight*sizeof(float));
                    dart::imageSquare(dErr,tracker.getDeviceDebugErrorObsToMod(),glWidth,glHeight);
                    dart::colorRampHeatMap(errColored.devicePtr(),dErr,glWidth,glHeight,errorMin,errorMax);
                    cudaFree(dErr);
                    errColored.syncDeviceToHost();
                    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
                    glBufferDataARB(GL_ARRAY_BUFFER_ARB,glWidth*glHeight*sizeof(uchar3),errColored.hostPtr(),GL_DYNAMIC_DRAW_ARB);
                    glColorPointer(3,GL_UNSIGNED_BYTE, 0, 0);
                    glEnableClientState(GL_COLOR_ARRAY);
                    glDisable(GL_LIGHTING);
                }
                    break;
                case pointColoringDA:
                {
                    const int *dDebugDA = tracker.getDeviceDebugDataAssociationObsToMod();
                    dart::MirroredVector<uchar3> daColored(glWidth*glHeight);
                    dart::colorDataAssociationMultiModel(daColored.devicePtr(),dDebugDA,allSdfColors.devicePtr(),glWidth,glHeight);
                    daColored.syncDeviceToHost();
                    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
                    glBufferDataARB(GL_ARRAY_BUFFER_ARB,glWidth*glHeight*sizeof(uchar3),daColored.hostPtr(),GL_DYNAMIC_DRAW_ARB);
                    glColorPointer(3,GL_UNSIGNED_BYTE, 0, 0);
                    glEnableClientState(GL_COLOR_ARRAY);
                    glDisable(GL_LIGHTING);
                }
                    break;
            }

            // Draw points
            glDrawArrays(GL_POINTS,0,glWidth*glHeight);
            glBindBuffer(GL_ARRAY_BUFFER_ARB,0);

            // Reset OpenGL stuff
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_NORMAL_ARRAY);
            glPointSize(1.0f);
        }

        // ==== Show collision clouds
        if (showCollisionClouds)
        {
            // Setup OpenGL stuff
            glPointSize(10);
            glColor3f(0,0,1.0f);
            glDisable(GL_LIGHTING);
            glBegin(GL_POINTS);

            // Iterate over models to display collision clouds
            for (int m=0; m<tracker.getNumModels(); ++m)
            {
                const float4 * collisionCloud = tracker.getCollisionCloud(m);
                for (int i=0; i<tracker.getCollisionCloudSize(m); ++i)
                {
                    int grid = round(collisionCloud[i].w);
                    int frame = tracker.getModel(m).getSdfFrameNumber(grid);
                    float4 v = tracker.getModel(m).getTransformModelToCamera()*
                               tracker.getModel(m).getTransformFrameToModel(frame)*
                               make_float4(make_float3(collisionCloud[i]),1.0);
                    glVertex3fv((float *)&v);
                }
            }

            // Reset OpenGL stuff
            glEnd();
            glEnable(GL_LIGHTING);
            glPointSize(1.0f);
            glColor3f(1,1,1);
        }

        // ==== Show intersection potentials
        if (showIntersectionPotentials)
        {
            // Get baxter model and compute potential intersections
            const dart::MirroredModel& model = tracker.getModel(0);
            const int nSdfs = model.getNumSdfs();
            static pangolin::Var<int> intersectingSdf("ui.intersectingSdf",0,0,nSdfs-1);
            const int* potentialIntersections = &tracker.getIntersectionPotentialMatrix(0)[intersectingSdf*nSdfs];
            const float4 intersectingSdfOrigin = model.getTransformModelToCamera()*
                    model.getTransformFrameToModel(model.getSdfFrameNumber(intersectingSdf))*make_float4(0,0,0,1);

            // Show intersections
            glDisable(GL_LIGHTING);
            glDisable(GL_DEPTH_TEST);
            glBegin(GL_LINES);
            glColor3f(1,0,0);
            for (int i=0; i<nSdfs; ++i)
            {
                if (potentialIntersections[i])
                {
                    const float4 origin = model.getTransformModelToCamera()*
                            model.getTransformFrameToModel(model.getSdfFrameNumber(i))*make_float4(0,0,0,1);
                    glVertex3fv((float*)&intersectingSdfOrigin);
                    glVertex3fv((float*)&origin);

                }
            }
            glEnd();
            glEnable(GL_LIGHTING);
            glEnable(GL_DEPTH_TEST);
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                                                                                      //
        // Save tracked data - DA, Tracked poses, Camera to model data                                          //
        //                                                                                                      //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        // ====== Get the depth map
        unsigned int frameid = depthSource->getFrame();
        assert(frameid == currframeid);
        const float4* ptcloud = tracker.getHostVertMap();
        cv::Mat pointCloud(glHeight, glWidth, CV_32FC3, cv::Scalar(0));
        for(int j = 0; j < glWidth; j++)
            for(int k = 0; k < glHeight; k++)
            {
                int idx = j + k * glWidth;
                if(ptcloud[j+k*glWidth].w != 0)
                    pointCloud.at<cv::Vec3f>(k,j) = cv::Vec3f(ptcloud[idx].x, ptcloud[idx].y, ptcloud[idx].z);
            }
        pointClouds[frameid] = pointCloud; // Save for later
//        cv::imshow("Depth", getChannel(pointClouds[frameid], 2, true));

        // ===== Get data associations
        // Display data associations between models and depth data
        const int *dDebugDA = tracker.getDeviceDebugDataAssociationObsToMod(); // Get device memory
//        dart::MirroredVector<uchar3> daColored(glWidth*glHeight);
//        dart::colorDataAssociationMultiModel(daColored.devicePtr(),dDebugDA,allSdfColors.devicePtr(),glWidth,glHeight);
//        daColored.syncDeviceToHost();

//        // Display the masks
//        cv::Mat maskImg(glHeight, glWidth, CV_8UC3, daColored.hostPtr());
//        cv::imshow("Mask", maskImg);

        // Get the model & SDF indices
        dart::MirroredVector<int2> modelSDFIndices(glWidth*glHeight);
        dart::getIndicesFromDataAssociationMultiModel(modelSDFIndices.devicePtr(),dDebugDA,glWidth,glHeight);
        modelSDFIndices.syncDeviceToHost();

        // Display the indices
        cv::Mat modelSDFIndicesImg(glHeight, glWidth, CV_32SC2, modelSDFIndices.hostPtr());
        dataAssMaps[frameid] = modelSDFIndicesImg.clone(); // Data association for each depth point
        cv::imshow("ModelID", getChannel(modelSDFIndicesImg, 0, true));
        cv::imshow("SdfID", getChannel(modelSDFIndicesImg, 1, true));
        cv::waitKey(2);

        // ====== Get the mesh transforms
        tracked_transforms[frameid].resize(tracker.getNumModels());
        for(int m = 0; m < tracker.getNumModels(); m++)
        {
            int mNSdfs = tracker.getModel(m).getNumSdfs();
            tracked_transforms[frameid][m].resize(mNSdfs);
            for (int g = 0; g < mNSdfs; g++)
            {
                const int f = tracker.getModel(m).getSdfFrameNumber(g); // Frame number for SDF
                dart::SE3 tfm = tracker.getModel(m).getTransformCameraToFrame(f);
                tracked_transforms[frameid][m][g] = dart::SE3(tfm.r0, tfm.r1, tfm.r2); // Frame to camera
            }
        }

        // ====== Get tracker data
        tracked_joint_angles[frameid].resize(tracker.getNumModels());
        for (int m = 0; m < tracker.getNumModels(); ++m)
        {
            // Resize vector
            tracked_joint_angles[frameid][m].resize(tracker.getPose(m).getDimensions());

            // Update SE3 to base
            float3 t = dart::translationFromSE3(tracker.getPose(m).getTransformModelToCamera());
            tracked_joint_angles[frameid][m][0] =  t.x;
            tracked_joint_angles[frameid][m][1] =  t.y;
            tracked_joint_angles[frameid][m][2] =  t.z;

            // Set slider values (rotation)
            dart::se3 se3 = dart::se3FromSE3(tracker.getPose(m).getTransformModelToCamera()); // We can't use translation from this as it is not in the proper reference frame
            tracked_joint_angles[frameid][m][3] =  se3.p[3];
            tracked_joint_angles[frameid][m][4] =  se3.p[4];
            tracked_joint_angles[frameid][m][5] =  se3.p[5];

            // Update articulation values
            for (int i = 6; i < tracker.getPose(m).getReducedDimensions(); ++i)
            {
                tracked_joint_angles[frameid][m][i] = tracker.getPose(m).getReducedArticulation()[i-6];
            }
        }

        // ==== Go to next image
        if (depthSource->isLive() || pangolin::Pushed(stepVideo) || play) {
            frameNumberSlider = (depthSource->getFrame() == numframes-1) ? 0 : frameNumberSlider+1; // Reset at limits
            tracker.stepForward();
        }

        /////////////////
        glPopMatrix();
        glDisable(GL_LIGHTING);
        glColor4ub(255,255,255,255);

        /// Finish frame
        pangolin::FinishFrame();
        usleep(10000);
    }

    // ==== Free memory
    cudaFreeHost(hDepthColor);
    glDeleteBuffersARB(1,&pointCloudVbo);
    glDeleteBuffersARB(1,&pointCloudColorVbo);
    glDeleteBuffersARB(1,&pointCloudNormVbo);
    delete(depthSource);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                                                                                      //
    // Save to disk                                                                                         //
    //                                                                                                      //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (saveTrackingResults)
    {
        /// == Save tracker pose data (use boost serialization)
        printf("Saving tracker data to disk \n");
        {
            std::ofstream ofs(pokefolder + "/darttrackerdata.txt");
            boost::archive::text_oarchive oa(ofs);
            oa << tracked_joint_angles;
            oa << tracked_transforms;
            ofs.close();
        }

        /// ===================== Write labels to disk ==================== ///
        // == Save full res data
        printf("Saving generate data association labels to disk \n");
        for(int counter = 0; counter < numframes; counter++)
        {
            // Create new image (0,0 is BG label, everything else is 1-indexed)
            cv::Mat labels(glHeight, glWidth, CV_8UC3);
            for(int j = 0; j < glWidth; j++)
                for(int k = 0; k < glHeight; k++)
                    labels.at<cv::Vec3b>(k,j) = cv::Vec3b(dataAssMaps[counter].at<cv::Vec2i>(k,j)[0]+1,
                                                          dataAssMaps[counter].at<cv::Vec2i>(k,j)[1]+1,
                                                          0);
            // Save full res data
            if(save_fullres_data)
            {
                // Save label image
                cv::imwrite(pokefolder + "/labels" + std::to_string(counter) + ".png", labels); // Save labels
            }

            // Subsample label image and save it
            cv::Mat labels_sub(std::round(0.5 * glHeight), std::round(0.5 * glWidth), CV_8UC3);
            cv::resize(labels, labels_sub, cv::Size(), 0.5, 0.5, CV_INTER_NN); // Do NN interpolation
            cv::imwrite(pokefolder + "/labelssub" + std::to_string(counter) + ".png", labels_sub); // Save labels
        }

        /// ===== PCL viewer
        boost::shared_ptr<boost::thread> pcl_viewer_thread;
        if(visualize)
        {
            printf("Visualizing the point clouds and flow vectors \n");
            point_cloud_update = false; // Do not update point clouds yet
            pcl_viewer_thread.reset(new boost::thread(pclViewerThread));
        }

        /// ====================== Compute flow =================== ///
        /// == Compute flows for each of the steps and save the data to disk
        // Iterate over the different requested "step" lengths and compute flow for each
        // Each of these is stored in their own folder - "flow_k" where "k" is the step-length
        int randStep = floor(randDouble() * step_list.size());
        printf("Saving flow data to disk. Displaying flow for step: %d \n",step_list[randStep]);
        for(std::size_t k = 0; k < step_list.size(); k++)
        {
            // Get step size and create flow dir
            int step = step_list[k];
            std::string flow_dir = pokefolder + "/flow_" + std::to_string(step) + "/";
            createDirectory(flow_dir);
            printf("Computing flows for step length: %d \n", step);

            // Iterate over depth images to compute flow for each relevant one
            std::vector<cv::Mat> flowMaps(numframes); // used for visualizing data
            for(int counter = 0; counter < numframes-step; ++counter)
            {
                // Compute flow multi-threaded
                tg.addTask([&,counter]() // Lambda function/thread
                {
                    // Compute flow between frames @ t & t+step
                    flowMaps[counter] = compute_flow_between_frames(pointClouds[counter],
                                                                    dataAssMaps[counter],
                                                                    tracked_transforms[counter],
                                                                    tracked_transforms[counter+step],
                                                                    glWidth, glHeight);

                    // Convert the flow data from "m" to "0.1 mm" range and store it as a 16-bit unsigned short
                    // 2^16 = 65536. 2^16/2 ~ 32767 * 1e-4 (0.1mm) = +-3.2767 (since we have positive and negative values)
                    // We can represent motion ranges from -3.2767 m to +3.2767 m using this representation
                    // This amounts to ~300 cm/frame ~ 90m/s speed (which should be way more than enough for the motions we currently have)
                    if(save_fullres_data)
                    {
                        cv::Mat flowMap_u = convert_32FC3_to_16UC3(flowMaps[counter], M_TO_MM);
                        cv::imwrite(flow_dir + "flow" + std::to_string(counter) + ".png", flowMap_u);
                    }

                    // Save subsampled image by NN interpolation
                    // KEY: Don't do cubic/bilinear interpolation as it leads to interpolation of flow values across boundaries which is incorrect
                    cv::Mat flowMap_sub(round(0.5 * glHeight), round(0.5 * glWidth), CV_32FC3);
                    cv::resize(flowMaps[counter], flowMap_sub, cv::Size(), 0.5, 0.5, CV_INTER_NN); // Do NN interpolation (otherwise flow gets interpolated, which is incorrect)

                    // Scale the flow image data as a 16-bit 3-channel ushort image & save it
                    cv::Mat flowMap_u_sub = convert_32FC3_to_16UC3(flowMap_sub, M_TO_MM);
                    cv::imwrite(flow_dir + "flowsub" + std::to_string(counter) + ".png", flowMap_u_sub);
                });
            }
            tg.wait();

            // In case we are asked to visualize the flow outputs and/or images do it here
            if(visualize && (k == randStep))
            {
                // Iterate over all the depth images
                for(int counter = 0; counter < numframes-step; ++counter)
                {
                    // ==== Image stuff
                    cv::imshow("Subsampled Colorized Depth", colorize_depth(getChannel(pointClouds[counter], 2)));
                    cv::imshow("Labels-Model", colorize_depth(getChannel(dataAssMaps[counter], 0)));
                    cv::imshow("Labels-SDFs", colorize_depth(getChannel(dataAssMaps[counter], 1)));
                    cv::imshow("Flow", colorize_depth(flowMaps[counter]));
                    cv::waitKey(2);      // wait for 2 ms

                    // ==== Flow stuff
                    // Get lock on the mutex
                    boost::mutex::scoped_lock update_lock(point_cloud_mutex);

                    // Update point cloud and flow vector
                    point_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>(glWidth, glHeight)); // Set point cloud for viewing
                    flow_cloud.reset(new pcl::PointCloud<pcl::Normal>(glWidth, glHeight)); // Update flow cloud
                    for (std::size_t r = 0; r < point_cloud->height; r++) // copy over from the flow matrix
                    {
                        for (std::size_t c = 0; c < point_cloud->width; c++)
                        {
                            point_cloud->at(c,r) = pcl::PointXYZ(pointClouds[counter].at<cv::Vec3f>(r,c)[0],
                                                                 pointClouds[counter].at<cv::Vec3f>(r,c)[1],
                                                                 pointClouds[counter].at<cv::Vec3f>(r,c)[2]);
                            flow_cloud->at(c,r) = pcl::Normal(flowMaps[counter].at<cv::Vec3f>(r,c)[0],
                                                              flowMaps[counter].at<cv::Vec3f>(r,c)[1],
                                                              flowMaps[counter].at<cv::Vec3f>(r,c)[2]);
                        }
                    }

                    // Set flag for visualizer
                    point_cloud_update = true;
                    update_lock.unlock(); // Unlock mutex so that display happens
                }
            }
        }

        // Shutdown PCL viewer thread
        if(visualize)
        {
            pcl_viewer_terminate = true; // set it to exit at the end
            pcl_viewer_thread->join();
        }

        /// == Save the joint states, velocities and efforts (recorded - closest to depth maps)
        printf("Saving recorded joint states to disk \n");
        for(int counter = 0; counter < numframes; counter++)
        {
            // Create a new file per state
            std::ofstream statefile(pokefolder + "/state" + std::to_string(counter) + ".txt");

            // Write joint states, velocities and efforts
            writeVector(statefile, joint_positions[depth_pos_indices[counter]]);
            writeVector(statefile, joint_velocities[depth_pos_indices[counter]]);
            writeVector(statefile, joint_efforts[depth_pos_indices[counter]]);

            // Write end eff pose, twist, wrenches
            writeVector(statefile, endeff_poses[depth_endeff_indices[counter]]);
            writeVector(statefile, endeff_twists[depth_endeff_indices[counter]]);
            writeVector(statefile, endeff_wrenches[depth_endeff_indices[counter]]);

            // Close file
            statefile.close();
        }

        /// == Save the joint states, velocities and efforts (commanded - closest to depth maps)
        printf("Saving commanded joint states to disk \n");
        for(int counter = 0; counter < numframes; counter++)
        {
            // Create a new file per state
            std::ofstream statefile(pokefolder + "/commandedstate" + std::to_string(counter) + ".txt");

            // Write joint states, velocities and efforts
            writeVector(statefile, commanded_joint_positions[depth_commandedpos_indices[counter]]);
            writeVector(statefile, commanded_joint_velocities[depth_commandedpos_indices[counter]]);
            writeVector(statefile, commanded_joint_accelerations[depth_commandedpos_indices[counter]]);

            // Write end eff pose, twist, wrenches
            writeVector(statefile, commanded_endeff_poses[depth_commandedendeff_indices[counter]]);
            writeVector(statefile, commanded_endeff_twists[depth_commandedendeff_indices[counter]]);
            writeVector(statefile, commanded_endeff_accelerations[depth_commandedendeff_indices[counter]]);

            // Close file
            statefile.close();
        }

        /// == Save start/end frames for each dataset - remove frames where box falls off table etc - Add pangolin controls?
        std::ofstream trainingFrames(pokefolder + "/trainingframes.txt");
        trainingFrames << startFrame << " " << endFrame;
        trainingFrames.close();
    }

    return 0;
}
