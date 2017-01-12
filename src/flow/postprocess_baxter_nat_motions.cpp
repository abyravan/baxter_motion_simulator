// Common
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

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

//#include "optimization/kernels/modToObs.h"
//#include "util/mirrored_memory.h"
//#include "geometry/grid_3d.h"

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

// PCL
//#include <pcl/common/common_headers.h>
//#include <pcl/visualization/pcl_visualizer.h>

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

/*
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
*/

// -----------------------------------------
// Read/Write eigen matrices from/to disk
namespace Eigen{
template<class Matrix>
bool write_binary(const char* filename, const Matrix& matrix){
    std::ofstream out(filename,ios::out | ios::binary | ios::trunc);
    if (!out.is_open()) return false;
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
    return true;
}
template<class Matrix>
bool read_binary(const char* filename, Matrix& matrix){
    std::ifstream in(filename,ios::in | std::ios::binary);
    if (!in.is_open()) return false;
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
    return true;
}
} // Eigen::

template <typename T>
std::size_t closestID(std::vector<T> const& sorted_vec, T query, std::size_t offset=0)
{
    auto const it = std::lower_bound(sorted_vec.begin()+offset, sorted_vec.end(), query);
    if (it == sorted_vec.end())
        return sorted_vec.size()-1; // All values in vec are less than query value, so closest is last element of vector
    else if (it == sorted_vec.begin())
        return 0;                   // All values in vec are greater than query, so closest is first element of vector
    else
    {
        if (fabs(*it - query) < fabs(*(it-1) - query)) // Check current element & element before to get closest one
            return it-sorted_vec.begin(); // Current element is closest
        else
            return it-1-sorted_vec.begin(); // Element before is closest
    }
}

template <typename T1, typename T2>
std::vector<T1> closestElement(std::vector<std::vector<T1> > const& vec, std::vector<T2> const &vec_t, T2 query_t,
                               std::size_t offset=0)
{
    std::size_t id = closestID(vec_t, query_t, offset);
    if (id == -1)
        return std::vector<T1>();
    else
        return vec[id];
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

// -----------------------------------------
// Main
int main(int argc, char **argv)
{
    /// ===== Get parameters

    // Default parameters
    std::string load_folder     = "";       // Folder to load saved files from (loads positions.csv, velocities.csv & efforts.csv)
    std::string saveparentname  = "postprocessmotions"; // Folder to save all post processed data in
    std::string save_prefix     = "motion"; // Default save prefix names: motion0, motion1, ...
    bool visualize              = true;     // Visualize the data in pangolin
    bool save_fullres_data      = false;    // By default, do not save full res images
    int num_threads             = 5;        // By default, we use 5 threads
    std::string step_list_str   = "[1]";    // Steps (depth images) in future to look at computing flow (Default: 1 => t+1)
    int st_frame                = 0;        // Frame ID to start computations from
    bool allstats               = false;    // By default, we compute stats only for frames from st_frame.
	 float framerate				  = 30;		  // Default = 30 fps

    //PARSE INPUT ARGUMENTS
    po::options_description desc("Allowed options",1024);
    desc.add_options()
        ("help", "produce help message")
        ("loadfolder", po::value<std::string>(&load_folder), "Path to load data from: Will look for positions.csv, velocities.csv here")
        ("saveparentname", po::value<std::string>(&saveparentname), "Name of folder to save all the results in")
        ("saveprefix", po::value<std::string>(&save_prefix), "Name of folder to save each dataset in")
        ("visualize",  po::value<bool>(&visualize), "Flag for visualizing trajectories. [Default: 1]")
        ("savefullres",po::value<bool>(&save_fullres_data), "Flag for saving full resolution image data. [Default: 0]")
        ("numthreads", po::value<int>(&num_threads), "Number of threads to use")
        ("steplist",   po::value<std::string>(&step_list_str), "Comma separated list of steps to compute flow. [Default: [1]]")
        ("stwriteframe", po::value<int>(&st_frame), "Frame ID to start computations from")
        ("allstats",   po::value<bool>(&allstats), "Compute statistics for all the frames - not just from st_frame")
		  ("fps" , 		  po::value<float>(&framerate), "Data frame rate. Default = 30 fps")
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
    assert(!load_folder.empty() && "Please pass in folder with recorded data using --loadfolder <path_to_folder>");
    cout << "Loading from directory: " << load_folder << endl;
    cout << "Folder prefix for saved data: " << save_prefix << endl;
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

    // Allow multi-threaded support for speeding up computation
    rgbd::threadGroup tg(num_threads);
    printf("Using %d threads to speed up computation \n",num_threads);

    /*
    /// ===== PCL viewer
    boost::shared_ptr<boost::thread> pcl_viewer_thread;
    if(visualize)
    {
        printf("Visualizing the color, depth images, point clouds and flow vectors \n");
        point_cloud_update = false; // Do not update point clouds yet
        pcl_viewer_thread.reset(new boost::thread(pclViewerThread));
    }
    */

    /// ===== Set up a DART tracker with the baxter model

    // Setup OpenGL/CUDA/Pangolin stuff - Has to happen before DART tracker initialization
    cudaGLSetGLDevice(0);
    cudaDeviceReset();
    const float totalwidth = 1920;
    const float totalheight = 1080;
    pangolin::CreateWindowAndBind("Main",totalwidth,totalheight);
    glewInit();
    glutInit(&argc, argv);

    /// ===== Pangolin initialization
    /// Pangolin mirrors the display, so we need to use TopLeft display for it. Our rendering needs BottomLeft

    // Pangolin window setup
    pangolin::CreatePanel("opt").SetBounds(0.0,0.9,pangolin::Attach::Pix(panelWidth), pangolin::Attach::Pix(2*panelWidth));

    // Initialize camera parameters and projection matrix
    int glWidth = 640;
    int glHeight = 480;
    float glFL = 589.3664541825391;// not sure what to do about these dimensions
    float glPPx = 320.5;
    float glPPy = 240.5;
    pangolin::OpenGlMatrixSpec glK_pangolin = pangolin::ProjectionMatrixRDF_TopLeft(glWidth,glHeight,glFL,glFL,glPPx,glPPy,0.01,1000);

    // Create the pangolin state
    pangolin::OpenGlRenderState camState(glK_pangolin);
    pangolin::View & camDisp = pangolin::Display("cam").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));
    pangolin::CreatePanel("pose").SetBounds(0.0,1.0,pangolin::Attach::Pix(-panelWidth), pangolin::Attach::Pix(-2*panelWidth));
    pangolin::Display("multi")
            .SetBounds(1.0, 0.0, pangolin::Attach::Pix(2*panelWidth), pangolin::Attach::Pix(-2*panelWidth))
            .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(camDisp);

    /// ===== Setup DART

    // Load baxter model
    dart::Tracker tracker;
    //const std::string objectModelFile = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter.xml";
    const std::string objectModelFile = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter_rosmesh.xml";
    tracker.addModel(objectModelFile, 0.01); // Add baxter model with SDF resolution = 1 cm
    int baxterID = 0;
    cout << "Initialized DART tracker with Baxter model. Num DOF: " << tracker.getPose(baxterID).getReducedArticulatedDimensions() << endl;

    // set up intersection matrix (for self-collision check)
    {
        int * selfIntersectionMatrix = dart::loadSelfIntersectionMatrix("/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter_intersection.txt",
                                                                        tracker.getModel(baxterID).getNumSdfs());
        tracker.setIntersectionPotentialMatrix(baxterID,selfIntersectionMatrix);
        delete [] selfIntersectionMatrix;
    }

    // Get the baxter pose and create a map between "frame name" and "pose dimension"
    dart::Pose &baxter_pose(tracker.getPose(baxterID));
    std::vector<std::string> model_joint_names;
    std::map<std::string, int> joint_name_to_pose_dim;
    for(int i = 0; i < baxter_pose.getReducedArticulatedDimensions(); i++)
    {
        model_joint_names.push_back(baxter_pose.getReducedName(i));
        joint_name_to_pose_dim[baxter_pose.getReducedName(i)] = i;
    }

    /// ===== Load the joint angles and velocities for the given joints (along with timestamps)

    // Read joint names
    std::vector<std::string> commanded_joint_names = read_csv_labels(load_folder + "/commandedpositions.csv");
    std::vector<std::string> valid_joint_names     = find_common_strings(model_joint_names, commanded_joint_names);
    cout << "Number of valid DOF: " << valid_joint_names.size() << endl;

    // Read commanded data
    std::vector<std::vector<double> > valid_commanded_joint_positions     = read_csv_data<double>(load_folder + "/commandedpositions.csv", valid_joint_names);
    std::vector<std::vector<double> > valid_commanded_joint_velocities    = read_csv_data<double>(load_folder + "/commandedvelocities.csv", valid_joint_names);
    std::vector<std::vector<double> > valid_commanded_joint_accelerations = read_csv_data<double>(load_folder + "/commandedaccelerations.csv", valid_joint_names);
    cout << "Number of commanded joint data from file: " << valid_commanded_joint_positions.size() << endl;

    // Read commanded timestamps
    std::vector<double> compos_t = read_csv_data<double>(load_folder + "/commandedpositions.csv", "time");
    std::vector<double> comvel_t = read_csv_data<double>(load_folder + "/commandedvelocities.csv", "time");
    std::vector<double> comacc_t = read_csv_data<double>(load_folder + "/commandedaccelerations.csv", "time");

    // Read end effector data
    std::vector<std::string> commanded_endeff_labels = read_csv_labels(load_folder + "/commandedendeffpositions.csv");
    commanded_endeff_labels.erase(commanded_endeff_labels.begin()); // Remove the time label
    std::vector<std::vector<double> > commanded_endeff_positions = read_csv_data<double>(load_folder + "/commandedendeffpositions.csv", commanded_endeff_labels);
    std::vector<double> comendeff_t = read_csv_data<double>(load_folder + "/commandedendeffpositions.csv", "time");

    /// == Read actual data and timestamps

    // Read actual data
    std::vector<std::vector<double> > valid_joint_positions  = read_csv_data<double>(load_folder + "/positions.csv", valid_joint_names);
    std::vector<std::vector<double> > valid_joint_velocities = read_csv_data<double>(load_folder + "/velocities.csv", valid_joint_names);
    std::vector<std::vector<double> > valid_joint_efforts    = read_csv_data<double>(load_folder + "/efforts.csv", valid_joint_names);
    cout << "Number of actual joint data from file: " << valid_joint_positions.size() << endl;

    // Read actual timestamps
    std::vector<double> actpos_t = read_csv_data<double>(load_folder + "/positions.csv", "time");
    std::vector<double> actvel_t = read_csv_data<double>(load_folder + "/velocities.csv", "time");
    std::vector<double> acteff_t = read_csv_data<double>(load_folder + "/efforts.csv", "time");

    /// ====== Setup model view matrix from disk
    // Model view matrix
    Eigen::Matrix4f modelView = Eigen::Matrix4f::Identity();
    Eigen::read_binary("/tmp/cameramodelview.dat", modelView);

    // Set cam state
    camState.SetModelViewMatrix(modelView);
    camDisp.SetHandler(new pangolin::Handler3D(camState));

    // Create a renderer
    pangolin::OpenGlMatrixSpec glK = pangolin::ProjectionMatrixRDF_BottomLeft(glWidth,glHeight,glFL,glFL,glPPx,glPPy,0.01,1000);
    l2s::Renderer<TYPELIST2(l2s::IntToType<l2s::RenderVertMapWMeshID>, l2s::IntToType<l2s::RenderDepth>)> renderer(640, 480, glK);
    renderer.setModelViewMatrix(modelView);

    // Pre-process to compute the mesh vertices and indices for all the robot parts
    std::vector<std::vector<float3> > meshVertices, transformedMeshVertices;
    std::vector<std::vector<float4> > meshVerticesWMeshID;
    std::vector<pangolin::GlBuffer> meshIndexBuffers;
    std::vector<std::vector<pangolin::GlBuffer *> > meshVertexAttributeBuffers;
    std::vector<int> meshFrameids, meshModelids;
    for (int m = 0; m < tracker.getNumModels(); ++m)
    {
        // Get the model
        for (int s = 0; s < tracker.getModel(m).getNumSdfs(); ++s)
        {
            // Get the frame number for the SDF and it's transform w.r.t robot base
            int f = tracker.getModel(m).getSdfFrameNumber(s);
            const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);

            // Iterate over all the geometries for the model and get the mesh attributes for the data
            for(int g = 0; g < tracker.getModel(m).getFrameNumGeoms(f); ++g)
            {
                // Get the mesh index
                int gid = tracker.getModel(m).getFrameGeoms(f)[g];
                int mid = tracker.getModel(m).getMeshNumber(gid);
                if(mid == -1) continue; // Has no mesh

                // Get the mesh
                const dart::Mesh mesh = tracker.getModel(m).getMesh(mid);
                meshFrameids.push_back(f); // Index of the frame for that particular mesh
                meshModelids.push_back(m); // ID of the model for that particular mesh

                // Get their vertices and transform them using the given frame to model transform
                meshVertices.push_back(std::vector<float3>(mesh.nVertices));
                transformedMeshVertices.push_back(std::vector<float3>(mesh.nVertices));
                meshVerticesWMeshID.push_back(std::vector<float4>(mesh.nVertices));
                for(int i = 0; i < mesh.nVertices; ++i)
                {
                    // Get mesh vertex and transform it
                    meshVertices.back()[i] = mesh.vertices[i];
                    transformedMeshVertices.back()[i] = tfm * mesh.vertices[i];

                    // Update the canonical vertex with the mesh ID
                    meshVerticesWMeshID.back()[i] = make_float4(mesh.vertices[i].x, mesh.vertices[i].y,
                                                                mesh.vertices[i].z, meshVertices.size()); // Add +1 to mesh IDs (BG is zero)
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

    /// ===== Iterate over all the datasets and save the depth/flow/control data

    // Create a folder to save the processed datasets
    std::string save_dir = load_folder + "/" + saveparentname + "/";
    createDirectory(save_dir);

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
        int f = tracker.getModel(0).getJointFrame(jointid);
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

    // Start timer
    struct timespec tic, toc;
    clock_gettime(CLOCK_REALTIME, &tic);

    /// == Get timestamps for dataset (make it so that it moves from 0 onwards)

    // Calculate number of depth frames
    double tstart = actpos_t.front(); double tend = actpos_t.back();
    double ttotal = tend - tstart;
    int numframes = std::round(framerate * ttotal); // We want output at "FRAMERATE" Hz

    // Get frame timestamps
    std::vector<double> timestamps;
    for(int i = 0; i < numframes; i++)
        timestamps.push_back(tstart + (ttotal * i)/(numframes-1));

    // Create a buffer to store data
    int buffer = maxVal(step_list) + 1; // We store this many depth images in memory
    std::vector<float *> depth_images, vmapwmeshid_images;
    std::vector<std::vector<dart::SE3> > mesh_transforms;
    std::vector<int> frame_ids;

    /// == Create directories
    std::string data_dir = save_dir + "/" + save_prefix + "0/";
    createDirectory(data_dir); // Global dir

    // Flow directories
    std::vector<std::string> flow_dirs;
    for(std::size_t k = 0; k < step_list.size(); k++)
    {
        // Get step size and create flow dir
        int step = step_list[k];
        std::string flow_dir = data_dir + "/flow_" + std::to_string(step) + "/";
        createDirectory(flow_dir);
        flow_dirs.push_back(flow_dir);
    }

    // TODO: Check flow + depth alignment. Make sure all frames have flow
    // Check file names properly
    // Update masks - model/frame IDs

    // Iterate over all the actual positions to generate frames
    int numflowframes = 0;
    std::vector<int> target_switches; target_switches.push_back(0);
    std::vector<double> endeff_target;
    for(int i = st_frame; i < numframes; i++)
    {
        // Current time
        double tcurr = timestamps[i];
        frame_ids.push_back(i);

        // Flag for writing frames. Do not write the last "buffer" frames
        bool writeFrame = (i < numframes-maxVal(step_list));

        /// == Find closest actual/commanded data
        /// Assumes that the times are properly sorted
        std::vector<double> pos = closestElement(valid_joint_positions,  actpos_t, tcurr);
        std::vector<double> vel = closestElement(valid_joint_velocities, actvel_t, tcurr);
        std::vector<double> eff = closestElement(valid_joint_efforts,    acteff_t, tcurr);

        std::vector<double> compos = closestElement(valid_commanded_joint_positions,  compos_t, tcurr);
        std::vector<double> comvel = closestElement(valid_commanded_joint_velocities, comvel_t, tcurr);
        std::vector<double> comacc = closestElement(valid_commanded_joint_accelerations, comacc_t, tcurr);
        std::vector<double> comendeff = closestElement(commanded_endeff_positions, comendeff_t, tcurr);

        // Update target ID file
        if(i == st_frame) endeff_target = comendeff;

        // If target has changed w.r.t previous point, update target ID
        double dist = squaredDistance(comendeff.data(), endeff_target.data(), 3);
        if ((dist > 1e-10) && (writeFrame))
        {
            endeff_target = comendeff;
            target_switches.push_back(i); // Target has changed in this frame
        }

        /// == Update the pose of the robot
        for(int k = 0; k < valid_joint_names.size(); k++)
        {
            if (joint_name_to_pose_dim.find(valid_joint_names[k]) != joint_name_to_pose_dim.end())
            {
                int pose_dim = joint_name_to_pose_dim[valid_joint_names[k]];
                tracker.getPose(baxterID).getReducedArticulation()[pose_dim] = pos[k];
            }
        }

        // Update the pose so that the FK is computed properly
        tracker.updatePose(baxterID);

        /// == Update mesh vertices based on new pose
        mesh_transforms.push_back(std::vector<dart::SE3> ());
        for (int k = 0; k < meshVertices.size(); k++)
        {
            // Get the SE3 transform for the frame
            int m = meshModelids[k];
            int f = meshFrameids[k];
            const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);
            mesh_transforms.back().push_back(dart::SE3(tfm.r0, tfm.r1, tfm.r2)); // Create a copy of the SE3 and save it

            // Transform the canonical vertices based on the new transform
            for(int j = 0; j < meshVertices[k].size(); ++j)
            {
                transformedMeshVertices[k][j] = tfm * meshVertices[k][j];
            }

            // Upload to pangolin vertex buffer
            meshVertexAttributeBuffers[k][0]->Upload(transformedMeshVertices[k].data(), transformedMeshVertices[k].size()*sizeof(float3));
            meshVertexAttributeBuffers[k][1]->Upload(meshVerticesWMeshID[k].data(), meshVerticesWMeshID[k].size()*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
        }

        /// == Render a depth image
        glEnable(GL_DEPTH_TEST);
        renderer.renderMeshes<l2s::RenderDepth>(meshVertexAttributeBuffers, meshIndexBuffers);

        // Get a float image
        depth_images.push_back(new float[glWidth * glHeight]); // 1-channel float image
        renderer.texture<l2s::RenderDepth>().Download(depth_images.back());

        /// == Render a vertex map with the mesh ids
        renderer.renderMeshes<l2s::RenderVertMapWMeshID>(meshVertexAttributeBuffers, meshIndexBuffers);

        // Get the vertex image with mesh id
        vmapwmeshid_images.push_back(new float[glWidth * glHeight * 4]); // 4-channel float image
        renderer.texture<l2s::RenderVertMapWMeshID>().Download(vmapwmeshid_images.back(), GL_RGBA, GL_FLOAT);

        /// == Save the depth and pixel label images to disk

        // == Convert depth float to an opencv matrix
        cv::Mat depth_mat_f = cv::Mat(glHeight, glWidth, CV_32FC1, depth_images.back());

        // == Create a new matrix for the mask
        cv::Mat labels_mat_c = cv::Mat(glHeight, glWidth, CV_8UC1);
        int ct = 0;
        for(int r = 0; r < glHeight; r++)
        {
            for(int c = 0; c < glWidth; c++, ct+=4)
            {
                float *point = &(vmapwmeshid_images.back()[ct]);
                labels_mat_c.at<char>(r,c) = (char) std::round(point[3]);
            }
        }

        if(save_fullres_data)
        {
            // Convert the depth data from "m" to "0.1 mm" range and store it as a 16-bit unsigned short
            // 2^16 = 65536  ;  1/10 mm = 1/10 * 1e-3 m = 1e-4 m  ;  65536 * 1e-4 = 6.5536
            // We can represent depth from 0 to +6.5536 m using this representation (enough for our data)
            cv::Mat depth_mat;
            depth_mat_f.convertTo(depth_mat, CV_16UC1, M_TO_TENTH_MM); // 0.1 mm resolution and round off to *nearest* unsigned short
            if (writeFrame) cv::imwrite(data_dir + "depth" + std::to_string(i) + ".png", depth_mat); // Save depth image

            // Save mask image
            if (writeFrame) cv::imwrite(data_dir + "labels" + std::to_string(i) + ".png", labels_mat_c); // Save depth image
        }

        // == Subsample depth image by NN interpolation (don't do cubic/bilinear interp) and save it
        cv::Mat depth_mat_f_sub(std::round(0.5 * glHeight), std::round(0.5 * glWidth), CV_32FC1);
        cv::resize(depth_mat_f, depth_mat_f_sub, cv::Size(), 0.5, 0.5, CV_INTER_NN); // Do NN interpolation

        // Scale the depth image as a 16-bit single channel unsigned char image
        cv::Mat depth_mat_sub;
        depth_mat_f_sub.convertTo(depth_mat_sub, CV_16UC1, M_TO_TENTH_MM); // Scale from m to 0.1 mm resolution and save as ushort
        if (writeFrame) cv::imwrite(data_dir + "depthsub" + std::to_string(i) + ".png", depth_mat_sub); // Save depth image

        /// == Subsample label image and save it
        cv::Mat labels_mat_c_sub(std::round(0.5 * glHeight), std::round(0.5 * glWidth), CV_8UC1);
        cv::resize(labels_mat_c, labels_mat_c_sub, cv::Size(), 0.5, 0.5, CV_INTER_NN); // Do NN interpolation
        if (writeFrame) cv::imwrite(data_dir + "labelssub" + std::to_string(i) + ".png", labels_mat_c_sub); // Save depth image

        /// == State file - Save pos, vel, eff, com vals
        if (writeFrame)
        {
            std::ofstream statefile(data_dir + "/state" + std::to_string(i) + ".txt");
            writeVector(statefile, pos);
            writeVector(statefile, vel);
            writeVector(statefile, eff);
            writeVector(statefile, compos);
            writeVector(statefile, comvel);
            writeVector(statefile, comacc);
            writeVector(statefile, comendeff);
            statefile.close();
        }

        /// == SE3 file - Write SE3 of each of the meshes as labeled in the "labels" image
        if (writeFrame)
        {
            std::ofstream se3file(data_dir + "/se3state" + std::to_string(i) + ".txt");
            for(std::size_t k = 0; k < mesh_transforms.back().size(); k++)
            {
                se3file << k+1 << endl; // Frame ID (BG is frame 0)
                serializeSE3(se3file, mesh_transforms.back()[k]); // SE3 (no new line here)
                se3file << endl << endl; // Add a space before the next frame
            }
            se3file.close();
        }

        // Write stuff to disk / compute flows for all images except the last "buffer" images
        if (depth_images.size() >= buffer)
        {
            /// == Compute flows for each of the steps and save the data to disk
            // Iterate over the different requested "step" lengths and compute flow for each
            // Each of these is stored in their own folder - "flow_k" where "k" is the step-length
            std::vector<cv::Mat> flow_mat_f(step_list.size());
            for(std::size_t k = 0; k < step_list.size(); k++)
            {
                // Compute flow multi-threaded
                tg.addTask([&,k]() // Lambda function/thread
                {
                    // Compute flow between frames @ t & t+step
                    flow_mat_f[k] = compute_flow_between_frames(vmapwmeshid_images[0],
                                                                mesh_transforms[0],
                                                                mesh_transforms[step_list[k]],
                                                                modelView, glWidth, glHeight);

                    // Convert the flow data from "m" to "0.1 mm" range and store it as a 16-bit unsigned short
                    // 2^16 = 65536. 2^16/2 ~ 32767 * 1e-4 (0.1mm) = +-3.2767 (since we have positive and negative values)
                    // We can represent motion ranges from -3.2767 m to +3.2767 m using this representation
                    // This amounts to ~300 cm/frame ~ 90m/s speed (which should be way more than enough for the motions we currently have)
                    if(save_fullres_data)
                    {
                        cv::Mat flow_mat = convert_32FC3_to_16UC3(flow_mat_f[k], M_TO_TENTH_MM);
                        cv::imwrite(flow_dirs[k] + "flow" + std::to_string(frame_ids[0]) + ".png", flow_mat);
                    }

                    // Save subsampled image by NN interpolation
                    // KEY: Don't do cubic/bilinear interpolation as it leads to interpolation of flow values across boundaries which is incorrect
                    cv::Mat flow_mat_f_sub(round(0.5 * glHeight), round(0.5 * glWidth), CV_32FC3);
                    cv::resize(flow_mat_f[k], flow_mat_f_sub, cv::Size(), 0.5, 0.5, CV_INTER_NN); // Do NN interpolation (otherwise flow gets interpolated, which is incorrect)

                    // Scale the flow image data as a 16-bit 3-channel ushort image & save it
                    cv::Mat flow_mat_sub = convert_32FC3_to_16UC3(flow_mat_f_sub, M_TO_TENTH_MM);
                    cv::imwrite(flow_dirs[k] + "flowsub" + std::to_string(frame_ids[0]) + ".png", flow_mat_sub);
                });
            }
            tg.wait();

            // In case we are asked to visualize the flow outputs and/or images do it here
            if(visualize)
            {
                // ==== Image stuff
                cv::Mat depth(glHeight, glWidth, CV_32F, depth_images[0]);
                cv::imshow("Subsampled Colorized Depth", colorize_depth(depth));
                cv::imshow("Labels", colorize_depth(labels_mat_c));

                // ==== Set robot at proper pose
                for(int j = 0; j < valid_joint_names.size(); j++)
                {
                    if (joint_name_to_pose_dim.find(valid_joint_names[j]) != joint_name_to_pose_dim.end())
                    {
                        int pose_dim = joint_name_to_pose_dim[valid_joint_names[j]];
                        tracker.getPose(baxterID).getReducedArticulation()[pose_dim] = pos[j];
                    }
                }

                // ==== Render pangolin frame
                renderPangolinFrame(tracker, camState, camDisp);

                // Iterate over all the depth images
                for(std::size_t k = 0; k < step_list.size(); k++)
                {
                    // ==== Show flow image
                    cv::imshow("Flow", colorize_depth(flow_mat_f[k]));
                    cv::waitKey(2);      // wait for 2 ms

                    /*
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
                           float z = depth_mat_f.at<float>(r,c);
                           float x = (c - glPPx)/glFL; x *= z;
                           float y = (r - glPPy)/glFL; y *= z;
                           point_cloud->at(c,r) = pcl::PointXYZ(x,y,z);
                           flow_cloud->at(c,r) = pcl::Normal(flow_mat_f[k].at<cv::Vec3f>(r,c)[0],
                                                             flow_mat_f[k].at<cv::Vec3f>(r,c)[1],
                                                             flow_mat_f[k].at<cv::Vec3f>(r,c)[2]);
                       }
                    }

                    // Set flag for visualizer
                    point_cloud_update = true;
                    update_lock.unlock(); // Unlock mutex so that display happens
                    */
                }
            }

            // Free the first image -> we just computed the flow for it
            free(depth_images[0]);
            free(vmapwmeshid_images[0]);

            // Clear buffer
            depth_images.erase(depth_images.begin());
            vmapwmeshid_images.erase(vmapwmeshid_images.begin());
            mesh_transforms.erase(mesh_transforms.begin());
            frame_ids.erase(frame_ids.begin());
            numflowframes++;
        }

        // Time taken
        if ((i % 1000 == 0) || (i == numframes-1))
        {
            clock_gettime(CLOCK_REALTIME, &toc);
            timespec_sub(&toc, &tic);
            printf("Frame: [%d/%d], Time taken so far: %f \n",i+1,numframes,timespec_double(&toc));
        }
    }

    // In case we are computing statistics for all frames, do some modifications
    if (allstats && st_frame > 0)
    {
        // Update number of flow frames:
        numflowframes += st_frame;

        // Update the target_switches vector
        target_switches.clear();
        target_switches.push_back(0); // Initial switch is at frame 0
        for(int i = 0; i < numframes; i++)
        {
            bool writeFrame = (i < numframes-maxVal(step_list));
            double tcurr = timestamps[i];
            std::vector<double> comendeff = closestElement(commanded_endeff_positions, comendeff_t, tcurr);

            // Update target ID file
            if(i == 0) endeff_target = comendeff;

            // If target has changed w.r.t previous point, update target ID
            double dist = squaredDistance(comendeff.data(), endeff_target.data(), 3);
            if ((dist > 1e-10) && (writeFrame))
            {
                endeff_target = comendeff;
                target_switches.push_back(i); // Target has changed in this frame
            }
        }
    }

    // Save post-process statistics
    std::ofstream postprocessfile(data_dir + "/postprocessstats.txt");
    postprocessfile << numflowframes << " " << maxVal(step_list) << endl;
    postprocessfile.close();

    // Save target statistics
    std::ofstream targetstatsfile(data_dir + "/targetstats.txt");
    for(std::size_t i = 0; i < target_switches.size()-1; i++)
    {
        targetstatsfile << target_switches[i] << " " << target_switches[i+1]-1 << endl; // Start/end frame IDs for each target
    }
    targetstatsfile << target_switches.back() << " " << numflowframes-1; // Final set ends at numflowframes
    targetstatsfile.close();

    // Free memory for the rendering buffers
    for(std::size_t i = 0; i < meshVertexAttributeBuffers.size(); i++)
        for(size_t j = 0; j < meshVertexAttributeBuffers[i].size(); j++)
            free(meshVertexAttributeBuffers[i][j]);

    // Free memory for depth images
    for(std::size_t i = 0; i < depth_images.size(); i++)
        free(depth_images[i]);

    // Free memory for the vertex map images
    for(std::size_t i = 0; i < vmapwmeshid_images.size(); i++)
        free(vmapwmeshid_images[i]);

    return 0;
}
