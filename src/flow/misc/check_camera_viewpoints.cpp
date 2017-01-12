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

#define EIGEN_DONT_ALIGN
#define M_TO_MM 1e3
#define M_TO_TENTH_MM 1e4 // Represent data as one tenth of a mm
#define M_TO_HUNDREDTH_MM 1e5 // Represent data as one hundredth of a mm

using namespace std;
namespace po = boost::program_options;
const int panelWidth = 180;

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

// -----------------------------------------
// Main
int main(int argc, char **argv)
{
    /// ===== Get parameters

    // Default parameters
    std::string load_folder     = "";       // Folder to load saved files from (loads positions.csv, velocities.csv & efforts.csv)

    //PARSE INPUT ARGUMENTS
    po::options_description desc("Allowed options",1024);
    desc.add_options()
        ("help", "produce help message")
        ("loadfolder", po::value<std::string>(&load_folder), "Path to load data from: Will look for positions.csv, velocities.csv here")
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

    /// ===== Pangolin options
    static pangolin::Var<float> camX("opt.camX", -2, -4.0, 4.0);
    static pangolin::Var<float> camY("opt.camY", 0.0715, -3.0, 3.0);
    static pangolin::Var<float> camZ("opt.camZ", -0.715, -3.0, 3.0);
    static pangolin::Var<float> camRX("opt.camRX", -0.4114, -M_PI, M_PI);
    static pangolin::Var<float> camRY("opt.camRY", 1.795, -M_PI, M_PI);
    static pangolin::Var<float> camRZ("opt.camRZ", -M_PI/2, -M_PI, M_PI);
    static pangolin::Var<bool> playRobotPoses("opt.playRobotPoses",false,true); // Play video
    static pangolin::Var<bool> setCameraPose("opt.setCamPose",false,true); // Play video
    static pangolin::Var<bool> done("opt.done",false,true); // Exit pangolin loop

    /// ===== Setup DART

    // Load baxter model
    dart::Tracker tracker;
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

    // Read the labels in the CSV file and get valid joint names (both recorded and from model)
    std::vector<std::string> recorded_joint_names = read_csv_labels(load_folder + "/positions.csv");
    std::vector<std::string> valid_joint_names    = find_common_strings(model_joint_names, recorded_joint_names);
    cout << "Number of valid DOF: " << valid_joint_names.size() << endl;

    // Read joint angles for all joints on the robot and the file
    std::vector<std::vector<double> > valid_joint_positions  = read_csv_data<double>(load_folder + "/positions.csv", valid_joint_names);
    cout << "Number of valid joint data from recorded file: " << valid_joint_positions.size() << endl;

    // Read joint velocities (recorded) for all joints on the robot and the file
    std::vector<std::vector<double> > valid_joint_velocities = read_csv_data<double>(load_folder + "/velocities.csv", valid_joint_names);
    assert(valid_joint_positions.size() == valid_joint_velocities.size() && "Position and Velocity files do not have same number of rows.");

    // Read joint efforts (recorded) for all joints on the robot and the file
    std::vector<std::vector<double> > valid_joint_efforts = read_csv_data<double>(load_folder + "/efforts.csv", valid_joint_names);
    assert(valid_joint_positions.size() == valid_joint_efforts.size() && "Position and Effort files do not have same number of rows.");

    // Read joint velocities (commanded) for all joints on the robot and the file
    std::vector<std::vector<double> > valid_commanded_joint_velocities = read_csv_data<double>(load_folder + "/commandedvelocities.csv", valid_joint_names);
    assert(valid_joint_positions.size() == valid_commanded_joint_velocities.size() && "Position and Commanded velocity files do not have same number of rows.");

    // Create a default parameter map
    std::map<std::string, double> parameter_map;
    parameter_map[std::string("recordrate")] = 50.0; // Default parameter

    // Read parameters file and update the map if the file exists
    std::vector<std::string> parameter_names = read_csv_labels(load_folder + "/parameters.csv");
    std::vector<std::vector<double> > parameter_vector = read_csv_data<double>(load_folder + "/parameters.csv", parameter_names);
    for(int i = 0; i < parameter_names.size(); i++)
        parameter_map[parameter_names[i]] = parameter_vector[0][i]; // Get the parameters

    // Get the timestamps from the files
    std::vector<double> timestamps = read_csv_data<double>(load_folder + "/positions.csv", "time");
    assert(valid_joint_positions.size() == timestamps.size() && "Position and Timestamps do not have same number of rows.");
    cout << "Finished reading all files from disk" << endl;

    /// ====== Setup model view matrix from disk
    // Load from file
    Eigen::Matrix4f modelView = Eigen::Matrix4f::Identity();
    Eigen::read_binary("/tmp/cameramodelview.dat", modelView);

    // Set cam state
    camState.SetModelViewMatrix(modelView);
    camDisp.SetHandler(new pangolin::Handler3D(camState));

    /// ====== Use pangolin to decide camera pose (set flag for this)
    std::size_t frameid = 0;
    for (int pangolinFrame=1; !pangolin::ShouldQuit() && !pangolin::Pushed(done); ++pangolinFrame)
    {
        /// == Set model view matrix (if asked)
        pangolin::OpenGlMatrix glM = pangolin::OpenGlMatrix::RotateZ(camRZ) *
                                     pangolin::OpenGlMatrix::RotateY(camRY) *
                                     pangolin::OpenGlMatrix::RotateZ(camRX) *
                                     pangolin::OpenGlMatrix::Translate(camX, camY, camZ);
        if (setCameraPose)
        {
            Eigen::Matrix4f modelView  = glM;
            camState.SetModelViewMatrix(modelView);
            camDisp.SetHandler(new pangolin::Handler3D(camState));
        }

        /// == If play is pressed, play the trajectory
        /// == Once the end is reached, rewind.
        bool play = playRobotPoses;
        if (play)
        {
            frameid++;
            if (frameid == valid_joint_positions.size()) frameid = 0;
        }

        /// == Set the baxter pose
        for(std::size_t k = 0; k < valid_joint_names.size(); k++)
        {
            if (joint_name_to_pose_dim.find(valid_joint_names[k]) != joint_name_to_pose_dim.end())
            {
                int pose_dim = joint_name_to_pose_dim[valid_joint_names[k]];
                tracker.getPose(baxterID).getReducedArticulation()[pose_dim] = valid_joint_positions[frameid][k];
            }
        }

        // === TODO: Internally, we can show the rendered depth images as well

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
        }

        // Disable lighting
        glPopMatrix();
        glDisable(GL_LIGHTING);
        glColor4ub(255,255,255,255);

        // Finish frame
        pangolin::FinishFrame();
        //usleep(1000);

        if (pangolinFrame % 100 == 0) cout << camState.GetModelViewMatrix() << endl;
    }

    /// == Once done, print/save the cam data
    Eigen::Matrix4f modelViewFinal = camState.GetModelViewMatrix();
    Eigen::write_binary("/tmp/cameramodelview.dat", modelViewFinal);
    cout << modelViewFinal << endl;

    return 0;
}
