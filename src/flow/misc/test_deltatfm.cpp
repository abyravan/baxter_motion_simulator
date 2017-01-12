// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/legacy/legacy.hpp>

// C++
#include <fstream>
#include <iostream>
#include <string>

// Boost thread/mutex
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>

// Pangolin
#include <pangolin/pangolin.h>

// PCL
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

Eigen::Transform<float,3,Eigen::Affine> generate_tfm(const std::vector<float> tfm)
{
    assert(tfm.size() == 7 && "Transform needs to have position and quaternion");
    Eigen::Transform<float,3,Eigen::Affine> t = Eigen::Translation3f(tfm[0], tfm[1], tfm[2]) *
                                                Eigen::Quaternionf(tfm[6], tfm[3], tfm[4], tfm[5]);
    return t;
}

int main(int argc, char** argv)
{
    // Get depth and color data
    std::string color_name = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/src/flow/misc/color.png";
    std::string depth_name = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/src/flow/misc/depth.png";

    // Load images
    printf("Loading images from disk \n");
    cv::Mat color = cv::imread(color_name); // Load color image from disk
    cv::Mat depth = cv::imread(depth_name, CV_LOAD_IMAGE_ANYDEPTH); // Load depth image from disk
    printf("Finished Loading images from disk \n");

    // Get pose
    std::vector<float> camera_pose = {-0.025, -1.4, 1.75, -0.866025, 8.96228e-13, 8.96071e-13, 0.5};
    std::vector<float> box_pose = {-0.0440096, -0.350383, 0.977264, -4.1696471e-12, -6.3972442e-14, -3.1992092e-15, 1};
    std::vector<float> ball_pose = {-0.35366762, -0.54303966, 1.0363114, 0, 0, 0, 1};
    std::vector<float> box_pose_next = {0.037657779, -0.32124858, 0.97726111, 2.7180286e-06, 1.3963831e-05, 0.095635656, 0.99541641};

    // Convert these to matrices
    Eigen::Transform<float,3,Eigen::Affine> camTinWorld  = generate_tfm(camera_pose);
    Eigen::Transform<float,3,Eigen::Affine> boxTinWorld  = generate_tfm(box_pose);
    Eigen::Transform<float,3,Eigen::Affine> ballTinWorld = generate_tfm(ball_pose);
    Eigen::Transform<float,3,Eigen::Affine> nextBoxTinWorld  = generate_tfm(box_pose_next);

    // Get box transform w.r.t camera
    Eigen::Transform<float,3,Eigen::Affine> boxTinCam = camTinWorld.inverse() * boxTinWorld;
    Eigen::Transform<float,3,Eigen::Affine> nextBoxTinCam = camTinWorld.inverse() * nextBoxTinWorld;
    Eigen::Transform<float,3,Eigen::Affine> fullT = nextBoxTinCam * boxTinCam.inverse();

    // Camera parameters
    int imageWidth=640,imageHeight=480;
    float f = 589.376;

    // Create pangolin window
    pangolin::CreateWindowAndBind("Delta tfm test",1280,960);
    pangolin::CreatePanel("opt").SetBounds(0,1,0,0.2);
    pangolin::OpenGlRenderState camState(pangolin::ProjectionMatrixRDF_TopLeft(imageWidth,imageHeight,f,f,320,240,0.01,1000.0));
    pangolin::Handler3D handler(camState);
    pangolin::View & pcDisp = pangolin::Display("pointcloud").SetAspect(imageWidth/static_cast<float>(imageHeight)).SetHandler(&handler);
    pangolin::Display("multi").SetBounds(0,1.0,0.2,1).SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pcDisp);

    // Create sliders for the delta transforms
    static pangolin::Var<bool> deltaTransform("opt.deltaTransform",true,true);
    static pangolin::Var<float> dx("opt.dx",0,-0.5,0.5);
    static pangolin::Var<float> dy("opt.dy",0,-0.5,0.5);
    static pangolin::Var<float> dz("opt.dz",0,-0.5,0.5);
    static pangolin::Var<float> wx("opt.wx",0,-1,1);
    static pangolin::Var<float> wy("opt.wy",0,-1,1);
    static pangolin::Var<float> wz("opt.wz",0,-1,1);
    static pangolin::Var<bool> translationFirst("opt.translationFirst",true,true);
    static pangolin::Var<bool> usePivot("opt.usePivot",true,true);
    static pangolin::Var<float> pivotx("opt.pivotx",0,-5,5);
    static pangolin::Var<float> pivoty("opt.pivoty",0,-5,5);
    static pangolin::Var<float> pivotz("opt.pivotz",0,-5,5);
    static pangolin::Var<float> fullx("opt.fullx",0,-5,5);
    static pangolin::Var<float> fully("opt.fully",0,-5,5);
    static pangolin::Var<float> fullz("opt.fullz",0,-5,5);
    static pangolin::Var<float> fullwx("opt.fullwx",0,-1,1);
    static pangolin::Var<float> fullwy("opt.fullwy",0,-1,1);
    static pangolin::Var<float> fullwz("opt.fullwz",0,-1,1);
    static pangolin::Var<float> gtfullx("opt.gtfullx",0,-5,5);
    static pangolin::Var<float> gtfully("opt.gtfully",0,-5,5);
    static pangolin::Var<float> gtfullz("opt.gtfullz",0,-5,5);
    static pangolin::Var<float> gtfullwx("opt.gtfullwx",0,-1,1);
    static pangolin::Var<float> gtfullwy("opt.gtfullwy",0,-1,1);
    static pangolin::Var<float> gtfullwz("opt.gtfullwz",0,-1,1);

    while (!pangolin::ShouldQuit())
    {
        // General stuff
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        if (pangolin::HasResized())
        {
            pangolin::DisplayBase().ActivateScissorAndClear();
        }

        // Get the delta transform based on slider values
        Eigen::Transform<float,3,Eigen::Affine> deltaT;
        float angle = sqrt(wx*wx + wy*wy + wz*wz); // Angle
        if (angle == 0)
        {
            deltaT = Eigen::Translation3f(dx, dy, dz);
        }
        else
        {
            deltaT = Eigen::Translation3f(dx, dy, dz) * Eigen::AngleAxisf(angle, Eigen::Vector3f(wx/angle, wy/angle, wz/angle));
        }

        // Get transform to be applied to points
        Eigen::Transform<float,3,Eigen::Affine> ptT;
        if (deltaTransform)
        {
            ptT = boxTinCam * deltaT * boxTinCam.inverse();
        }
        else
        {
            if (usePivot)
            {
                // Rotate around not-camera
                ptT = Eigen::Translation3f(pivotx, pivoty, pivotz) * deltaT * Eigen::Translation3f(-pivotx, -pivoty, -pivotz); // Full transform
            }
            else if (translationFirst)
            {
                ptT = Eigen::AngleAxisf(angle, Eigen::Vector3f(wx/angle, wy/angle, wz/angle));
            }
            else
            {
                ptT = deltaT; // Full transform
            }
        }

        // Set full transform vars
        fullx = ptT.translation().x();
        fully = ptT.translation().y();
        fullz = ptT.translation().z();
        Eigen::AngleAxisf aa = Eigen::AngleAxisf(ptT.rotation());
        fullwx = aa.axis()(0) * aa.angle();
        fullwy = aa.axis()(1) * aa.angle();
        fullwz = aa.axis()(2) * aa.angle();

        // Set GT full transform vars
        gtfullx = fullT.translation().x();
        gtfully = fullT.translation().y();
        gtfullz = fullT.translation().z();
        Eigen::AngleAxisf aa1 = Eigen::AngleAxisf(fullT.rotation());
        gtfullwx = aa1.axis()(0) * aa1.angle();
        gtfullwy = aa1.axis()(1) * aa1.angle();
        gtfullwz = aa1.axis()(2) * aa1.angle();

        // Upload the depth
        if (!depth.empty() )
        {
            //show the point cloud
            glClearColor(1.0,1.0,1.0,0.0);
            pcDisp.ActivateScissorAndClear(camState);
            glEnable(GL_DEPTH_TEST);
            glPointSize(3.0);
            glBegin(GL_POINTS);

            // Convert to point cloud
            float scale = 0.0001;
            for (int i=0; i < depth.rows; i++)
            {
                for (int j=0; j < depth.cols; j++)
                {
                    // Get the 3D point
                    float x,y,z;
                    z = (float)  depth.at<ushort>(i,j) * scale ;
                    x = (j - 320) * z / f;
                    y = (i - 240) * z / f;

                    // Display the 3D point colored based on the "png"
                    float ppert1[4] = {x,y,z,1.0};
                    glColor3ub(color.at<cv::Vec3b>(i,j)[0],
                               color.at<cv::Vec3b>(i,j)[1],
                               color.at<cv::Vec3b>(i,j)[2]);
                    glVertex3fv((float *)ppert1);

                    // Transform only the box using the delta, box 1 & camera transforms
                    if (color.at<cv::Vec3b>(i,j)[0] == 0 && color.at<cv::Vec3b>(i,j)[1] == 0 && color.at<cv::Vec3b>(i,j)[2] == 255)
                    {
                        Eigen::Vector3f tp;
                        if (!deltaTransform && translationFirst)
                            tp = ptT * Eigen::Vector3f(x+dx, y+dy, z+dz);
                        else
                            tp = ptT * Eigen::Vector3f(x,y,z);
                        float ppert2[4] = {tp(0),tp(1),tp(2),1.0};
                        glColor3ub(255,0,0); // Red color
                        glVertex3fv((float *)ppert2);
                    }

                    // Get next transform
                    if (color.at<cv::Vec3b>(i,j)[0] == 0 && color.at<cv::Vec3b>(i,j)[1] == 0 && color.at<cv::Vec3b>(i,j)[2] == 255)
                    {
                        Eigen::Vector3f tp = fullT * Eigen::Vector3f(x,y,z);
                        float ppert3[4] = {tp(0),tp(1),tp(2),1.0};
                        glColor3ub(0,255,0); // Green color
                        glVertex3fv((float *)ppert3);
                    }
                }
            }

            // Finish
            glEnd();
            glDisable(GL_DEPTH_TEST);

        }

        // Render
        glColor3ub(255,255,255);

        // Finish rendering
        pangolin::FinishFrame();
    }

    return 0;
}

