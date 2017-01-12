#include <iostream>
#include <stdio.h>
#include <string.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/utils/timer.h>

#include <Eigen/Dense>

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

#include "optimization/kernels/modToObs.h"
#include "util/mirrored_memory.h"
#include "geometry/grid_3d.h"

#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>

#include <unistd.h>

#define EIGEN_DONT_ALIGN

using namespace std;

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

int main(int argc, char **argv) {

    const std::string objectModelFile = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter_rosmesh.xml";
    //const std::string objectModelFile = "/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter.xml";

    // Get the joint angles
    std::ifstream file("/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/data/test_2016_05_04_22_47_47/positions.csv");

    // Get the first line and save data to a vector of names
    std::string first_line;
    std::getline(file, first_line);
    assert(file.good() && "Reached end of line with first line!");

    // Get the joint names
    int ct = 0;
    std::vector<std::string> joint_names;
    std::stringstream iss(first_line);
    while(iss.good())
    {
        // Iterate all comma-separated values
        std::string val;
        std::getline(iss, val, ',');
        if (ct > 0) joint_names.push_back(val);
        ct++;
    }

    // Create a double 2D array with the data
    std::vector<std::vector<double> > joint_positions;
    std::vector<double> times;
    while(file.good())
    {
        // Get a line
        std::string line;
        std::getline(file, line);

        // Read into a vector
        int ct = 0;
        std::vector<double> positions;
        std::stringstream iss(line);
        while(iss.good())
        {
            // Iterate all comma-separated values
            std::string val;
            std::getline(iss, val, ',');

            // Get value
            if (ct == 0)
                times.push_back(atof(val.c_str()));
            else
                positions.push_back(atof(val.c_str()));
            ct++;
        }

        // Add to joint positions and time
        joint_positions.push_back(positions);
    }
    cout << "Number of different joint positions: " << joint_positions.size() << endl;
    cout << "Number of joint angles saved: " << joint_names.size() << endl;
    assert(joint_names.size() == joint_positions[0].size());

    // -=-=-=- initializations -=-=-=-

    cudaGLSetGLDevice(0);
    cudaDeviceReset();

    const float totalwidth = 1920;
    const float totalheight = 1080;
    pangolin::CreateWindowAndBind("Main",totalwidth,totalheight);

    glewInit();
    dart::Tracker tracker;

    // -=-=-=- pangolin window setup -=-=-=-

    int glWidth = 640;
    int glHeight=480;
    float glFL = 589.3664541825391;// not sure what to do about these dimensions
    float glPPx = 320.5;
    float glPPy = 240.5;
    pangolin::OpenGlMatrixSpec glK = pangolin::ProjectionMatrixRDF_BottomLeft(glWidth,glHeight,glFL,glFL,glPPx,glPPy,0.01,1000);
    pangolin::OpenGlRenderState camState(glK);
    pangolin::View & camDisp = pangolin::Display("cam").SetAspect(glWidth*1.0f/(glHeight*1.0f)).SetHandler(new pangolin::Handler3D(camState));

//    pangolin::View & imgDisp = pangolin::Display("img").SetAspect(640.0f/480.0f);
    pangolin::DataLog infoLog;
    {
        std::vector<std::string> infoLogLabels;
        infoLogLabels.push_back("errObsToMod");
        infoLogLabels.push_back("errModToObs");
        infoLogLabels.push_back("stabilityThreshold");
        infoLogLabels.push_back("resetThreshold");
        infoLog.SetLabels(infoLogLabels);
    }

    pangolin::CreatePanel("pose").SetBounds(0.0,1.0,pangolin::Attach::Pix(-panelWidth), pangolin::Attach::Pix(-2*panelWidth));

    pangolin::Display("multi")
            .SetBounds(1.0, 0.0, pangolin::Attach::Pix(2*panelWidth), pangolin::Attach::Pix(-2*panelWidth))
            .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(camDisp);

    std::vector<pangolin::Var<float> *> sizeVars;

    // ----

    cout << "Before adding model" << endl;
    // resolution down to 1cm to not run out of memory
    tracker.addModel(objectModelFile, 0.01);
    cout << "Added model successfully" << endl;
    dart::Pose &pose(tracker.getPose(0));

    // set up potential intersections
    {
        int * selfIntersectionMatrix = dart::loadSelfIntersectionMatrix("/home/barun/Projects/rgbd/ros-pkg-irs/wamTeach/ros_pkgs/catkin_ws/src/baxter_motion_simulator/models/baxter/baxter_intersection.txt",tracker.getModel(0).getNumSdfs());
        tracker.setIntersectionPotentialMatrix(0,selfIntersectionMatrix);
        cout << "Number of SDFs: " << tracker.getModel(0).getNumSdfs() << endl;
        cout << "Intersection matrix: " << endl;
        for(int ct = 0; ct <  tracker.getModel(0).getNumSdfs() * tracker.getModel(0).getNumSdfs(); ct++)
        {
            cout << selfIntersectionMatrix[ct] << " ";
            if (((ct + 1) % tracker.getModel(0).getNumSdfs()) == 0)
                cout << endl;
        }
        delete [] selfIntersectionMatrix;
    }

    /*
    cout << "Num SDFs: " << tracker.getModel(0).getNumSdfs() << endl;
    cout << "Num Collision clouds: " << tracker.getCollisionCloudSize(0) << endl;
    for(int i = 0; i < tracker.getModel(0).getNumJoints(); i++)
    {
        cout << "Joint: " << i << " Name: " << tracker.getModel(0).getJointName(i) << " Frame: " <<
                tracker.getModel(0).getJointFrame(i) << " SDF: " << tracker.getModel(0).getFrameSdfNumber(tracker.getModel(0).getJointFrame(i)) << endl;
    }
    */

    // I think this is the identity
    pose.setTransformModelToCamera(dart::SE3());// need an se3

    float *jointangles = pose.getReducedArticulation(); // array of joint angles
    int numjointangles = pose.getReducedArticulatedDimensions();
    cout << "Num DOF: " << numjointangles << endl;

    // Create a map between joint names and dimension of the pose
    std::map<std::string, int> joint_name_to_pose_dim;
    for(int i = 0; i < pose.getReducedArticulatedDimensions(); i++)
    {
        joint_name_to_pose_dim[pose.getReducedName(i)] = i;
        cout << i << " " << pose.getReducedName(i) << " " << tracker.getModel(0).getJointName(i) << endl;
    }

    // stuff having to do with FK
    cout << "Results before gpu" << endl;
    std::vector<dart::SE3> transfs_before;
    for (int i = 0; i < tracker.getModel(0).getNumFrames(); ++i) {
        // frame to model = model to frame in python
        cout << tracker.getModel(0).getTransformFrameToModel(i) << endl << endl;
        transfs_before.push_back(tracker.getModel(0).getTransformFrameToModel(i));
        // gpu version is getDeviceTransformsModelToFrame
        // joint angles are pose.getDeviceReducedArticulation
        // which axis to use can be found from getDeviceJointAxis
        // prism/revolute can be determined by axis (??)
        // can then calculate using loop over things
        // how to calculate collisions - mesh models aren't very nice
        // mirroredmodel::setarticulation
        //getTransformsParentJointToFrame

        // want something passed the pointer to the list of transformations and then the joint configurations
        // will then allocate and create the SE3 things for the joint configs and pass things to the gpu
    }

    // PANGOLIN SLIDER STUFF
    std::vector<pangolin::Var<float> * *> poseVars;

    pangolin::Var<bool> sliderControlled("pose.sliderControl",false,true);
    for (int m=0; m<tracker.getNumModels(); ++m) {

        const int dimensions = tracker.getModel(m).getPoseDimensionality();

        int nadd = 0;
        if (m == 0) nadd = 6;
        pangolin::Var<float> * * vars = new pangolin::Var<float> *[dimensions+nadd];
        poseVars.push_back(vars);
        poseVars[m][0] = new pangolin::Var<float>(dart::stringFormat("pose.%d x",m),0,-0.5,0.5);
        poseVars[m][1] = new pangolin::Var<float>(dart::stringFormat("pose.%d y",m),0,-0.5,0.5);
        poseVars[m][2] = new pangolin::Var<float>(dart::stringFormat("pose.%d z",m),0.3,0.5,1.5);
        poseVars[m][3] = new pangolin::Var<float>(dart::stringFormat("pose.%d wx",m),    0,-M_PI,M_PI);
        poseVars[m][4] = new pangolin::Var<float>(dart::stringFormat("pose.%d wy",m),    0,-M_PI,M_PI);
        poseVars[m][5] = new pangolin::Var<float>(dart::stringFormat("pose.%d wz",m), M_PI,-M_PI,M_PI);

        const dart::Pose & pose = tracker.getPose(m);
        for (int i=0; i<pose.getReducedArticulatedDimensions(); ++i) {
            //cout << "Pose:" << pose.getReducedMin(i) << " " << pose.getReducedMax(i) << endl;
            poseVars[m][i+6] = new pangolin::Var<float>(dart::stringFormat("pose.%d %s",m,pose.getReducedName(i).c_str()),0,pose.getReducedMin(i),pose.getReducedMax(i));
        }

        if (m == 0)
        {
            cout << "Here" << endl;
            poseVars[m][dimensions+0] = new pangolin::Var<float>(dart::stringFormat("pose.camx"),0,-1.5,1.5);
            poseVars[m][dimensions+1] = new pangolin::Var<float>(dart::stringFormat("pose.camy"),0.2,-1.5,1.5);
            poseVars[m][dimensions+2] = new pangolin::Var<float>(dart::stringFormat("pose.camz"),2.75,0,5);
            poseVars[m][dimensions+3] = new pangolin::Var<float>(dart::stringFormat("pose.camrotx"), M_PI/8,-M_PI,M_PI);
            poseVars[m][dimensions+4] = new pangolin::Var<float>(dart::stringFormat("pose.camroty"), M_PI/2,-M_PI,M_PI);
            poseVars[m][dimensions+5] = new pangolin::Var<float>(dart::stringFormat("pose.camrotz"),-M_PI/2,-M_PI,M_PI);
        }
    }

    // ================

    // Create a model view matrix
    //pangolin::OpenGlMatrix glM = pangolin::OpenGlMatrix::Translate(0,0,3) * pangolin::OpenGlMatrix::RotateX(-M_PI/2) * pangolin::OpenGlMatrix::RotateY(M_PI) * pangolin::OpenGlMatrix::RotateZ(M_PI/2) ;
    //pangolin::OpenGlMatrix glM = pangolin::OpenGlMatrix::Translate(camera_center[0],camera_center[1],camera_center[2]) *
    //        pangolin::OpenGlMatrix::RotateX(M_PI/8) * pangolin::OpenGlMatrix::RotateZ(-M_PI/2) * pangolin::OpenGlMatrix::RotateY(M_PI/2);
    const int dimensions = tracker.getModel(0).getPoseDimensionality();
    pangolin::OpenGlMatrix glM = pangolin::OpenGlMatrix::Translate(*poseVars[0][dimensions+0],
                                                                   *poseVars[0][dimensions+1],
                                                                   *poseVars[0][dimensions+2]) *
                                pangolin::OpenGlMatrix::RotateX(*poseVars[0][dimensions+3]) *
                                pangolin::OpenGlMatrix::RotateZ(*poseVars[0][dimensions+5]) *
                                pangolin::OpenGlMatrix::RotateY(*poseVars[0][dimensions+4]);
    Eigen::Matrix4f modelView  = glM;

    // Create a renderer
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

    for(int i = 0; i < tracker.getModel(0).getNumJoints(); i++)
    {
        int f = tracker.getModel(0).getJointFrame(i);
        for(int k = 0; k < meshFrameids.size(); k++)
        {
            if (f == meshFrameids[k])
            {
                cout << "Joint: " << tracker.getModel(0).getJointName(i) << " has mesh id: " << k << endl;
            }
        }
    }

    // ------------------- main loop ---------------------
    float *depth_image = new float[glWidth * glHeight];
    float *vmapwmeshid_image = new float[glWidth * glHeight * 4];
    float *vmap = new float[glWidth * glHeight * 3];
    float *vmap2 = new float[glWidth * glHeight * 3];

    GLuint pointCloudVbo;
    glGenBuffersARB(1,&pointCloudVbo);
    glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudVbo);
    glBufferDataARB(GL_ARRAY_BUFFER_ARB,glWidth*glHeight*sizeof(float)*3,vmap,GL_DYNAMIC_DRAW_ARB);

    for (int pangolinFrame=1; !pangolin::ShouldQuit(); ++pangolinFrame)
    {
        /// =====
        /// Update mesh vertices based on new pose
        for (int i = 0; i < meshVertices.size(); i++)
        {
            // Get the SE3 transform for the frame
            int m = meshModelids[i];
            int f = meshFrameids[i];
            const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);

            // Transform the canonical vertices based on the new transform
            for(int j = 0; j < meshVertices[i].size(); ++j)
            {
                transformedMeshVertices[i][j] = tfm * meshVertices[i][j];
            }

            // Upload to pangolin vertex buffer
            meshVertexAttributeBuffers[i][0]->Upload(transformedMeshVertices[i].data(), transformedMeshVertices[i].size()*sizeof(float3));
            meshVertexAttributeBuffers[i][1]->Upload(meshVerticesWMeshID[i].data(), meshVerticesWMeshID[i].size()*sizeof(float4)); // Upload the canonical vertices with mesh ids (fixed)
        }

        /// =====
        // Render a depth image
        glEnable(GL_DEPTH_TEST);
        renderer.renderMeshes<l2s::RenderDepth>(meshVertexAttributeBuffers, meshIndexBuffers);

        // Get a float image and save as png to disk
        renderer.texture<l2s::RenderDepth>().Download(depth_image);

        // Convert the depth data from "m" to "0.1 mm" range and store it as a 16-bit unsigned short
        // 2^16 = 65536  ;  1/10 mm = 1/10 * 1e-3 m = 1e-4 m  ;  65536 * 1e-4 = 6.5536
        // We can represent depth from 0 to +6.5536 m using this representation (enough for our data)
        cv::Mat depth(glHeight, glWidth, CV_32F, depth_image);
        cv::Mat depth_ushort;
        depth.convertTo(depth_ushort, CV_16UC1, 1e4); // 0.1 mm resolution and round off to *nearest* unsigned short
        cv::imwrite("depth.png", depth_ushort); // Save depth image
        cv::imshow("Colorized Depth Image", colorize_depth(depth_ushort));

        /// =====
        // Render a vertex map with the mesh ids
        renderer.renderMeshes<l2s::RenderVertMapWMeshID>(meshVertexAttributeBuffers, meshIndexBuffers);

        // Get the vertex image with mesh id
        renderer.texture<l2s::RenderVertMapWMeshID>().Download(vmapwmeshid_image, GL_RGBA, GL_FLOAT);

        // Convert to a vertex image
        cv::Mat vmapwmeshid(glHeight, glWidth, CV_32FC4, vmapwmeshid_image);
        cv::Mat channels[4];
        cv::split(vmapwmeshid, channels);
        cv::imshow("Mesh IDs", colorize_depth(channels[3]));
        cv::imshow("Local coords x", colorize_depth(channels[0]));
        cv::imshow("Local coords y", colorize_depth(channels[1]));
        cv::imshow("Local coords z", colorize_depth(channels[2]));

        // Compute the depth map based on the link transform and see if it matches what the depth map has
        cv::Mat newdepth(glHeight, glWidth, CV_32F, cv::Scalar(0));
        int ct = 0;
        for(int i = 0; i < newdepth.rows; i++)
        {
            for(int j = 0; j < newdepth.cols; j++)
            {
                // Get the mesh vertex in local co-ordinate frame
                cv::Vec4f vec = vmapwmeshid.at<cv::Vec4f>(i,j);
                if (vec[3] == 0) // Background
                {
                    newdepth.at<float>(i,j) = 0; // no valid depth
                    vmap[ct++] = 0;
                    vmap[ct++] = 0;
                    vmap[ct++] = 0;
                    continue;
                }

                // Get mesh values
                float3 localmesh = make_float3(vec[0], vec[1], vec[2]);
                assert((vec[3] - (int)vec[3]) < 1e-4);
                int meshid = (int) std::round(vec[3]) - 1; // Reduce by 1 to get the ID

                // Get the SE3 transform for the frame
                int m = meshModelids[meshid];
                int f = meshFrameids[meshid];
                const dart::SE3 tfm = tracker.getModel(m).getTransformFrameToModel(f);
                float3 modelmesh = tfm * localmesh;

                vmap[ct++] = modelmesh.x;
                vmap[ct++] = modelmesh.y;
                vmap[ct++] = modelmesh.z;

                // Transform to camera frame
                Eigen::Vector4f vec1(modelmesh.x, modelmesh.y, modelmesh.z, 1.0);
                Eigen::Vector4f cameramesh = modelView * vec1;
                newdepth.at<float>(i,j) = cameramesh(2); // Get "Z" value in camera frame of reference
            }
        }
        cv::imshow("New depth", colorize_depth(newdepth));
        cv::Mat newdepth_ushort;
        newdepth.convertTo(newdepth_ushort, CV_16UC1, 1e4); // 0.1 mm resolution and round off to *nearest* unsigned short
        cv::imwrite("newdepth.png", newdepth_ushort); // Save depth image


        cv::imshow("Depth diff", colorize_depth(newdepth - depth));
        cv::Mat depthdiff_ushort,depthdiff;
        cv::absdiff(newdepth,depth,depthdiff);
        depthdiff.convertTo(depthdiff_ushort, CV_16UC1, 1e4); // 0.1 mm resolution and round off to *nearest* unsigned short
        cv::imwrite("depthdiff.png", depthdiff_ushort); // Save depth image

        // Min/max
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(newdepth - depth, &minVal, &maxVal, &minLoc, &maxLoc );
        cout << "Min: " << minVal << " Max: " << maxVal << " Avg abs: " <<  cv::mean(cv::abs(newdepth - depth)) << endl;

        /// OpenCV wait
        cv::waitKey(2);

        /////////////////////////////////////////////////////////////////////////////////////

        // Create a model view matrix
        const int dimensions = tracker.getModel(0).getPoseDimensionality();
        pangolin::OpenGlMatrix glM = pangolin::OpenGlMatrix::Translate(*poseVars[0][dimensions+0],
                                                                       *poseVars[0][dimensions+1],
                                                                       *poseVars[0][dimensions+2]) *
                                    pangolin::OpenGlMatrix::RotateX(*poseVars[0][dimensions+3]) *
                                    pangolin::OpenGlMatrix::RotateZ(*poseVars[0][dimensions+5]) *
                                    pangolin::OpenGlMatrix::RotateY(*poseVars[0][dimensions+4]);
        Eigen::Matrix4f modelView  = glM;

        // Create a renderer
        renderer.setModelViewMatrix(modelView);

        // update pose based on sliders
        if (sliderControlled) {
            for (int m=0; m<tracker.getNumModels(); ++m) {
                for (int i=0; i<tracker.getPose(m).getReducedArticulatedDimensions(); ++i) {
                    tracker.getPose(m).getReducedArticulation()[i] = *poseVars[m][i+6];
                }
                tracker.getPose(m).setTransformModelToCamera(dart::SE3Fromse3(dart::se3(*poseVars[m][0],*poseVars[m][1],*poseVars[m][2],0,0,0))*
                        dart::SE3Fromse3(dart::se3(0,0,0,*poseVars[m][3],*poseVars[m][4],*poseVars[m][5])));
                tracker.updatePose(m);
            }
        }

        /////////////////////////////////////////////////////////////////////////////////////

        // Update pose based on the saved file
        if (pangolinFrame < joint_positions.size())
        {
            for(int k = 0; k < joint_names.size(); k++)
            {
                if (joint_name_to_pose_dim.find(joint_names[k]) != joint_name_to_pose_dim.end())
                {
                    int pose_dim = joint_name_to_pose_dim[joint_names[k]];
                    tracker.getPose(0).getReducedArticulation()[pose_dim] = joint_positions[pangolinFrame][k];
                }
            }
        }

        // Update the pose in the tracker model
        tracker.updatePose(0);

        // Find if pose is causing self-collisions
        // NOTE: Had to flip "FrameToModel" and "ModelToFrame" before passing it in to the countSelfIntersections function!
        // Seems like a naming bug in many of the GPU functions
        int numSelfIntersections = dart::countSelfIntersections(tracker.getDeviceCollisionCloud(0),
                                                                tracker.getCollisionCloudSize(0),
                                                                tracker.getModel(0).getDeviceTransformsFrameToModel(),
                                                                tracker.getModel(0).getDeviceTransformsModelToFrame(),
                                                                tracker.getModel(0).getDeviceSdfFrames(),
                                                                tracker.getModel(0).getDeviceSdfs(),
                                                                tracker.getModel(0).getNumSdfs(),
                                                                tracker.getDeviceIntersectionPotentialMatrix(0));
        cout << "Number of self intersections: " << numSelfIntersections << "/" << tracker.getCollisionCloudSize(0) << endl;

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                                                                                      //
        // Render this frame                                                                                    //
        //                                                                                                      //
        //////////////////////////////////////////////////////////////////////////////////////////////////////////

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

        /*
        /// === Get the vertex map and show it as a 3D point cloud
        // set up VBO to display point cloud

        glPointSize(4.0f);
        glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudVbo);
        glBufferDataARB(GL_ARRAY_BUFFER_ARB,glWidth*glHeight*sizeof(float)*3,vmap,GL_DYNAMIC_DRAW_ARB);

        glEnableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, 0);

        glDrawArrays(GL_POINTS,0,glWidth*glHeight);
        glBindBuffer(GL_ARRAY_BUFFER_ARB,0);

        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);

        glPointSize(1.0f);
        */

        /*
        // Show collision clouds
        glPointSize(10);
        glColor3f(0,0,1.0f);
        glDisable(GL_LIGHTING);
        glBegin(GL_POINTS);
        for (int m=0; m<tracker.getNumModels(); ++m) {
            const float4 * collisionCloud = tracker.getCollisionCloud(m);
            for (int i=0; i<tracker.getCollisionCloudSize(m); ++i) {
                int grid = round(collisionCloud[i].w);
                int frame = tracker.getModel(m).getSdfFrameNumber(grid);
                float4 v = tracker.getModel(m).getTransformModelToCamera()*
                           tracker.getModel(m).getTransformFrameToModel(frame)*
                           make_float4(make_float3(collisionCloud[i]),1.0);
                glVertex3fv((float*)&v);
            }
        }
        glEnd();
        glEnable(GL_LIGHTING);

        glPointSize(1);
        glColor3f(1,1,1);
        */

        /////////////////
        glPopMatrix();
        glDisable(GL_LIGHTING);
        glColor4ub(255,255,255,255);

        for (int m=0; m<tracker.getNumModels(); ++m) {
            for (int i=0; i<tracker.getPose(m).getReducedArticulatedDimensions(); ++i) {
                *poseVars[m][i+6] = tracker.getPose(m).getReducedArticulation()[i];
            }
            dart::SE3 T_cm = tracker.getPose(m).getTransformModelToCamera();
            *poseVars[m][0] = T_cm.r0.w; T_cm.r0.w = 0;
            *poseVars[m][1] = T_cm.r1.w; T_cm.r1.w = 0;
            *poseVars[m][2] = T_cm.r2.w; T_cm.r2.w = 0;
            dart::se3 t_cm = dart::se3FromSE3(T_cm);
            *poseVars[m][3] = t_cm.p[3];
            *poseVars[m][4] = t_cm.p[4];
            *poseVars[m][5] = t_cm.p[5];
        }

        /// Finish frame
        pangolin::FinishFrame();
    }

    for (int m=0; m<tracker.getNumModels(); ++m) {
        for (int i=0; i<tracker.getPose(m).getReducedDimensions(); ++i) {
            delete poseVars[m][i];
        }
        delete [] poseVars[m];
    }

    for (uint i=0; i<sizeVars.size(); ++i) {
        delete sizeVars[i];
    }

    for(size_t i = 0; i < meshVertexAttributeBuffers.size(); i++)
        for(size_t j = 0; j < meshVertexAttributeBuffers[i].size(); j++)
            free(meshVertexAttributeBuffers[i][j]);
    free(depth_image);

    return 0;
}



/*
 *
 *
    std::vector<std::vector<pangolin::GlBuffer *> > vertexAttributeBuffers;
    std::vector<pangolin::GlBuffer> indexBuffers;
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

                // Get their vertices and transform them using the given frame to model transform
                std::vector<float3> verts(mesh.nVertices), transfVerts(mesh.nVertices);
                for(int i = 0; i < verts.size(); ++i)
                {
                    verts[i] = mesh.vertices[i];
                    transfVerts[i] = tfm * mesh.vertices[i];
                }
                //vertices.push_back(verts);
                //transformedVertices.push_back(transfVerts);

                // For each mesh, get the vertices
                pangolin::GlBuffer *vertices = new pangolin::GlBuffer(pangolin::GlBufferType::GlArrayBuffer, transfVerts.size(), GL_FLOAT, 3);
                vertices->Upload(transfVerts.data(), transfVerts.size()*sizeof(float3));

                // For each mesh, get the normals
                pangolin::GlBuffer *normals = new pangolin::GlBuffer(pangolin::GlBufferType::GlArrayBuffer, mesh.nVertices, GL_FLOAT, 3);
                normals->Upload(mesh.normals, transfVerts.size()*sizeof(float3));

                // Get the attributes together
                std::vector<pangolin::GlBuffer *> attributes;
                attributes.push_back(vertices); attributes.push_back(normals);
                vertexAttributeBuffers.push_back(attributes);

                // For each mesh, get the faces (do nFaces * 3)
                indexBuffers.resize(indexBuffers.size()+1);
                indexBuffers.back().Reinitialise(pangolin::GlBufferType::GlElementArrayBuffer, mesh.nFaces*3, GL_INT, 3, GL_DYNAMIC_DRAW);
                indexBuffers.back().Upload(mesh.faces, mesh.nFaces*sizeof(int3));
            }
        }
    }

    /// =====
    // Render a color image
    renderer.renderMeshes<l2s::RenderColor>(vertexAttributeBuffers, indexBuffers);

    // Get a float image and save as png to disk
    uchar *color_image = new uchar[glWidth * glHeight*4];
    renderer.texture<l2s::RenderColor>().Download(color_image, GL_RGBA, GL_UNSIGNED_BYTE);

    /// =====
    // Render a depth image
    renderer.renderMeshes<l2s::RenderDepth>(vertexAttributeBuffers, indexBuffers);

    // Get a float image and save as png to disk
    float *depth_image = new float[glWidth * glHeight];
    renderer.texture<l2s::RenderDepth>().Download(depth_image);

    /// =====
    // Convert the depth data from "m" to "0.1 mm" range and store it as a 16-bit unsigned short
    // 2^16 = 65536  ;  1/10 mm = 1/10 * 1e-3 m = 1e-4 m  ;  65536 * 1e-4 = 6.5536
    // We can represent depth from 0 to +6.5536 m using this representation (enough for our data)
    cv::Mat depth(glHeight,glWidth,CV_32F,depth_image);
    cv::Mat depth_ushort;
    depth.convertTo(depth_ushort, CV_16UC1, 1e4); // 0.1 mm resolution and round off to *nearest* unsigned short
    cv::imwrite("depth.png", depth_ushort); // Save depth image

    // Save color image
    cv::Mat color(glHeight,glWidth,CV_8UC4,color_image);
    cv::imwrite("color.png", color); // Save depth image

    /// =====
    // Show images
    cv::imshow("Colorized Depth Image", colorize_depth(depth_ushort));
    cv::imshow("Color Image", color);
*/
