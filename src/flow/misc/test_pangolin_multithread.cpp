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

// Pointers to data
cv::Mat color, depth;
boost::mutex pangolin_mutex;
bool update_pangolin;

void run_pangolin()
{
    // Create pangolin window
    pangolin::CreateWindowAndBind("Display iamge and point cloud",1280,960);
    int imageWidth=640,imageHeight=480;
    printf("Init Pangolin  \n");
    pangolin::OpenGlRenderState camState(pangolin::ProjectionMatrixRDF_TopLeft(imageWidth,imageHeight,525,525,320,240,0.01,1000.0));
    pangolin::Handler3D handler(camState);
    pangolin::View & img1 = pangolin::Display("img1").SetAspect(imageWidth/static_cast<float>(imageHeight));
    pangolin::View & estDisp = pangolin::Display("estimated").SetAspect(imageWidth/static_cast<float>(imageHeight));
    pangolin::View & GTDisp = pangolin::Display("gt").SetAspect(imageWidth/static_cast<float>(imageHeight));
    pangolin::View & pcDisp = pangolin::Display("pointcloud").SetAspect(imageWidth/static_cast<float>(imageHeight)).SetHandler(&handler);
    pangolin::Display("multi").SetBounds(0,0.95,pangolin::Attach::Pix(200),1).SetLayout(pangolin::LayoutEqual)
            .AddDisplay(img1)
            .AddDisplay(pcDisp)
            .AddDisplay(GTDisp)
            .AddDisplay(estDisp);
    pangolin::GlTexture img1Tex(imageWidth,imageHeight);
    pangolin::GlTexture estTex(imageWidth,imageHeight);
    pangolin::GlTexture gtTex(imageWidth,imageHeight);

    while (!pangolin::ShouldQuit())
    {
        // General stuff
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        if (pangolin::HasResized())
        {
            pangolin::DisplayBase().ActivateScissorAndClear();
        }

        // Update in the mutex
        boost::mutex::scoped_lock update_lock(pangolin_mutex);
        // Updating pangolin with new data from disk
        //printf("Updating GUI \n");

        // Upload the color
        if (!color.empty())
        {
            img1Tex.Upload( color.data, GL_BGR,GL_UNSIGNED_BYTE);
        }

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
            float mmTom = 0.001;
            for (int i=0; i < depth.rows; i++)
            {
                for (int j=0; j < depth.cols; j++)
                {
                    float x,y,z,z1;
                    x = (j - 320) / 525.0;
                    y = (i - 240) / 525.0;
                    z1 = (float)  depth.at<ushort>(i,j) * mmTom ;
                    float ppert1[4] = {x*z1,y*z1,z1,1.0};
                    glColor3ub(0,255,0);
                    glVertex3fv((float *)ppert1);
                }
            }

            // Finish
            glEnd();
            glDisable(GL_DEPTH_TEST);
        }

        // Finished updating
        //printf("Finished updating GUI \n");
        update_lock.unlock();

        // Render
        glColor3ub(255,255,255);
        img1.ActivateScissorAndClear();
        img1Tex.RenderToViewportFlipY();
        GTDisp.ActivateScissorAndClear();
        gtTex.RenderToViewportFlipY();
        estDisp.ActivateScissorAndClear();
        estTex.RenderToViewportFlipY();
        // Finish rendering
        pangolin::FinishFrame();
     }
}

int main(int argc, char** argv)
{
    // Get depth and color data
    std::string color_name = std::string (argv[1]);
    std::string depth_name = std::string (argv[2]);

    // Create pangolin GUI and run it in a separate thread
    boost::shared_ptr<boost::thread> pangolin_gui_thread;
    printf("Creating pangolin thread \n");
    update_pangolin = false; // Do not update point clouds yet
    pangolin_gui_thread.reset(new boost::thread(run_pangolin));

    // Sleep for a bit
    sleep(5);

    // Lock mutex
    boost::mutex::scoped_lock update_lock(pangolin_mutex);

    // Load images
    printf("Loading images from disk \n");
    color = cv::imread(color_name); // Load color image from disk
    depth = cv::imread(depth_name, CV_LOAD_IMAGE_ANYDEPTH); // Load depth image from disk
    printf("Finished Loading images from disk \n");

    // Set flag
    update_pangolin = true; // Ask pangolin to update

    // Unlock
    update_lock.unlock();

    // Wait till GUI exits
    pangolin_gui_thread->join();

    return 0;
}
