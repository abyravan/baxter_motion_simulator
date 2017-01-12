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

// Eigen
#include <Eigen/Dense>

// Messages
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf2_msgs/TFMessage.h>

// CUDA
#include <cuda_runtime.h>
#include <vector_types.h>

// Read CSV stuff
#include "util/csv_util.h"

// Utility functions
#include <helperfuncs.h>

// Boost
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/foreach.hpp>

// Learn physics models
#include "learn_physics_models/LinkStates.h"

// Gazebo messages
#include <gazebo_msgs/ContactState.h>
#include <gazebo_msgs/ContactsState.h>

using namespace std;
namespace po = boost::program_options;

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

// ==== Read OBB data ==== //
struct OBB
{
    Eigen::Vector3f center;
    Eigen::Vector3f halfextents;
    Eigen::Matrix3f rotationmatrix;
};

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
    std::string pokefolder      = "";       // Folder to load saved poke images from. Flow vectors will be saved in these folders.
    std::string modelfolder     = "";       // Model folder

    //PARSE INPUT ARGUMENTS
    po::options_description desc("Allowed options",1024);
    desc.add_options()
        ("help", "produce help message")
        ("pokefolder", po::value<std::string>(&pokefolder), "Path to load data from. Will load/save data from all sub-directories")
        ("modelfolder", po::value<std::string>(&modelfolder), "Path to folder containing the models in the data")
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
    assert(!modelfolder.empty() && "Please pass in folder which has all the model xml files --modelfolder <path_to_folder>");

    /// == Get model names
    std::vector<std::string> dart_model_dirs = findAllDirectoriesInFolder(modelfolder);
    std::vector<std::string> model_names;
    for(std::size_t i = 0; i < dart_model_dirs.size(); i++)
    {
        // Get object name
        std::vector<std::string> words;
        boost::split(words, dart_model_dirs[i], boost::is_any_of("/"), boost::token_compress_on); // Split on "/"
        std::string modelName = words.back();
        model_names.push_back(modelName);
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

    // == Setup time values for depth images
    // Get the timestamps from the files
    std::vector<double> joint_timestamps = read_csv_data<double>(pokefolder + "/positions.csv", "time");

    // Start at initial joint angle timestamp, move in steps of 1/30 (30 Hz) till we reach end
    double t = joint_timestamps[0];
    std::vector<double> depth_timestamps;
    while(t <= joint_timestamps.back())
    {
        depth_timestamps.push_back(t);
        t += (1.0/30);
    }

    // Some stats
    int num_frames = depth_timestamps.size();

    /// == Read table OBB
    std::vector<OBB> obbs = read_obb_data(pokefolder + "/" + "tableobbdata.txt");
    assert(obbs.size() == 1 && "Table OBB not read correctly");
    OBB tableOBB = obbs[0];

    /// == Read meta data file, use it to post-process results to generate the event times
    /// Assume that this is a single object dataset
    // Meta data vars
    std::vector<std::string> object_names;
    int num_objects;
    double table_friction;
    std::string target_obj_name;

    // Read the meta data file
    std::ifstream ifs(pokefolder + "/metadata.txt");
    ifs >> num_objects;
    ifs >> table_friction;

    // Get target link name
    learn_physics_models::LinkStates::ConstPtr msg = linkstates_messages[0];
    for(std::size_t j = 0; j < msg->name.size(); ++j)
    {
        for(std::size_t k = 0; k < model_names.size(); k++)
        {
            if(msg->name[j].find(model_names[k] + "::link") != std::string::npos)
            {
                target_obj_name = model_names[k];
                std::cout << "Target object name: " << target_obj_name << std::endl;
            }
        }
    }

    // Set target link name and object names
    object_names.push_back(target_obj_name);

    /// == Rewrite metadata file
    std::ofstream ofs;
    ofs.open(pokefolder + "/metadata.txt");
    ofs << num_objects << std::endl;
    ofs << table_friction << std::endl;
    ofs << target_obj_name << std::endl;
    ofs << target_obj_name << std::endl;
    ofs << num_frames << std::endl; // Write number of depth files to meta data
    ofs.close();

    // Print metadata
    std::cout << std::endl << "Metadata: " << std::endl;
    std::cout << "Num objects: " << num_objects << ", Friction: " << table_friction << std::endl;
    std::cout <<  "Target obj name: " << target_obj_name << std::endl;

    // Common string = object name now (As there is only one object)
    std::string target_link_name = target_obj_name + "::link";
    std::string obj_common_str = target_link_name; // Common string in all object names

    // == Post process the experiment state (search for all objects)
    std::map<std::string, double> map_events_to_times = get_experiment_event_times(contactstates_messages,
                                                                                   linkstates_messages,
                                                                                   tableOBB,
                                                                                   obj_common_str,
                                                                                   target_link_name);
    // Save this data to disk
    std::string events_file_path = pokefolder + "/events.txt";
    save_event_time_data(map_events_to_times, depth_timestamps, events_file_path);
    std::cout << std::endl;

    // Close bag and exit
    bag.close();
}
