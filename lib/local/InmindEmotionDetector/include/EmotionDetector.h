#pragma once
#pragma once
// System includes
#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

// Local includes
#include <LandmarkCoreIncludes.h>
#include <LandmarkDetectorFunc.h>
#include <LandmarkDetectorModel.h>

#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>

using namespace std;
using namespace boost::filesystem;
#define INFO_STREAM( stream ) cout << stream << endl

#define WARN_STREAM( stream ) cout << "Warning: " << stream << endl

#define ERROR_STREAM( stream ) cout << "Error: " << stream << endl

#define FATAL_STREAM( stream ) cout<< string( "Fatal error: " ) + stream <<endl

namespace EmotionRecognition {

	class EmotionDetector {
	public:

		// For visulizing the results, probably will not use in InMind project
		double fps_tracker = -1.0;
		int64 t0 = 0;

		// some initial parameters that can be overriden from command line
		vector<string> input_files;
		vector<string> depth_directories;
		vector<string> output_files;
		vector<string> tracked_videos_output;
		bool use_world_coordinates;

		// Grad camera parameters, if they are not defined (approximate values will be used)
		float fx = 0;
		float fy = 0;
		float cx = 0;
		float cy = 0;
		int d = 0;
		// If cx (optical axis center) is undefined, will use the image size/2 as an estimate
		bool cx_undefined = false;
		bool fx_undefined = false;

		// determine if input files are images or videos
		bool video_input = true;
		bool images_as_video = false;
		vector<vector<string> > input_image_files;

		// output feature parameters
		vector<string> output_similarity_align;
		vector<string> output_hog_align_files;
		double sim_scale = 0.7;
		int sim_size = 112;
		bool verbose = true;
		bool grayscale = false;
		bool rigid = false;
		bool dynamic = true; // Indicates if a dynamic AU model should be used (dynamic is useful if the video is long enough to include neutral expressions)
		int num_hog_rows;
		int num_hog_cols;

		// By default output all parameters, but these can be turned off to get smaller files or slightly faster processing times
		// use -no2Dfp, -no3Dfp, -noMparams, -noPose, -noAUs, -noGaze to turn them off
		bool output_2D_landmarks = false;
		bool output_3D_landmarks = false;
		bool output_model_params = true;
		bool output_pose = false;
		bool output_AUs = false;
		bool output_gaze = false;

		// used for image masking
		string tri_loc;
		string au_loc;

		// work out the pose of the head from the tracked model
		cv::Vec6d pose_estimate;

		// Gaze tracking, absolute gaze direction

		cv::Point3f gazeDirection0 = cv::Point3f(0, 0, -1);
		cv::Point3f gazeDirection1 = cv::Point3f(0, 0, -1);


		// initialization
		vector<string> get_arguments(int argc, char **argv);
		int initialize_params(vector<string> &arguments);
		void get_output_feature_params(vector<string> &arguments);

		void output_HOG_frame(std::ofstream* hog_file, bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_rows, int num_cols);
		void visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model,
			const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0,
			cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy);
		void set_cf(cv::Mat captured_image);
		cv::Mat_<uchar> get_gray(cv::Mat captured_image);
		int get_pose_estimate(const LandmarkDetector::CLNF& clnf_model);
		int get_gaze_direction(const LandmarkDetector::FaceModelParameters det_parameters, bool detection_sucess, const LandmarkDetector::CLNF face_model);
	
		double predict_confusion(vector<pair<string, double> > current_AUsReg);
		double predict_surprise(vector<pair<string, double> > current_AUsReg);
	};

}