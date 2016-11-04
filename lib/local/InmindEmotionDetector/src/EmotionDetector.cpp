#include "EmotionDetector.h"

using namespace EmotionRecognition;
using namespace cv;

// initialize the arguments
vector<string> EmotionDetector::get_arguments(int argc, char **argv)
{
	vector<string> arguments;

	// First argument is reserved for the name of the executable
	for (int i = 0; i < argc; i++) {
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}


// Initialize parameters, fail if return 0, succeed if return 1
int EmotionDetector::initialize_params(vector<string> &arguments)
{
	// Get camera parameters
	LandmarkDetector::get_camera_params(d, fx, fy, cx, cy, arguments);
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// By default output all parameters, but these can be turned off to get smaller files or slightly faster processing times
	// use -no2Dfp, -no3Dfp, -noMparams, -noPose, -noAUs, -noGaze to turn them off
	get_output_feature_params(arguments);

	if (boost::filesystem::exists(path("model/tris_68_full.txt")))
	{
		tri_loc = "model/tris_68_full.txt";
	}
	else
	{
		path loc = path(arguments[0]).parent_path() / "model/tris_68_full.txt";
		tri_loc = loc.string();
		if (!exists(loc))
		{
			cout << "Can't find triangulation files, exiting" << endl;
			return 0;
		}
	}

	string au_loc_local;
	if (dynamic)
	{
		au_loc_local = "AU_predictors/AU_all_best.txt";
	}
	else
	{
		au_loc_local = "AU_predictors/AU_all_static.txt";
	}

	if (boost::filesystem::exists(path(au_loc_local)))
	{
		au_loc = au_loc_local;
	}
	else
	{
		path loc = path(arguments[0]).parent_path() / au_loc_local;

		if (exists(loc))
		{
			au_loc = loc.string();
		}
		else
		{
			cout << "Can't find AU prediction files, exiting" << endl;
			return 0;
		}
	}
	return 1;
}

// if optical centers are not defined just use center of image
// Use a rough guess-timate of focal length
void EmotionDetector::set_cf(cv::Mat captured_image)
{
	if (cx_undefined)
	{
		cx = captured_image.cols / 2.0f;
		cy = captured_image.rows / 2.0f;
	}
	// Use a rough guess-timate of focal length
	if (fx_undefined)
	{
		fx = 500 * (captured_image.cols / 640.0);
		fy = 500 * (captured_image.rows / 480.0);

		fx = (fx + fy) / 2.0;
		fy = fx;
	}
}

// Convert BGR to gray images
cv::Mat_<uchar> EmotionDetector::get_gray(cv::Mat captured_image)
{
	cv::Mat_<uchar> grayscale_image;
	if (captured_image.channels() == 3)
	{
		cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
	}
	else
	{
		grayscale_image = captured_image.clone();
	}
	return grayscale_image;
}

// Work out the pose of the head from the tracked model
int EmotionDetector::get_pose_estimate(const LandmarkDetector::CLNF& face_model)
{
	if (use_world_coordinates)
	{
		pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);
		return 1;
	}
	else
	{
		pose_estimate = LandmarkDetector::GetCorrectedPoseCamera(face_model, fx, fy, cx, cy);
		return 1;
	}
	return 0;
}

int EmotionDetector::get_gaze_direction(const LandmarkDetector::FaceModelParameters det_parameters, bool detection_success,
	const LandmarkDetector::CLNF face_model)
{
	gazeDirection0 = cv::Point3f(0, 0, -1);
	gazeDirection1 = cv::Point3f(0, 0, -1);
	if (det_parameters.track_gaze && detection_success && face_model.eye_model)
	{
		FaceAnalysis::EstimateGaze(face_model, gazeDirection0, fx, fy, cx, cy, true);
		FaceAnalysis::EstimateGaze(face_model, gazeDirection1, fx, fy, cx, cy, false);
		return 1;
	}
	else
	{
		return 0;
	}
}

void EmotionDetector::get_output_feature_params(vector<string> &arguments)
{
	output_similarity_align.clear();
	output_hog_align_files.clear();

	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	string input_root = "";
	string output_root = "";

	// By default the model is dynamic
	dynamic = true;

	// First check if there is a root argument (so that videos and outputs could be defined more easilly)
	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-root") == 0)
		{
			input_root = arguments[i + 1];
			output_root = arguments[i + 1];
			i++;
		}
		if (arguments[i].compare("-inroot") == 0)
		{
			input_root = arguments[i + 1];
			i++;
		}
		if (arguments[i].compare("-outroot") == 0)
		{
			output_root = arguments[i + 1];
			i++;
		}
	}

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-simalign") == 0)
		{
			output_similarity_align.push_back(output_root + arguments[i + 1]);
			create_directory(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-hogalign") == 0)
		{
			output_hog_align_files.push_back(output_root + arguments[i + 1]);
			//create_directory_from_file(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-verbose") == 0)
		{
			verbose = true;
		}
		else if (arguments[i].compare("-rigid") == 0)
		{
			rigid = true;
		}
		else if (arguments[i].compare("-au_static") == 0)
		{
			dynamic = false;
		}
		else if (arguments[i].compare("-g") == 0)
		{
			grayscale = true;
			valid[i] = false;
		}
		else if (arguments[i].compare("-simscale") == 0)
		{
			sim_scale = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-simsize") == 0)
		{
			sim_size = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-no2Dfp") == 0)
		{
			output_2D_landmarks = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-no3Dfp") == 0)
		{
			output_3D_landmarks = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noMparams") == 0)
		{
			output_model_params = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noPose") == 0)
		{
			output_pose = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noAUs") == 0)
		{
			output_AUs = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noGaze") == 0)
		{
			output_gaze = false;
			valid[i] = false;
		}
	}

	for (int i = arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}
}

double EmotionDetector::predict_confusion(vector<pair<string, double> > current_AUsReg)
{
	double score = 0.0;
	int numOfAUs = 0;
	string AUName = "";
	double AUScore = 0.0;
	for (size_t i = 0; i < current_AUsReg.size(); i++)
	{
		AUName = current_AUsReg[i].first;
		AUScore = current_AUsReg[i].second;
		if (AUName == "AU04")
		{
			score += AUScore;
			numOfAUs++;
		}
	}
	score /= numOfAUs;
	return max(score, 0.0);
}
double EmotionDetector::predict_surprise(vector<pair<string, double> > current_AUsReg)
{
	double score = 0.0;
	int numOfAUs = 0;
	string AUName = "";
	double AUScore = 0.0;

	for (size_t i = 0; i < current_AUsReg.size(); i++)
	{
		AUName = current_AUsReg[i].first;
		AUScore = current_AUsReg[i].second;
		if (AUName == "AU01")
		{
			score += AUScore;
			numOfAUs++;
		}
		else if (AUName == "AU02")
		{
			score += AUScore;
			numOfAUs++;
		}
		else if (AUName == "AU05")
		{
			score += AUScore;
			numOfAUs++;
		}
		else if (AUName == "AU26")
		{
			score += AUScore;
			numOfAUs++;
		}
	}
	score /= numOfAUs;
	return max(score,0.0);
}

void EmotionDetector::output_HOG_frame(std::ofstream* hog_file, bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_rows, int num_cols)
{

	// Using FHOGs, hence 31 channels
	int num_channels = 31;

	hog_file->write((char*)(&num_cols), 4);
	hog_file->write((char*)(&num_rows), 4);
	hog_file->write((char*)(&num_channels), 4);

	// Not the best way to store a bool, but will be much easier to read it
	float good_frame_float;
	if (good_frame)
		good_frame_float = 1;
	else
		good_frame_float = -1;

	hog_file->write((char*)(&good_frame_float), 4);

	cv::MatConstIterator_<double> descriptor_it = hog_descriptor.begin();

	for (int y = 0; y < num_cols; ++y)
	{
		for (int x = 0; x < num_rows; ++x)
		{
			for (unsigned int o = 0; o < 31; ++o)
			{

				float hog_data = (float)(*descriptor_it++);
				hog_file->write((char*)&hog_data, 4);
			}
		}
	}
}

// visulizing the results
void EmotionDetector::visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model,
	const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0,
	cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{
	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);

		// Draw it in reddish if uncertain, blueish if certain
		LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);

		if (det_parameters.track_gaze && detection_success && face_model.eye_model)
		{
			FaceAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
		}
	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		this->fps_tracker = 10.0 / (double(t1 - this->t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)this->fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));

	if (!det_parameters.quiet_mode)
	{
		cv::namedWindow("tracking_result", 1);
		cv::imshow("tracking_result", captured_image);
	}
}
