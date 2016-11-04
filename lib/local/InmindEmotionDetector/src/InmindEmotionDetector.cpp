#include "InmindEmotionDetector.h"

using namespace InmindDemo;

InmindEmotionDetector::InmindEmotionDetector(string s):root_path(path(s).parent_path().string()),face_model(root_path +"/model/main_clnf_general.txt"),face_analyser(vector<cv::Vec3d>(), 0.7, 112, 112, root_path +"/AU_predictors/AU_all_best.txt", root_path +"/model/tris_68_full.txt")
{
	vector<string> arguments;
	arguments.push_back(s);
	int isSuccess = detector.initialize_params(arguments);
	det_parameters = LandmarkDetector::FaceModelParameters(arguments);
	det_parameters.track_gaze = true;
	det_parameters.quiet_mode = true;

	string conf_thres_path = "emotion_models/confusion_threshold.txt";
	threshold_confusion = get_confusion_thres(arguments[0], conf_thres_path);
	cout << "Load Confusion Threshold: " << threshold_confusion << endl;
	string surp_thres_path = "emotion_models/surprise_threshold.txt";
	threshold_surprise = get_surprise_thres(arguments[0], surp_thres_path);
	cout << "Load Surprise Threshold: " << threshold_surprise << endl;

}

vector<double> InmindEmotionDetector::DetectEmotion(Mat frame, double time_stamp)
{
	result_emotions.clear();
	if (!isCfSet)
	{
		detector.set_cf(frame);
	}
	grayscale_image = detector.get_gray(frame);
	detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model, det_parameters);
	if (detection_success)
	{
		// Do face alignment
		face_analyser.AddNextFrame(frame, face_model, time_stamp, online, !det_parameters.quiet_mode);

		// Get features
		face_analyser.GetLatestHOG(hog_descriptor, detector.num_hog_rows, detector.num_hog_cols);
		face_analyser.GetGeomDescriptor(geom_descriptor);

		// Do predictions
		face_analyser.PredictAUs(hog_descriptor, geom_descriptor, face_model, online);

		// Get prediction results
		current_AusReg = face_analyser.GetCurrentAUsReg();

		score_confusion = detector.predict_confusion(current_AusReg);
		score_surprise = detector.predict_surprise(current_AusReg);
	}
	else
	{
		score_confusion = 0;
		score_surprise = 0;
	}

	if(prev_confusion < 0)
	{
		prev_confusion = score_confusion;
	}else
	{
		score_confusion = alpha * prev_confusion + (1-alpha) * score_confusion;
		prev_confusion = score_confusion;
	}
	if(prev_surprise < 0)
	{
		prev_surprise = score_surprise;
	}else
	{
		score_surprise = alpha * prev_surprise + (1-alpha)* score_surprise;
		prev_surprise = score_surprise;
	}

	if (score_confusion >= threshold_confusion)
	{
		decision_confusion = 1.0;

	}
	else
	{
		decision_confusion = 0;
	}

	if (score_surprise >= threshold_surprise)
	{
		decision_surprise = 1.0;
	}
	else
	{
		decision_surprise = 0;
	}
	result_emotions.push_back(score_confusion);
	result_emotions.push_back(score_surprise);
	result_emotions.push_back(decision_confusion);
	result_emotions.push_back(decision_surprise);

	return result_emotions;
}
void InmindEmotionDetector::visualize_emotions(Mat &frame)
{
	string face_detected = "false";
	string confusion_sco;
	string surprise_sco;
	string confusion_dec = "No Confusion Detected";
	string surprise_dec = "No Surprise Detected";

	if (detection_success)
	{
		face_detected = "true";
	}
	if (score_confusion >= threshold_confusion)
	{
		confusion_dec = "Confusion Detected";
	}
	if (score_surprise >= threshold_surprise)
	{
		surprise_dec = "Surprise Detected";
	}

	face_detected = "Face Detected: " + face_detected;
	confusion_sco = "Confusion Score: " + to_string(score_confusion);
	surprise_sco = "Surprise Score: " + to_string(score_surprise);
	confusion_dec = "Confusion Decision: " + confusion_dec;
	surprise_dec = "Surprise Decision: " + surprise_dec;

	putText(frame, face_detected, cvPoint(20, 30),
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	putText(frame, confusion_sco, cvPoint(20, 50),
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	putText(frame, confusion_dec, cvPoint(20, 70),
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	putText(frame, surprise_sco, cvPoint(20, 90),
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	putText(frame, surprise_dec, cvPoint(20, 110),
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);

}

double InmindEmotionDetector::get_confusion_thres(string root, string thres_path)
{
	string data_path;
	double threshold = 1.0;
	if (boost::filesystem::exists(path(thres_path)))
	{
		data_path = thres_path;
	}
	else
	{
		path loc = path(root).parent_path() / thres_path;
		data_path = loc.string();

		if (!exists(loc))
		{
			cout << "Can't find threshold files, exiting" << endl;
			return threshold;
		}
	}
	std::ifstream ifile(data_path, std::ios::in);
	ifile >> threshold;

	return threshold;
}
double InmindEmotionDetector::get_surprise_thres(string root, string thres_path)
{
	string data_path;
	double threshold = 1.0;
	if (boost::filesystem::exists(path(thres_path)))
	{
		data_path = thres_path;
	}
	else
	{
		path loc = path(root).parent_path() / thres_path;
		data_path = loc.string();
		if (!exists(loc))
		{
			cout << "Can't find threshold files, exiting" << endl;
			return threshold;
		}
	}
	std::ifstream ifile(data_path, std::ios::in);
	ifile >> threshold;

	return threshold;
}
