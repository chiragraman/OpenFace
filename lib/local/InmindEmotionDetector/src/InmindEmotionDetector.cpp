/**
* @Author: Chirag Raman <chirag>
* @Date:   2016-11-04T11:11:49-04:00
* @Email:  chirag.raman@gmail.com
* @Last modified by:   chirag
* @Last modified time: 2016-12-08T19:40:12-05:00
* @License: Copyright (C) 2016 Multicomp Lab. All rights reserved.
*/



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

vector<double> InmindEmotionDetector::process_frame(Mat frame,
													double time_stamp)
{
	// If optical centers are not defined just use center of image
	cx = frame.cols / 2.0f;
	cy = frame.rows / 2.0f;

	// Use a rough guess-timate of focal length
	fx = 500 * (frame.cols / 640.0);
	fy = 500 * (frame.rows / 480.0);

	fx = (fx + fy) / 2.0;
	fy = fx;

	//Reset data
	result_emotions.clear();
	current_AusReg.clear();
	Vec6d pose_estimate;
	vector<Point3f> gaze(2);
	vector<double> aus;

	if (!isCfSet)
	{
		detector.set_cf(frame);
	}
	grayscale_image = detector.get_gray(frame);
	detection_success =
		LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model,
												 det_parameters);
	if (detection_success)
	{
		// Gaze tracking, absolute gaze direction
		Point3f gaze_direction0(0, 0, -1);
		Point3f gaze_direction1(0, 0, -1);
		FaceAnalysis::EstimateGaze(face_model, gaze_direction0, fx, fy, cx, cy,
								   true);
		FaceAnalysis::EstimateGaze(face_model, gaze_direction1, fx, fy, cx, cy,
								   false);
		gaze[0] = gaze_direction0;
		gaze[1] = gaze_direction1;

		// Head pose estimation
		pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(
			face_model, fx, fy, cx, cy);

		// Do face alignment
		face_analyser.AddNextFrame(frame, face_model, time_stamp, online,
								   !det_parameters.quiet_mode);

		// Get features
		face_analyser.GetLatestHOG(hog_descriptor, detector.num_hog_rows,
								   detector.num_hog_cols);
		face_analyser.GetGeomDescriptor(geom_descriptor);

		// Do predictions
		face_analyser.PredictAUs(hog_descriptor, geom_descriptor, face_model,
								 online);

		// Get prediction results
		current_AusReg = face_analyser.GetCurrentAUsReg();

		// Extract the AU intensities
		aus.reserve(current_AusReg.size());
		for(size_t it = 0; it < current_AusReg.size(); ++it) {
			aus.push_back(current_AusReg[it].second);
		}

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

	decision_confusion = score_confusion >= threshold_confusion ? 1.0 : 0
	decision_surprise = score_surprise >= threshold_surprise ? 1.0 : 0

	result_emotions.push_back(score_confusion);
	result_emotions.push_back(decision_confusion);
	result_emotions.push_back(score_surprise);
	result_emotions.push_back(decision_surprise);

	return {detection_success, pose_estimate, gaze, aus, result_emotions};
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
