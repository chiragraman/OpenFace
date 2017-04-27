/**
* @Author: Chirag Raman <chirag>
* @Date:   2016-11-04T11:11:49-04:00
* @Email:  chirag.raman@gmail.com
* @Last modified by:   chirag
* @Last modified time: 2016-12-08T20:40:05-05:00
* @License: Copyright (C) 2016 Multicomp Lab. All rights reserved.
*/



#ifndef InmindDemo_InmindEmotionDetector_H_
#define InmindDemo_InmindEmotionDetector_H_

#include "EmotionDetector.h"

using namespace std;
using namespace boost::filesystem;
using namespace EmotionRecognition;
using namespace cv;

namespace InmindDemo {

struct FrameData{
	bool success;
	vector<double> pose_estimate;
	vector<vector<float>> gaze_estimate;
	vector<double> aus;
	vector<double> emotions;
};

class InmindEmotionDetector {
public:
	InmindEmotionDetector(string s);
	FrameData process_frame(Mat frame, double time_stamp);
	void visualize_emotions(Mat &frame);

private:
	string root_path;
	EmotionDetector detector;
	LandmarkDetector::FaceModelParameters det_parameters;
	LandmarkDetector::CLNF face_model;
	FaceAnalysis::FaceAnalyser face_analyser;

	bool detection_success = false;
	bool online = true;

	// Used for post-processing of AU detection
	double time_stamp = 0;
	int frame_count = -1;

	// Features
	Mat_<double> hog_descriptor;
	Mat_<double> geom_descriptor;

	// Thresholds for confusion and surprise
	double threshold_confusion;
	double threshold_surprise;

    // Store old predictions
    vector<double> previous_emotions;
	double alpha = 0.15;

	// Set cf flag
	bool isCfSet = false;

	// temporary data
	Mat_<uchar> grayscale_image;
	vector<pair<string, double> > current_AusReg;

	// final predictions and decisions
	vector<double> result_emotions;

	// Get thresholds for confusion and surprise
	double get_confusion_thres(string root, string thres_path);
	double get_surprise_thres(string root, string thres_path);

    // Smoothing helper
    double smooth(const double &a, const double &b);

};
} /* InmindDemo */

#endif /* end of include guard: InmindDemo_InmindEmotionDetector_H_ */
