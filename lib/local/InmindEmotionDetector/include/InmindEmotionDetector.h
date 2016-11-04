#ifndef InmindDemo_InmindEmotionDetector_H_
#define InmindDemo_InmindEmotionDetector_H_

#include "EmotionDetector.h"

using namespace std;
using namespace boost::filesystem;
using namespace EmotionRecognition;
using namespace cv;

namespace InmindDemo {
	class InmindEmotionDetector {
	public:
		InmindEmotionDetector(string s);
		vector<double> DetectEmotion(Mat frame, double time_stamp);
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

        // Store old prediction
		double prev_confusion = -1.0;
		double prev_surprise = -1.0;
		double alpha = 0.15;

		// Scores
		double score_confusion;
		double score_surprise;
		double decision_confusion;
		double decision_surprise;

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

	};
}

#endif /* end of include guard: InmindDemo_InmindEmotionDetector_H_ */
