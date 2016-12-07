/**
* @Author: Chirag Raman <chirag>
* @Date:   2016-11-29T14:25:14-05:00
* @Email:  chirag.raman@gmail.com
* @Last modified by:   chirag
* @Last modified time: 2016-12-07T13:46:57-05:00
* @License: Copyright (C) 2016 Multicomp Lab. All rights reserved.
*/

#ifndef EXE_INMIND_DEMO_VISUAL_FEATURE_EXTRACTOR_H_
#define EXE_INMIND_DEMO_VISUAL_FEATURE_EXTRACTOR_H_

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace inmind {

class VisualFeatureExtractor {
 public:
     VisualFeatureExtractor();
     virtual ~VisualFeatureExtractor ();
     void get_features(cv::Mat &image);
 private:
};

} /* inmind */


#endif /* end of include guard: EXE_INMIND_DEMO_VISUAL_FEATURE_EXTRACTOR_H_ */
