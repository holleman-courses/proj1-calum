
// Provides an interface to take an action based on the output from the person
// detection model.

#ifndef CAT_DETECTION_DETECTION_RESPONDER_H_
#define CAT_DETECTION_DETECTION_RESPONDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        int8_t cat_score, int8_t no_cat_score);

#endif  // CAT_DETECTION_DETECTION_RESPONDER_H_
