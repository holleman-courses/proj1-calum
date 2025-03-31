
#ifndef CAT_DETECTION_MODEL_SETTINGS_H_
#define CAT_DETECTION_MODEL_SETTINGS_H_

constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 1;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 2;
constexpr int kCatIndex = 1;
constexpr int kNoCatIndex = 0;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // CAT_DETECTION_MODEL_SETTINGS_H_
