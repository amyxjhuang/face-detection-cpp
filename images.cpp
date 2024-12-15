


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // string path = "/Users/amy/Desktop/opencv-intro/images/test.jpg";
    Mat img = imread("images/selfie.jpg", IMREAD_COLOR);
    if (img.empty()) {
        cout << "Could not read the image: " << endl;
        return 1;
    }
    imshow("Image", img);
    int k = waitKey(0);
    if (k == 's') {
        imwrite("images/selfie.png", img);
    }
    return 0;
}