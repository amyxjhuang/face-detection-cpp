// #include "/usr/local/include/opencv2/objdetect.hpp"
// #include "/usr/local/include/opencv2/highgui.hpp"
// #include "/usr/local/include/opencv2/imgproc.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat& img, CascadeClassifier& faceCascade, CascadeClassifier& nestedCascade, double scaleFactor);

string cascadeName, nestedCascadeName;


int main(int argc, const char** argv) 
{
    VideoCapture capture;
    Mat frame, image;

    CascadeClassifier faceCascade;
    CascadeClassifier nestedCascade;
    double scaleFactor = 1.1;
    int minNeighbors = 2;

    // Load classifiers from "opencv/data/haarcascades" directory 
    nestedCascade.load( "opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml" ) ;

    faceCascade.load( "opencv/data/haarcascades/haarcascade_frontalcatface.xml" ) ; 
    capture.open(0);
    if (capture.isOpened()) {
        cout << "Started detecting face" << endl;
        while (capture.read(frame)) {
            capture >> frame; 
            if (frame.empty())
                break;
            
            Mat frame1 = frame.clone();
            detectAndDisplay(frame1, faceCascade, nestedCascade, scaleFactor);
            if (waitKey(10) == 'q')
                break;
        }
    } else 
        cout <<"Could not open the camera";
    return 0;
}

void detectAndDisplay(Mat& img, CascadeClassifier& faceCascade, CascadeClassifier& nestedCascade, double scaleFactor) 
{
    vector<Rect> faces, faces2;
    Mat gray, smallImg;

    // Convert to grayscale 
    cvtColor(img, gray, COLOR_BGR2GRAY);
    double fx = 1 / scaleFactor;

    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
    equalizeHist(smallImg, smallImg);
    
    faceCascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (size_t i = 0; i < faces.size(); i++) {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = Scalar(255, 0, 0);

        double aspect_ratio = (double)r.width/r.height;
        if (!nestedCascade.empty() && aspect_ratio > 0.8 && aspect_ratio < 1.2) {
            nestedCascade.detectMultiScale(smallImg(r), nestedObjects, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
            for (size_t j = 0; j < nestedObjects.size(); j++) {
                Rect nr = nestedObjects[j];
                center.x = cvRound((r.x + nr.x + nr.width*0.5)*fx);
                center.y = cvRound((r.y + nr.y + nr.height*0.5)*fx);
                ellipse(img, center, Size(nr.width*0.5*fx, nr.height*0.5*fx), 0, 0, 360, color, 4, 8, 0);
            }
        }
    }
    imshow("result", img);
}