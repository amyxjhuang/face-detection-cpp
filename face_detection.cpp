// #include "/usr/local/include/opencv2/objdetect.hpp"
// #include "/usr/local/include/opencv2/highgui.hpp"
// #include "/usr/local/include/opencv2/imgproc.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat& img, CascadeClassifier& faceCascade, CascadeClassifier& eyeglassesCascade, double scaleFactor);

string cascadeName, eyeglassesCascadeName;


int main(int argc, const char** argv) 
{
    VideoCapture capture;
    Mat frame, image;

    CascadeClassifier faceCascade;
    CascadeClassifier eyeglassesCascade;
    double scaleFactor = 1.2;
    int minNeighbors = 5;

    // Load classifiers from "opencv/data/haarcascades" directory 
    eyeglassesCascade.load( "opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml" ) ;
    faceCascade.load("opencv/data/haarcascades/haarcascade_frontalface_default.xml");
    capture.open(1); // Use macbook camera, 0 for continuity camera
    if (capture.isOpened()) {
        cout << "Started detecting face" << endl;
        while (capture.read(frame)) {
            capture >> frame; 
            if (frame.empty())
                break;
            
            Mat frame1 = frame.clone();
            detectAndDisplay(frame1, faceCascade, eyeglassesCascade, scaleFactor);
            if (waitKey(10) == 'q')
                break;
        }
    } else 
        cout <<"Could not open the camera";
    return 0;
}

void detectAndDisplay(Mat& img, CascadeClassifier& faceCascade, CascadeClassifier& eyeglassesCascade, double scaleFactor) 
{
    vector<Rect> faces, faces2;
    Mat gray, smallImg;

    // Convert to grayscale 
    cvtColor(img, gray, COLOR_BGR2GRAY);
    double fx = 1 / scaleFactor;

    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
    equalizeHist(smallImg, smallImg);
    
    int minNeighbors = 7;
    faceCascade.detectMultiScale(smallImg, faces, 1.1, minNeighbors, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (size_t i = 0; i < faces.size(); i++) {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar blue = Scalar(255, 0, 0);
        Scalar red = Scalar(0, 0, 255);
        rectangle( img, r.tl()*scaleFactor, r.br()*scaleFactor, red, 3, 8, 0);
        

        double aspect_ratio = (double)r.width/r.height;
        if (!eyeglassesCascade.empty() && aspect_ratio > 0.8 && aspect_ratio < 1.2) {
            eyeglassesCascade.detectMultiScale(smallImg(r), nestedObjects, 1.1, 6, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
            for (size_t j = 0; j < nestedObjects.size(); j++) {
                Rect nr = nestedObjects[j];
                center.x = cvRound((r.x + nr.x + nr.width*0.5)*scaleFactor);
                center.y = cvRound((r.y + nr.y + nr.height*0.5)*scaleFactor);
                ellipse(img, center, Size(nr.width*0.5*fx, nr.height*0.5*fx), 0, 0, 360, blue, 4, 8, 0);
            }
        }
    }
    imshow("result", img);
}