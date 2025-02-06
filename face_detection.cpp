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

vector<Point> findLargestContour(vector<vector<Point>> contours) {
    vector<Point> largestContour;
    double maxArea = 0.0;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largestContour = contours[i];
        }
    }
    return largestContour;
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

                // draws circles around the eyes
                // ellipse(img, center, Size(nr.width*0.5*fx, nr.height*0.5*fx), 0, 0, 360, blue, 4, 8, 0); 
            }
        }
    }
    Mat blurred, thresholded;
    // Rect face = faces[0];  // Exclude above the first face
    // int handRegionY = face.y + face.height + 10;  // Area below face

    // Rect handROI(0, handRegionY, img.cols, img.rows - handRegionY);
    // Mat handRegion = img(handROI);
    // rectangle(img, handROI, 1);

    // // Process only the hand region
    // cvtColor(handRegion, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, blurred, Size(5, 5), 0);
    // adaptiveThreshold(blurred, thresholded, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
    threshold(blurred, thresholded, 60, 255, THRESH_BINARY_INV);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    findContours(thresholded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        vector<Point> largestContour = findLargestContour(contours);

        if (!largestContour.empty() && largestContour.size() > 3) {
            drawContours(img, vector<vector<Point>>{largestContour}, -1, Scalar(0, 255, 0), 2);

            // Find Convex Hull
            vector<int> hullIndices;
            convexHull(largestContour, hullIndices, false, false);

            std::sort(hullIndices.begin(), hullIndices.end()); 
            // Find Convexity Defects
            vector<Vec4i> convexityDefectsVec;

            vector<Point> approxContour;
            approxPolyDP(largestContour, approxContour, 5, true);  
            if (approxContour.size() > 3 && hullIndices.size() > 3) {
                convexityDefects(largestContour, hullIndices, convexityDefectsVec);

                for (size_t i = 0; i < convexityDefectsVec.size(); i++) {
                    Vec4i defect = convexityDefectsVec[i];
                    Point start = largestContour[defect[0]];
                    Point end = largestContour[defect[1]];
                    Point far = largestContour[defect[2]];

                    // Draw convex hull points
                    circle(img, start, 5, Scalar(255, 0, 0), -1);
                    circle(img, end, 5, Scalar(255, 0, 0), -1);
                    circle(img, far, 5, Scalar(0, 0, 255), -1);

                    // Draw convexity defect line
                    line(img, start, end, Scalar(0, 255, 255), 2);
                    line(img, start, far, Scalar(255, 255, 0), 2);
                    line(img, end, far, Scalar(255, 255, 0), 2);
                }
            }
        }
    }

    // Display results
    imshow("Hand Detection", img);
    // imshow("Thresholded", thresholded);

    // imshow("result", img);
}