#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Path to the image
    std::string imagePath = "images/selfie.jpg";

    // Read the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    // Check if the image was successfully loaded
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Display the image
    cv::imshow("Display Image", image);

    // Wait for a key press indefinitely
    cv::waitKey(0);

    return 0;
}