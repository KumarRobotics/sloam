// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <onnxruntime_cxx_api.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/dnn/dnn.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using Point = pcl::PointXYZI;
using Cloud = pcl::PointCloud<Point>;

namespace seg {
struct Timer{
    std::vector<std::chrono::system_clock::time_point> stimes;

    void tic() {
        stimes.push_back(std::chrono::high_resolution_clock::now());
    }

    /**
     * @brief stops the last timer started and outputs \a msg, if given.
     * @return elapsed time in seconds.
     **/
    double toc() {
        assert(stimes.begin() != stimes.end());

        std::chrono::system_clock::time_point endtime = std::chrono::high_resolution_clock::now();
        std::chrono::system_clock::time_point starttime = stimes.back();
        stimes.pop_back();

        std::chrono::duration<double> elapsed_seconds = endtime - starttime;

        auto t = elapsed_seconds.count();
        std::cout << "Elapsed: " << t * 1000 << "ms" << std::endl;
        return t;
    }
};

class Segmentation {
 public:
    explicit Segmentation(const std::string modelFilePath,
        const float fov_up, const float fov_down, const int img_w, const int img_h, const int img_d, bool do_destagger);

    Segmentation(const Segmentation &) = delete;
    Segmentation operator=(const Segmentation &) = delete;
    using Ptr = boost::shared_ptr<Segmentation>;
    using ConstPtr = boost::shared_ptr<const Segmentation>;

    void runERF(cv::Mat& rImg, cv::Mat& maskImg);
    void run(const Cloud::Ptr cloud, cv::Mat& maskImg);
    void maskCloud(const Cloud::Ptr cloud, cv::Mat mask, Cloud::Ptr& outCloud, unsigned char val, bool dense = false);
    void speedTest(const Cloud::Ptr cloud, size_t numTests);
 private:
    std::vector<std::vector<float>> _doProjection(const std::vector<float>& scan, const uint32_t& num_points);

    void _argmax(const float *in, cv::Mat& maskImg);

    void _preProcessRange(const cv::Mat& img, cv::Mat& pImg, float maxDist);
    // void _preProcessRange(cv::Mat& rImg, float maxDist);
    void _makeTensor(std::vector<std::vector<float>>& projected_data, std::vector<float>& tensor, std::vector<size_t>& invalid_idxs);
    void _startONNXSession(const std::string sessionName, const std::string modelFilePath, bool useCUDA, size_t numThreads);
    void _mask(const float* output, const std::vector<size_t>& invalid_idxs, cv::Mat& maskImg);
    void _destaggerCloud(const Cloud::Ptr cloud, Cloud::Ptr& outCloud);

    boost::shared_ptr<Ort::Session> _session = nullptr;
    boost::shared_ptr<Ort::MemoryInfo> _memoryInfo = nullptr;
    boost::shared_ptr<Ort::Env> _env = nullptr;

    Ort::AllocatorWithDefaultOptions _allocator;
    std::vector<const char*> _inputNames;
    std::vector<const char*> _outputNames;
    std::vector<int64_t> _inputDims;
    std::vector<int64_t> _outputDims;
    size_t _inputTensorSize;
    size_t _outputTensorSize;
    Timer _timer;
    bool _verbose = false;

    std::vector<float> proj_xs; // stope a copy in original order
    std::vector<float> proj_ys;

    float _fov_up;    // field of view up in radians
    float _fov_down;  // field of view down in radians
    float _fov; // get field of view total in radians
    int _img_w;
    int _img_h;
    int _img_d;
    bool _do_destagger;
};

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
void printVector(const std::vector<T>& v){
    std::cout << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        std::cout << v[i];
        if (i != v.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
// template <typename T>
// std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
// {
//     os << "[";
//     for (int i = 0; i < v.size(); ++i)
//     {
//         os << v[i];
//         if (i != v.size() - 1)
//         {
//             os << ", ";
//         }
//     }
//     os << "]";
//     return os;
// }

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
// std::ostream& operator<<(std::ostream& os,
//                          const ONNXTensorElementDataType& type)
// {
//     switch (type)
//     {
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
//             os << "undefined";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
//             os << "float";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
//             os << "uint8_t";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
//             os << "int8_t";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
//             os << "uint16_t";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
//             os << "int16_t";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
//             os << "int32_t";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
//             os << "int64_t";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
//             os << "std::string";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
//             os << "bool";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
//             os << "float16";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
//             os << "double";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
//             os << "uint32_t";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
//             os << "uint64_t";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
//             os << "float real + float imaginary";
//             break;
//         case ONNXTensorElementDataType::
//             ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
//             os << "double real + float imaginary";
//             break;
//         case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
//             os << "bfloat16";
//             break;
//         default:
//             break;
//     }

//     return os;
// }

// std::vector<std::string> readLabels(std::string& labelFilepath)
// {
//     std::vector<std::string> labels;
//     std::string line;
//     std::ifstream fp(labelFilepath);
//     while (std::getline(fp, line))
//     {
//         labels.push_back(line);
//     }
//     return labels;
// }

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v. >: decrease <: increase
    std::sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}
}// namespace segmentation