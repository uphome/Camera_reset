#define CVUI_IMPLEMENTATION
#include "/home/hu/Libraries/cvui-2.7.0/cvui.h"
#include <random>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <stdlib.h>
#include "opencv2/viz.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <cassert>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
using namespace ::std;
using namespace ::cv;
using namespace ::Eigen;




const int mSigma=10;
const int mMaxIterations=200;
//TODO：完成旋转角度的可视化。  采用欧拉角
//TODO: 完善计算R，T的方法。
struct key_img
{
  Mat image; // 原图
  vector<KeyPoint> key_point; //特征点
  Mat image_key; //画好特征点的图片
  Mat dest; // 描述点

};
float CheckHomography(
    const cv::Mat &H21,                 //从参考帧到当前帧的单应矩阵
    const cv::Mat &H12,                 //从当前帧到参考帧的单应矩阵
    vector<bool> &vbMatchesInliers,     //匹配好的特征点对的Inliers标记
    float sigma,
    vector<DMatch>&mvMatches12,
    key_img old_img,
    key_img new_img)                               //估计误差
{
  // 说明：在已值n维观测数据误差服从N(0，sigma）的高斯分布时
  // 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
  // 其中：e(i) = [e_x,e_y,...]^T, Q维观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
  // 误差加权最小二次结果越小，说明观测数据精度越高
  // 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
  // 算法目标： 检查单应变换矩阵
  // 检查方式：通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权最小二乘投影误差

  // 算法流程
  // input: 单应性矩阵 H21, H12, 匹配点集 mvKeys1
  //    do:
  //        for p1(i), p2(i) in mvKeys:
  //           error_i1 = ||p2(i) - H21 * p1(i)||2
  //           error_i2 = ||p1(i) - H12 * p2(i)||2
  //
  //           w1 = 1 / sigma / sigma
  //           w2 = 1 / sigma / sigma
  //
  //           if error1 < th
  //              score +=   th - error_i1 * w1
  //           if error2 < th
  //              score +=   th - error_i2 * w2
  //
  //           if error_1i > th or error_2i > th
  //              p1(i), p2(i) are inner points
  //              vbMatchesInliers(i) = true
  //           else
  //              p1(i), p2(i) are outliers
  //              vbMatchesInliers(i) = false
  //           end
  //        end
  //   output: score, inliers

  // 特点匹配个数
  const int N = mvMatches12.size();

  // Step 1 获取从参考帧到当前帧的单应矩阵的各个元素
  const float h11 = H21.at<float>(0,0);
  const float h12 = H21.at<float>(0,1);
  const float h13 = H21.at<float>(0,2);
  const float h21 = H21.at<float>(1,0);
  const float h22 = H21.at<float>(1,1);
  const float h23 = H21.at<float>(1,2);
  const float h31 = H21.at<float>(2,0);
  const float h32 = H21.at<float>(2,1);
  const float h33 = H21.at<float>(2,2);

  // 获取从当前帧到参考帧的单应矩阵的各个元素
  const float h11inv = H12.at<float>(0,0);
  const float h12inv = H12.at<float>(0,1);
  const float h13inv = H12.at<float>(0,2);
  const float h21inv = H12.at<float>(1,0);
  const float h22inv = H12.at<float>(1,1);
  const float h23inv = H12.at<float>(1,2);
  const float h31inv = H12.at<float>(2,0);
  const float h32inv = H12.at<float>(2,1);
  const float h33inv = H12.at<float>(2,2);

  // 给特征点对的Inliers标记预分配空间
  vbMatchesInliers.resize(N);

  // 初始化score值
  float score = 0;

  // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
  // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
  const float th = 5.991;

  //信息矩阵，方差平方的倒数
  const float invSigmaSquare = 1.0/(sigma * sigma);

  // Step 2 通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权重投影误差
  // H21 表示从img1 到 img2的变换矩阵
  // H12 表示从img2 到 img1的变换矩阵
  for(int i = 0; i < N; i++)
  {
    // 一开始都默认为Inlier
    bool bIn = true;

    // Step 2.1 提取参考帧和当前帧之间的特征匹配点对
    const cv::KeyPoint &kp1 = old_img.key_point[mvMatches12[i].queryIdx];
    const cv::KeyPoint &kp2 = new_img.key_point[mvMatches12[i].trainIdx];
    const float u1 = kp1.pt.x;
    const float v1 = kp1.pt.y;
    const float u2 = kp2.pt.x;
    const float v2 = kp2.pt.y;

    // Step 2.2 计算 img2 到 img1 的重投影误差
    // x1 = H12*x2
    // 将图像2中的特征点通过单应变换投影到图像1中
    // |u1|   |h11inv h12inv h13inv||u2|   |u2in1|
    // |v1| = |h21inv h22inv h23inv||v2| = |v2in1| * w2in1inv
    // |1 |   |h31inv h32inv h33inv||1 |   |  1  |
    // 计算投影归一化坐标
    const float w2in1inv = 1.0/(h31inv * u2 + h32inv * v2 + h33inv);
    const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
    const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

    // 计算重投影误差 = ||p1(i) - H12 * p2(i)||2
    const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);
    const float chiSquare1 = squareDist1 * invSigmaSquare;

    // Step 2.3 用阈值标记离群点，内点的话累加得分
    if(chiSquare1>th)
      bIn = false;
    else
      // 误差越大，得分越低
      score += th - chiSquare1;

    // 计算从img1 到 img2 的投影变换误差
    // x1in2 = H21*x1
    // 将图像2中的特征点通过单应变换投影到图像1中
    // |u2|   |h11 h12 h13||u1|   |u1in2|
    // |v2| = |h21 h22 h23||v1| = |v1in2| * w1in2inv
    // |1 |   |h31 h32 h33||1 |   |  1  |
    // 计算投影归一化坐标
    const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
    const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
    const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

    // 计算重投影误差
    const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
    const float chiSquare2 = squareDist2*invSigmaSquare;

    // 用阈值标记离群点，内点的话累加得分
    if(chiSquare2>th)
      bIn = false;
    else
      score += th - chiSquare2;

    // Step 2.4 如果从img2 到 img1 和 从img1 到img2的重投影误差均满足要求，则说明是Inlier point
    if(bIn)
      vbMatchesInliers[i]=true;
    else
      vbMatchesInliers[i]=false;
  }
  return score;
}

 float CheckFundamental(
    const cv::Mat &F21,             //当前帧和参考帧之间的基础矩阵
    vector<bool> &vbMatchesInliers, //匹配的特征点对属于inliers的标记
    float sigma,
    vector<DMatch>&mvMatches12,
    key_img old_img,
    key_img new_img)                    //方差
{

  // 说明：在已值n维观测数据误差服从N(0，sigma）的高斯分布时
  // 其误差加权最小二乘结果为  sum_error = SUM(e(i)^T * Q^(-1) * e(i))
  // 其中：e(i) = [e_x,e_y,...]^T, Q维观测数据协方差矩阵，即sigma * sigma组成的协方差矩阵
  // 误差加权最小二次结果越小，说明观测数据精度越高
  // 那么，score = SUM((th - e(i)^T * Q^(-1) * e(i)))的分数就越高
  // 算法目标：检查基础矩阵
  // 检查方式：利用对极几何原理 p2^T * F * p1 = 0
  // 假设：三维空间中的点 P 在 img1 和 img2 两图像上的投影分别为 p1 和 p2（两个为同名点）
  //   则：p2 一定存在于极线 l2 上，即 p2*l2 = 0. 而l2 = F*p1 = (a, b, c)^T
  //      所以，这里的误差项 e 为 p2 到 极线 l2 的距离，如果在直线上，则 e = 0
  //      根据点到直线的距离公式：d = (ax + by + c) / sqrt(a * a + b * b)
  //      所以，e =  (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)

  // 算法流程
  // input: 基础矩阵 F 左右视图匹配点集 mvKeys1
  //    do:
  //        for p1(i), p2(i) in mvKeys:
  //           l2 = F * p1(i)
  //           l1 = p2(i) * F
  //           error_i1 = dist_point_to_line(x2,l2)
  //           error_i2 = dist_point_to_line(x1,l1)
  //
  //           w1 = 1 / sigma / sigma
  //           w2 = 1 / sigma / sigma
  //
  //           if error1 < th
  //              score +=   thScore - error_i1 * w1
  //           if error2 < th
  //              score +=   thScore - error_i2 * w2
  //
  //           if error_1i > th or error_2i > th
  //              p1(i), p2(i) are inner points
  //              vbMatchesInliers(i) = true
  //           else
  //              p1(i), p2(i) are outliers
  //              vbMatchesInliers(i) = false
  //           end
  //        end
  //   output: score, inliers

  // 获取匹配的特征点对的总对数
  const int N = mvMatches12.size();

  // Step 1 提取基础矩阵中的元素数据
  const float f11 = F21.at<float>(0,0);
  const float f12 = F21.at<float>(0,1);
  const float f13 = F21.at<float>(0,2);
  const float f21 = F21.at<float>(1,0);
  const float f22 = F21.at<float>(1,1);
  const float f23 = F21.at<float>(1,2);
  const float f31 = F21.at<float>(2,0);
  const float f32 = F21.at<float>(2,1);
  const float f33 = F21.at<float>(2,2);

  // 预分配空间
  vbMatchesInliers.resize(N);

  // 设置评分初始值（因为后面需要进行这个数值的累计）
  float score = 0;

  // 基于卡方检验计算出的阈值
  // 自由度为1的卡方分布，显著性水平为0.05，对应的临界阈值
  // ?是因为点到直线距离是一个自由度吗？
  const float th = 3.841;

  // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
  const float thScore = 5.991;

  // 信息矩阵，或 协方差矩阵的逆矩阵
  const float invSigmaSquare = 1.0/(sigma*sigma);


  // Step 2 计算img1 和 img2 在估计 F 时的score值
  for(int i=0; i<N; i++)
  {
    //默认为这对特征点是Inliers
    bool bIn = true;

    // Step 2.1 提取参考帧和当前帧之间的特征匹配点对
    const cv::KeyPoint &kp1 = old_img.key_point[mvMatches12[i].queryIdx];
    const cv::KeyPoint &kp2 = new_img.key_point[mvMatches12[i].trainIdx];

    // 提取出特征点的坐标
    const float u1 = kp1.pt.x;
    const float v1 = kp1.pt.y;
    const float u2 = kp2.pt.x;
    const float v2 = kp2.pt.y;

    // Reprojection error in second image
    // Step 2.2 计算 img1 上的点在 img2 上投影得到的极线 l2 = F21 * p1 = (a2,b2,c2)
    const float a2 = f11*u1+f12*v1+f13;
    const float b2 = f21*u1+f22*v1+f23;
    const float c2 = f31*u1+f32*v1+f33;

    // Step 2.3 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
    const float num2 = a2*u2+b2*v2+c2;
    const float squareDist1 = num2*num2/(a2*a2+b2*b2);
    // 带权重误差
    const float chiSquare1 = squareDist1*invSigmaSquare;

    // Step 2.4 误差大于阈值就说明这个点是Outlier
    // ? 为什么判断阈值用的 th（1自由度），计算得分用的thScore（2自由度）
    // ? 可能是为了和CheckHomography 得分统一？
    if(chiSquare1>th)
      bIn = false;
    else
      // 误差越大，得分越低
      score += thScore - chiSquare1;

    // 计算img2上的点在 img1 上投影得到的极线 l1= p2 * F21 = (a1,b1,c1)
    const float a1 = f11*u2+f21*v2+f31;
    const float b1 = f12*u2+f22*v2+f32;
    const float c1 = f13*u2+f23*v2+f33;

    // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
    const float num1 = a1*u1+b1*v1+c1;
    const float squareDist2 = num1*num1/(a1*a1+b1*b1);

    // 带权重误差
    const float chiSquare2 = squareDist2*invSigmaSquare;

    // 误差大于阈值就说明这个点是Outlier
    if(chiSquare2>th)
      bIn = false;
    else
      score += thScore - chiSquare2;

    // Step 2.5 保存结果
    if(bIn)
      vbMatchesInliers[i]=true;
    else
      vbMatchesInliers[i]=false;
  }
  //  返回评分
  return score;
}
Mat ComputeH21(
    const vector<cv::Point2f> &vP1, //归一化后的点, in reference frame
    const vector<cv::Point2f> &vP2) //归一化后的点, in current frame
{
  // 基本原理：见附件推导过程：
  // |x'|     | h1 h2 h3 ||x|
  // |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
  // |1 |     | h7 h8 h9 ||1|
  // 使用DLT(direct linear tranform)求解该模型
  // x' = a H x
  // ---> (x') 叉乘 (H x)  = 0  (因为方向相同) (取前两行就可以推导出下面的了)
  // ---> Ah = 0
  // A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
  //     |-x -y -1  0  0  0 xx' yx' x'|
  // 通过SVD求解Ah = 0，A^T*A最小特征值对应的特征向量即为解
  // 其实也就是右奇异值矩阵的最后一列

  //获取参与计算的特征点的数目
  const int N = vP1.size();

  // 构造用于计算的矩阵 A
  cv::Mat A(2*N,				//行，注意每一个点的数据对应两行
            9,				//列
            CV_32F);      	//float数据类型

  // 构造矩阵A，将每个特征点添加到矩阵A中的元素
  for(int i=0; i<N; i++)
  {
    //获取特征点对的像素坐标
    const float u1 = vP1[i].x;
    const float v1 = vP1[i].y;
    const float u2 = vP2[i].x;
    const float v2 = vP2[i].y;

    //生成这个点的第一行
    A.at<float>(2*i,0) = 0.0;
    A.at<float>(2*i,1) = 0.0;
    A.at<float>(2*i,2) = 0.0;
    A.at<float>(2*i,3) = -u1;
    A.at<float>(2*i,4) = -v1;
    A.at<float>(2*i,5) = -1;
    A.at<float>(2*i,6) = v2*u1;
    A.at<float>(2*i,7) = v2*v1;
    A.at<float>(2*i,8) = v2;

    //生成这个点的第二行
    A.at<float>(2*i+1,0) = u1;
    A.at<float>(2*i+1,1) = v1;
    A.at<float>(2*i+1,2) = 1;
    A.at<float>(2*i+1,3) = 0.0;
    A.at<float>(2*i+1,4) = 0.0;
    A.at<float>(2*i+1,5) = 0.0;
    A.at<float>(2*i+1,6) = -u2*u1;
    A.at<float>(2*i+1,7) = -u2*v1;
    A.at<float>(2*i+1,8) = -u2;

  }

  // 定义输出变量，u是左边的正交矩阵U， w为奇异矩阵，vt中的t表示是右正交矩阵V的转置
  cv::Mat u,w,vt;

  //使用opencv提供的进行奇异值分解的函数
  cv::SVDecomp(A,							//输入，待进行奇异值分解的矩阵
               w,							//输出，奇异值矩阵
               u,							//输出，矩阵U
               vt,						//输出，矩阵V^T
               cv::SVD::MODIFY_A | 		//输入，MODIFY_A是指允许计算函数可以修改待分解的矩阵，官方文档上说这样可以加快计算速度、节省内存
                   cv::SVD::FULL_UV);		//FULL_UV=把U和VT补充成单位正交方阵

  // 返回最小奇异值所对应的右奇异向量
  // 注意前面说的是右奇异值矩阵的最后一列，但是在这里因为是vt，转置后了，所以是行；由于A有9列数据，故最后一列的下标为8
  return vt.row(8).reshape(0, 			//转换后的通道数，这里设置为0表示是与前面相同
                           3); 			//转换后的行数,对应V的最后一列
}
Mat ComputeF21(
    const vector<cv::Point2f> &vP1, //归一化后的点, in reference frame
    const vector<cv::Point2f> &vP2) //归一化后的点, in current frame
{
  // 原理详见附件推导
  // x'Fx = 0 整理可得：Af = 0
  // A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
  // 通过SVD求解Af = 0，A'A最小特征值对应的特征向量即为解

  //获取参与计算的特征点对数
  const int N = vP1.size();

  //初始化A矩阵
  cv::Mat A(N,9,CV_32F); // N*9维

  // 构造矩阵A，将每个特征点添加到矩阵A中的元素
  for(int i=0; i<N; i++)
  {
    const float u1 = vP1[i].x;
    const float v1 = vP1[i].y;
    const float u2 = vP2[i].x;
    const float v2 = vP2[i].y;

    A.at<float>(i,0) = u2*u1;
    A.at<float>(i,1) = u2*v1;
    A.at<float>(i,2) = u2;
    A.at<float>(i,3) = v2*u1;
    A.at<float>(i,4) = v2*v1;
    A.at<float>(i,5) = v2;
    A.at<float>(i,6) = u1;
    A.at<float>(i,7) = v1;
    A.at<float>(i,8) = 1;
  }

  //存储奇异值分解结果的变量
  cv::Mat u,w,vt;


  // 定义输出变量，u是左边的正交矩阵U， w为奇异矩阵，vt中的t表示是右正交矩阵V的转置
  cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  // 转换成基础矩阵的形式
  cv::Mat Fpre = vt.row(8).reshape(0, 3); // v的最后一列

  //基础矩阵的秩为2,而我们不敢保证计算得到的这个结果的秩为2,所以需要通过第二次奇异值分解,来强制使其秩为2
  // 对初步得来的基础矩阵进行第2次奇异值分解
  cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

  // 秩2约束，强制将第3个奇异值设置为0
  w.at<float>(2)=0;

  // 重新组合好满足秩约束的基础矩阵，作为最终计算结果返回
  return  u*cv::Mat::diag(w)*vt;
}
void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)                           //将特征点归一化的矩阵
{
  // 归一化的是这些点在x方向和在y方向上的一阶绝对矩（随机变量的期望）。

  // Step 1 计算特征点X,Y坐标的均值 meanX, meanY
  float meanX = 0;
  float meanY = 0;

  //获取特征点的数量
  const int N = vKeys.size();

  //设置用来存储归一后特征点的向量大小，和归一化前保持一致
  vNormalizedPoints.resize(N);

  //开始遍历所有的特征点
  for (int i = 0; i < N; i++) {
    //分别累加特征点的X、Y坐标
    meanX += vKeys[i].pt.x;
    meanY += vKeys[i].pt.y;
  }

  //计算X、Y坐标的均值
  meanX = meanX / N;
  meanY = meanY / N;

  // Step 2 计算特征点X,Y坐标离均值的平均偏离程度 meanDevX, meanDevY，注意不是标准差
  float meanDevX = 0;
  float meanDevY = 0;

  // 将原始特征点减去均值坐标，使x坐标和y坐标均值分别为0
  for (int i = 0; i < N; i++) {
    vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
    vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

    //累计这些特征点偏离横纵坐标均值的程度
    meanDevX += fabs(vNormalizedPoints[i].x);
    meanDevY += fabs(vNormalizedPoints[i].y);
  }

  // 求出平均到每个点上，其坐标偏离横纵坐标均值的程度；将其倒数作为一个尺度缩放因子
  meanDevX = meanDevX / N;
  meanDevY = meanDevY / N;
  float sX = 1.0 / meanDevX;
  float sY = 1.0 / meanDevY;

  // Step 3 将x坐标和y坐标分别进行尺度归一化，使得x坐标和y坐标的一阶绝对矩分别为1
  // 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值（期望）
  for (int i = 0; i < N; i++) {
    //对，就是简单地对特征点的坐标进行进一步的缩放
    vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
    vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
  }

  // Step 4 计算归一化矩阵：其实就是前面做的操作用矩阵变换来表示而已
  // |sX  0  -meanx*sX|
  // |0   sY -meany*sY|
  // |0   0      1    |
  T = cv::Mat::eye(3, 3, CV_32F);
  T.at<float>(0, 0) = sX;
  T.at<float>(1, 1) = sY;
  T.at<float>(0, 2) = -meanX * sX;
  T.at<float>(1, 2) = -meanY * sY;
}
void FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21,vector<DMatch>&mvMatches12,key_img new_img,key_img old_img,vector<vector<size_t>> &mvSets)
{
// 计算基础矩阵,其过程和上面的计算单应矩阵的过程十分相似.

// Number of putative matches
// 匹配的特征点对总数
// const int N = vbMatchesInliers.size();  // !源代码出错！请使用下面代替
const int N = mvMatches12.size();
// Normalize coordinates
// Step 1 将当前帧和参考帧中的特征点坐标进行归一化，主要是平移和尺度变换
// 具体来说,就是将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1、T2
// 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值
// 归一化矩阵就是把上述归一化的操作用矩阵来表示。这样特征点坐标乘归一化矩阵可以得到归一化后的坐标

vector<cv::Point2f> vPn1, vPn2;
cv::Mat T1, T2;
Normalize(old_img.key_point,vPn1, T1);
Normalize(new_img.key_point,vPn2, T2);
// ! 注意这里取的是归一化矩阵T2的转置,因为基础矩阵的定义和单应矩阵不同，两者去归一化的计算也不相同
cv::Mat T2t = T2.t();

// Best Results variables
//最优结果
score = 0.0;
vbMatchesInliers = vector<bool>(N,false);

// Iteration variables
// 某次迭代中，参考帧的特征点坐标
vector<cv::Point2f> vPn1i(8);
// 某次迭代中，当前帧的特征点坐标
vector<cv::Point2f> vPn2i(8);
// 某次迭代中，计算的基础矩阵
cv::Mat F21i;

// 每次RANSAC记录的Inliers与得分
vector<bool> vbCurrentInliers(N,false);
float currentScore;

// Perform all RANSAC iterations and save the solution with highest score
// 下面进行每次的RANSAC迭代

for(int it=0; it<mMaxIterations; it++)
{
// Select a minimum set
// Step 2 选择8个归一化之后的点对进行迭代
for(int j=0; j<8; j++)
{
int idx = mvSets[it][j];

// vPn1i和vPn2i为匹配的特征点对的归一化后的坐标
// 首先根据这个特征点对的索引信息分别找到两个特征点在各自图像特征点向量中的索引，然后读取其归一化之后的特征点坐标
vPn1i[j] = vPn1[mvMatches12[idx].queryIdx];        //first存储在参考帧1中的特征点索引
vPn2i[j] = vPn2[mvMatches12[idx].trainIdx];       //second存储在参考帧1中的特征点索引
}

// Step 3 八点法计算基础矩阵
cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

// 基础矩阵约束：p2^t*F21*p1 = 0，其中p1,p2 为齐次化特征点坐标
// 特征点归一化：vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2
// 根据基础矩阵约束得到:(T2 * mvKeys2)^t* Hn * T1 * mvKeys1 = 0
// 进一步得到:mvKeys2^t * T2^t * Hn * T1 * mvKeys1 = 0
F21i = T2t*Fn*T1;

// Step 4 利用重投影误差为当次RANSAC的结果评分

currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma,mvMatches12,old_img,new_img);

// Step 5 更新具有最优评分的基础矩阵计算结果,并且保存所对应的特征点对的内点标记
if(currentScore>score)
{
//如果当前的结果得分更高，那么就更新最优计算结果
F21 = F21i.clone();
vbMatchesInliers = vbCurrentInliers;
score = currentScore;
}
}
}
void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21,vector<DMatch>&mvMatches12,key_img new_img,key_img old_img,vector<vector<size_t>> &mvSets)
{
// Number of putative matches
//匹配的特征点对总数
const int N = mvMatches12.size();

// Normalize coordinates
// Step 1 将当前帧和参考帧中的特征点坐标进行归一化，主要是平移和尺度变换
// 具体来说,就是将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1、T2
// 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值
// 归一化矩阵就是把上述归一化的操作用矩阵来表示。这样特征点坐标乘归一化矩阵可以得到归一化后的坐标


//归一化后的参考帧1和当前帧2中的特征点坐标
vector<cv::Point2f> vPn1, vPn2;
// 记录各自的归一化矩阵
cv::Mat T1, T2;
Normalize(new_img.key_point,vPn1, T1);
Normalize(old_img.key_point,vPn2, T2);

//这里求的逆在后面的代码中要用到，辅助进行原始尺度的恢复
cv::Mat T2inv = T2.inv();

// Best Results variables
// 记录最佳评分
score = 0.0;
// 取得历史最佳评分时,特征点对的inliers标记
vbMatchesInliers = vector<bool>(N,false);

// Iteration variables
//某次迭代中，参考帧的特征点坐标
vector<cv::Point2f> vPn1i(8);
//某次迭代中，当前帧的特征点坐标
vector<cv::Point2f> vPn2i(8);
//以及计算出来的单应矩阵、及其逆矩阵
cv::Mat H21i, H12i;

// 每次RANSAC记录Inliers与得分
vector<bool> vbCurrentInliers(N,false);
float currentScore;

// Perform all RANSAC iterations and save the solution with highest score
//下面进行每次的RANSAC迭代

for(int it=0; it<mMaxIterations; it++)
{
// Select a minimum set
// Step 2 选择8个归一化之后的点对进行迭代
for(size_t j=0; j<8; j++)
{
//从mvSets中获取当前次迭代的某个特征点对的索引信息
int idx = mvSets[it][j];

// vPn1i和vPn2i为匹配的特征点对的归一化后的坐标
// 首先根据这个特征点对的索引信息分别找到两个特征点在各自图像特征点向量中的索引，然后读取其归一化之后的特征点坐标
vPn1i[j] = vPn1[mvMatches12[idx].queryIdx];    //first存储在参考帧1中的特征点索引
vPn2i[j] = vPn2[mvMatches12[idx].trainIdx];   //second存储在参考帧1中的特征点索引
}//读取8对特征点的归一化之后的坐标

// Step 3 八点法计算单应矩阵
// 利用生成的8个归一化特征点对, 调用函数 Initializer::ComputeH21() 使用八点法计算单应矩阵
// 关于为什么计算之前要对特征点进行归一化，后面又恢复这个矩阵的尺度？
// 可以在《计算机视觉中的多视图几何》这本书中P193页中找到答案
// 书中这里说,8点算法成功的关键是在构造解的方称之前应对输入的数据认真进行适当的归一化

cv::Mat Hn = ComputeH21(vPn1i,vPn2i);

// 单应矩阵原理：X2=H21*X1，其中X1,X2 为归一化后的特征点
// 特征点归一化：vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2  得到:T2 * mvKeys2 =  Hn * T1 * mvKeys1
// 进一步得到:mvKeys2  = T2.inv * Hn * T1 * mvKeys1
H21i = T2inv*Hn*T1;
//然后计算逆
H12i = H21i.inv();

// Step 4 利用重投影误差为当次RANSAC的结果评分
currentScore = CheckHomography(H21i, H12i, 			//输入，单应矩阵的计算结果
                               vbCurrentInliers, 	//输出，特征点对的Inliers标记
                               mSigma,
                               mvMatches12,
                               old_img,
                               new_img);				//TODO  测量误差，在Initializer类对象构造的时候，由外部给定的


// Step 5 更新具有最优评分的单应矩阵计算结果,并且保存所对应的特征点对的内点标记
if(currentScore>score)
{
  cout<<currentScore<<endl;
//如果当前的结果得分更高，那么就更新最优计算结果
H21 = H21i.clone();
//保存匹配好的特征点对的Inliers标记
vbMatchesInliers = vbCurrentInliers;
//更新历史最优评分
score = currentScore;
}
}
}


void feature_points(Mat met,struct key_img &fe_point)
{ //met.resize(480,640);
  fe_point.image= met;
  Mat image_key;
  Mat dest;
  Ptr<ORB> orb = ORB::create();
  vector<KeyPoint> key_point;
  orb->detectAndCompute(fe_point.image, Mat(), fe_point.key_point, fe_point.dest);   //寻找特征点

  drawKeypoints(fe_point.image, fe_point.key_point, fe_point.image_key, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);

}

vector<DMatch>matching(struct key_img new_image, struct key_img old_image)
{
  vector<vector<Point2f>> maPoint;
  vector<Point2f>maPoint_old,maPoint_new;
  BFMatcher matcher;
  //Ptr<DescriptorMatcher> matches = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
  vector<DMatch>matches;
  vector<Mat>trian(1,old_image.dest);
  matcher.add(trian);
  matcher.train();

  const float minRatio = 0.7;


  vector<vector<DMatch>>matchpoints;
  matcher.knnMatch(new_image.dest,matchpoints,2);
  vector<DMatch>goodfeatur;

  for (int i =0;i<matchpoints.size();i++)
  {
    if (matchpoints[i][0].distance/matchpoints[i][1].distance<=minRatio)
    {
      goodfeatur.push_back(matchpoints[i][0]);
    }


  }
  cout<<"匹配特征点的筛选比值为"<<minRatio<<endl;
  cout<<"筛选好之后的匹配特征点数目为"<<goodfeatur.size()<<endl;
  assert(("筛选好之后的匹配特征点数目小于8对",goodfeatur.size()>=8));

  Mat result_img;
  drawMatches(new_image.image,new_image.key_point,old_image.image,old_image.key_point, goodfeatur,result_img);
  drawKeypoints(old_image.image,old_image.key_point,old_image.image);
  drawKeypoints(new_image.image,new_image.key_point,new_image.image);

  namedWindow("匹配结果",WINDOW_NORMAL);
  resizeWindow("匹配结果",500,500);
  cv::imshow("匹配结果",result_img);

  waitKey(0);
 // system("pause");

  return goodfeatur;
}

vector<Mat> calmatrix(vector<vector<cv::Point2f>> mpoint)
{ Mat R,t;
  Point2f pricipal(323.1992,240.1797);
  float focal=477.7987;
  Mat ess_matrix,Fundamental,homo_matrix;
  Fundamental= findFundamentalMat(mpoint[0],mpoint[1],CV_FM_8POINT);
  float SH,SF;


  ess_matrix =findEssentialMat(mpoint[0],mpoint[1],focal,pricipal);
  cout<<"本质矩阵为"<<endl;
  cout<<ess_matrix<<endl;

  homo_matrix= findHomography(mpoint[0],mpoint[1],RANSAC,3);


  recoverPose(ess_matrix,mpoint[0],mpoint[1],R,t,focal,pricipal);
  vector<Mat>Pose;
  Pose.push_back(R);
  Pose.push_back(t);

  return Pose;

}



void drawing_rot(vector<Mat>Pose,struct key_img oimg,struct key_img nimg)
{
  Vector3d euler;
  //euler=Pose[1].eulerAngles(2,1,0);  // ZYX顺序，即先绕x轴roll,再绕y轴pitch,最后绕z轴yaw,0表示X轴,1表示Y轴,2表示Z轴
  viz::Viz3d window("camere pose");
  Matx33f K(477.7987, 0, 323.1992, 0, 477.4408, 240.1797, 0, 0, 1); // 内参矩阵
  viz::Camera mainCamera(K,Size(640,480)); //初始化
  cout<<Mat::eye(3,3,CV_32F)<<endl;

  viz::WCameraPosition camparmo(mainCamera.getFov(),oimg.image,1.0,viz::Color::white()); // 参数设置
  cv::Affine3f camPosition(Matx33f (1,0,0,0,1,0,0,0,1),Vec3f(0,0,0));


  viz::WCameraPosition camparmn(mainCamera.getFov(),nimg.image,1.0,viz::Color::green());
  cv::Affine3d camPosition_2(Pose[0],Pose[1]);
  window.showWidget("oldimage",camparmo,camPosition);
  window.showWidget("newimage",camparmn,camPosition_2);
  window.spin();


}


int main() {
  key_img oimage, nimage;
  vector<vector<Point2f>> mPoint;// [0] 为new [1] 为old；
  vector<Mat> Pose;
  Mat frame;
  VideoCapture capture;

    Mat K = (Mat_<double>(3, 3) << 477.7987, 0, 323.1992, 0, 477.4408, 240.1797, 0, 0, 1);
    Mat image_address_new = imread( "/home/hu/下载/IMG20230328182935.jpg");
    Mat image_address_old = imread( "/home/hu/下载/IMG20230328182918.jpg");
    feature_points(image_address_new, nimage);
    feature_points(image_address_old, oimage);

  vector<DMatch>goodfeatur;
  goodfeatur = matching(nimage, oimage);



  int N=goodfeatur.size();
  vector<size_t> vAllIndices;
  vAllIndices.reserve(N); //储存特征点索引，并预分配空间
  vector<size_t> vAvailableIndices;//在RANSAC的某次迭代中，还可以被抽取来作为数据样本的特征点对的索引

  //初始化所有特征点对的索引，索引值0到N-1
  for(int i=0; i<N; i++)
  {
    vAllIndices.push_back(i);
  }
  //cout << '-----'<< endl;
  vector<vector<size_t>>mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));
  default_random_engine   e;//随机数引擎对象
  e.seed(time(NULL)); //初始化种子

  for(int it=0; it<mMaxIterations; it++)
  {

    vAvailableIndices = vAllIndices;

    // Select a minimum set
    //选择最小的数据样本集，使用八点法求，所以这里就循环了八次
    for(size_t j=0; j<8; j++)
    {
      // 随机产生一对点的id,范围从0到N-1
      uniform_int_distribution<unsigned>  u(0,vAvailableIndices.size()-1);
      int randi = u(e);
      // idx表示哪一个索引对应的特征点对被选中
      int idx = vAvailableIndices[randi];

      //将本次迭代这个选中的第j个特征点对的索引添加到mvSets中
      mvSets[it][j] = idx;


      vAvailableIndices[randi] = vAvailableIndices.back();
      vAvailableIndices.pop_back();  //删除尾部数据
    }//依次提取出8个特征点对
  }//迭代mMaxIterations次，选取各自迭代时需要用到的最小数据集

  vector<bool> vbMatchesInliersH, vbMatchesInliersF;
  float SH, SF; //score for H and F
  cv::Mat H, F;

  FindFundamental(vbMatchesInliersF,SF,F,goodfeatur,oimage,nimage,mvSets);
  FindHomography(vbMatchesInliersH,SH,H,goodfeatur,oimage,nimage,mvSets);
  cout<<"SF为："<<SF<<endl;
  cout<<"F为："<<F<<endl;
  cout<<"SH为："<<SH<<endl;
  float RH = SH/(SH+SF);
  cout<<RH<<endl;



}
