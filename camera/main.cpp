#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
using namespace ::std;
using namespace ::cv;

struct key_img
{
  Mat image; // 原图
  vector<KeyPoint> key_point; //特征点
  Mat image_key; //画好特征点的图片
  Mat dest; // 描述点

};

void feature_points(string address,struct key_img &fe_point)
{
  fe_point.image= imread(address);
  Mat image_key;
  Mat dest;
  Ptr<ORB> orb = ORB::create();
  vector<KeyPoint> key_point;
  orb->detectAndCompute(fe_point.image, Mat(), fe_point.key_point, fe_point.dest);   //寻找特征点
 /* for(int i =0;i<fe_point.key_point.size();i++) {

    KeyPoint ima=fe_point.key_point[i] ;
    cout<<ima.pt<<endl;
    cout<<ima.octave<<endl;//图像金字塔的层数
    cout<<ima.angle<<endl;//特征点的方向
    cout<<ima.class_id<<endl;
    cout<<"--"<<endl;

  }*/
  drawKeypoints(fe_point.image, fe_point.key_point, fe_point.image_key, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  //cv::namedWindow("camera", CV_WINDOW_NORMAL);
  //imshow("camera",fe_point.image_key);
  //waitKey();

}

void matching(struct key_img new_image,struct key_img old_image)
{
  BFMatcher matcher;
  //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
  vector<DMatch>matches;
  vector<Mat>trian(1,old_image.dest);
  matcher.add(trian);
  matcher.train();

  vector<vector<DMatch>>matchpoints;
  matcher.knnMatch(new_image.dest,matchpoints,2);

  vector<DMatch>goodfeatur;
  for (int i =0;i<matchpoints.size();i++)
  {
    if (matchpoints[i][0].distance<0.7*matchpoints[i][1].distance)
    {
      goodfeatur.push_back(matchpoints[i][0]);
      cout<<goodfeatur[i].trainIdx<<endl;
      cout<<goodfeatur[i].queryIdx<<endl;
      cout<<goodfeatur[i].imgIdx<<endl;
      cout<<i<<"--------"<<endl;
    }
  }
  cout<<goodfeatur.size()<<endl;
  Mat result_img;
  drawMatches(new_image.image,new_image.key_point,old_image.image,old_image.key_point, goodfeatur,result_img);
  drawKeypoints(old_image.image,old_image.key_point,old_image.image);
  drawKeypoints(new_image.image,new_image.key_point,new_image.image);


  namedWindow("匹配结果",WINDOW_NORMAL);
  resizeWindow("匹配结果",500,500);
  imshow("匹配结果",result_img);

  waitKey(0);
  system("pause");


}



int main()
{

  string image_address_new="/home/hu/CLionProjects/camera/image_test/old.jpg";
  string image_address_old="/home/hu/CLionProjects/camera/image_test/new.jpg";
  key_img oimage,nimage;
  feature_points(image_address_new,nimage);
  feature_points(image_address_old,oimage);
  matching(nimage,oimage);



}
