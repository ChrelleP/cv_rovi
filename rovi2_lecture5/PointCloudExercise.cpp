#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>

using namespace std;

int main()
{
	pcl::PCLPointCloud2 cloud_local;
	pcl::PCLPointCloud2 cloud_scene;

	pcl::io::loadPCDFile("../point_clouds/scene.pcd", cloud_scene);
	pcl::io::loadPCDFile("../point_clouds/object-local.pcd", cloud_local);

	pcl::PointCloud<pcl::PointXYZ> pc_local;
	pcl::PointCloud<pcl::PointXYZ> pc_scene;
	pcl::fromPCLPointCloud2 (cloud_local, pc_local);
	pcl::fromPCLPointCloud2 (cloud_scene, pc_scene);

	pcl::PointCloud<pcl::PointXYZ>::Ptr ptrLocal(&pc_local);
	pcl::PointCloud<pcl::PointXYZ>::Ptr ptrScene(&pc_scene);

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  	kdtree.setInputCloud (ptrScene);

	vector<float> distance;
	vector<int> index;

	vector<pcl::Correspondence> corr_vec;

	//STEP 1
	for(int i = 0; i < ptrLocal->size(); i++){
		kdtree.nearestKSearch(ptrLocal->at(i), 1, index, distance);
		if(distance[0] > 0 && distance[0] < 0.01){
			pcl::Correspondence temp(i, index[0], distance[0]);
			corr_vec.push_back(temp);
			cout << "Distance: " << distance[0] << endl;
			cout << "Index matched: " << i << ", " << index[0] << endl;
		}
	}

	// pcl::PointCloud<pcl::PointXYZ> Final;
	// Eigen::Matrix4f transformation;
	// pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
	// svd.estimateRigidTransformation(ptrLocal, ptrScene, &corr_vec, &transformation);

	// cout << correspondences.size() << endl;



	// /// ITERATIVE CLOSEST POINT PCL IMPL. /////
	// pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	// icp.setMaxCorrespondenceDistance(0.01);
	// icp.setInputSource(ptrLocal);
	// icp.setInputTarget(ptrScene);
	//
	// pcl::PointCloud<pcl::PointXYZ> Final;
	// icp.align(Final);
	// std::cout << "has converged:" << icp.hasConverged() << " score: " <<
	// icp.getFitnessScore() << std::endl;
	// std::cout << icp.getFinalTransformation() << std::endl;
	// pcl::PointCloud<pcl::PointXYZ>::Ptr ptrFinal(&Final);



	pcl::visualization::PCLVisualizer viewer("PCL viewer");

	viewer.addPointCloud(ptrLocal, "cloud_local", 0);
	// viewer.addPointCloud(ptrFinal, "cloud_final", 0);
	viewer.addPointCloud(ptrScene, "cloud_scene", 0);

	while (!viewer.wasStopped())
    {
		viewer.spinOnce(100);
    }


	return 0;
}
