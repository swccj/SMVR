<h1 align="center"> <p> SMVR</p></h1>
<h2 align="center"> Robust Multiview Point Cloud Registration using Algebraic Connectivity and Spatial Compatibility </h2>

In this paper, a spectral method for multiview point cloud registration is presented. The primary challenge of existing multiview registration methods is to accurately estimate the weights of the pose graph. Previous multiview registration methods rely on predicting the overlap between pairs of point clouds to assign weights for pairwise transformation matrices. This global way may result in higher weights being assigned to pairs of point clouds that are structurally similar but actually have no or low overlap. In addition, the learning-based approach is biased for training scenarios. However, we obtain the weights by measuring the confidence of each correspondence used for the estimation of pairwise transformation matrix. Specifically, we construct spatial compatibility matrices for the correspondences and compute the edge weights of the pose graphs by means of spectral decomposition. Considering the compatibility between correspondences, the proposed method is robust to outliers. Simultaneously, to mitigate the disruption of outliers to the global pose estimation and enhance the computational efficiency, we propose an overlap score estimation method based on the algebraic connectivity of the graph for pruning of fully connected pose graphs. For the initial sparse pose graph, we apply Iteratively Reweighted Least Square (IRLS) to refine the global transformation and design a new reweighting function based on historical weighted average. The non-learning framework empowers the proposed method to have better generalization to unknown scenarios. The experimental results further validate the performance of our method which achieves about 12.9% and 21.4% lower rotation and translation errors on the ScanNet dataset compared to the state-of-the-art method.

## Prerequisites

This code has been tested on PyTorch 1.7.1 (w/ Cuda 10.2) with PyTorch Geometric 1.7.1. Note that our code currently does not support PyTorch Geometric v2. You can install the required packages by running:

```bash
pip install -r requirements.txt
```

## 📚 Citation
Please consider citing this paper if you find it benefit your work:

```
@article{fang2024robust,
  title={Robust Multiview Point Cloud Registration using Algebraic Connectivity and Spatial Compatibility},
  author={Fang, Li and Li, Tianyu and Zhou, Shudong and Lin, Yanghong},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
