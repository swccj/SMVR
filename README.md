<h1 align="center"> <p> SMVR</p></h1>
<h2 align="center"> Robust Multiview Point Cloud Registration using Algebraic Connectivity and Spatial Compatibility </h2>

In this paper, a spectral method for multiview point cloud registration is presented. The primary challenge of existing multiview registration methods is to accurately estimate the weights of the pose graph. Previous multiview registration methods rely on predicting the overlap between pairs of point clouds to assign weights for pairwise transformation matrices. This global way may result in higher weights being assigned to pairs of point clouds that are structurally similar but actually have no or low overlap. In addition, the learning-based approach is biased for training scenarios. However, we obtain the weights by measuring the confidence of each correspondence used for the estimation of pairwise transformation matrix. Specifically, we construct spatial compatibility matrices for the correspondences and compute the edge weights of the pose graphs by means of spectral decomposition. Considering the compatibility between correspondences, the proposed method is robust to outliers. Simultaneously, to mitigate the disruption of outliers to the global pose estimation and enhance the computational efficiency, we propose an overlap score estimation method based on the algebraic connectivity of the graph for pruning of fully connected pose graphs. For the initial sparse pose graph, we apply Iteratively Reweighted Least Square (IRLS) to refine the global transformation and design a new reweighting function based on historical weighted average. The non-learning framework empowers the proposed method to have better generalization to unknown scenarios. The experimental results further validate the performance of our method which achieves about 12.9% and 21.4% lower rotation and translation errors on the ScanNet dataset compared to the state-of-the-art method.

## Prerequisites

This code has been tested on Ubuntu 18.04.6, CUDA 11.1.1, python 3.7, Pytorch 1.10.0, GeForce RTX A6000. You can install the required packages by running:

- First, create the conda environment:
```
conda create -n smvr python=3.7
conda activate smvr
pip install -r requirements.txt
```

- Second, intall Pytorch. We have checked version 1.10.0 and other versions can be referred to [Official Set](https://pytorch.org/get-started/previous-versions/).
  ```
  pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

- Third, install other packages, here we use 0.8.0.0 version [Open3d](http://www.open3d.org/):
  ```
  pip install -r requirements.txt
  ```

- Optional. If you want to use SMVR on your own dataset, you should install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) for FCGF/YOHO:
  ```
  conda install openblas-devel -c anaconda
  git clone https://github.com/NVIDIA/MinkowskiEngine.git
  cd MinkowskiEngine
  python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
  ```

## 💾 Dataset (with yoho-desc)
The datasets are accessible from [SGHR](https://github.com/WHU-USI3DV/SGHR/tree/master) in [BaiduDesk](https://pan.baidu.com/s/1FcAPjmrsJ6EEPLbtf85Irw)(Code:oouk) and Google Cloud:

Testset:
- [3DMatch/3DLomatch](https://drive.google.com/file/d/1T9fyU2XAYmXwiWZif--j5gP9G8As5cxn/view?usp=sharing);
- [ScanNet](https://drive.google.com/file/d/1GM6ePDDqZ3awJOZpctd3nqy1VgazV6CD/view?usp=sharing);
- [ETH](https://drive.google.com/file/d/1MW8SV44fuFTS5b2XrdADaqH5xRf3sLMk/view?usp=sharing).

Please place the data to ```./data``` following the example data structure as:

```
data/
├── 3dmatch/
    └── kitchen/
        ├── PointCloud/
            ├── cloud_bin_0.ply
            ├── gt.log
            └── gt.info
        ├── yoho_desc/
            └── 0.npy
        └── Keypoints/
            └── cloud_bin_0Keypoints.txt
├── 3dmatch_train/
├── scannet/
└── ETH/
```

## ✏️ Test
To evalute SMVR on 3DMatch and 3DLoMatch, you can use the following commands:
```
python Test_cycle.py --dataset 3dmatch --rr
```

To evalute SMVR on ScanNet, you can use the following commands:
```
python Test_cycle.py --dataset scannet --ecdf
```

To evalute SMVR on ETH, you can use the following commands:
```
python Test_cycle.py --dataset ETH --topk 6 --inlierd 0.2 --tau_2 0.5 --rr
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

## 🔗 Related Projects
We sincerely thank the fantastic projects:
- [SGHR](https://github.com/WHU-USI3DV/SGHR/tree/master);
- [YOHO](https://github.com/HpWang-whu/YOHO);
- [FCGF](https://github.com/chrischoy/FCGF);

