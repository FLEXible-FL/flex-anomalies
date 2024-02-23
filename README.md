# flex-anomalies 
flex-anomalies is a Python library dedicated to anomaly detection in machine learning. It offers a wide range of algorithms and techniques, including models based on distance, density, trees, and neural networks such as convolutional and recurrent architectures. The library also provides aggregators, anomaly score processing techniques, and pre-processing techniques for data. 

Anomaly detection involves examining data and detecting deviations or anomalies present in the data, with the goal of purifying data sets and identifying anomalies for further analysis.


### Details

Anomaly Detection with FLEXible Federated Learning: This repository contains implementations of anomaly detection algorithms using the Flexible Federated Learning library. FLEXible is a Python library for realizing federated learning in an efficient and scalable manner. 
From the study of state-of-the-art research works on federated learning for network intrusion detection.

This repository also includes:
- An organized folder structure that makes it easy to navigate and understand the project.
- Explanatory notebooks showing practical examples and detailed explanations for the use of the library.

####  Folder structure
- **flexanomalies/pool**: Here are the aggregators and primitives for each of the models following the FLEXible structure.
- **flexanomalies/utils**: Contains the source code of the implementations of the anomaly detection algorithms, anomaly score processing techniques, metrics for the evaluation,
function to federate a centralized dataset using FLEXible and data loading.
- **flexanomalies/datasets**: some pre-processing techniques for data.
- **notebooks**: Contains explanatory notebooks showing how to use the anomaly detection algorithms on data.  

#### Explanatory Notebooks
- **AnomalyDetection_Autoencoder_FLEX.ipynb**: A notebook showing a step-by-step example of how to use Auto Encoder model for anomaly detection with federated learning for static data.
- **AnomalyDetection_Cluster_Cluster_FLEX.ipynb**: Notebook showing a step-by-step example of how to use the ClusterAnomaly model for anomaly detection with federated learning for static data and evaluating the model on test sets. 


## Features
For more information on the implemented algorithms see the table that follows:
<table>
    <thead>
        <tr>
            <th>Models</th>
            <th>Description</th>
            <th>Citation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan= 1>IsolationForest</td>
            <td rowspan=1 align="center"> 
             Algorithm for data anomaly detection, detects anomalies using binary trees. 
            </td>
            <td>
            <a href=>
            Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. In *International Conference on Data Mining*\ , pp. 413-422. IEEE.
            </td>    
        </tr>
        <tr>
            <td rowspan= 1>PCA_Anomaly</td>
            <td rowspan=1 align="center"> 
            Principal component analysis (PCA), algorithm for detecting outlier.Outlier scores can be obtained as  the sum of weighted euclidean distance between each sample to the hyperplane constructed by the selected eigenvectors
            </td>
            <td>
            <a href=>
            Shyu, M.L., Chen, S.C., Sarinnapakorn, K. and Chang, L., 2003. A novel anomaly detection scheme based on principal component classifier. *MIAMI UNIV CORAL GABLES FL DEPT OF ELECTRICAL AND COMPUTER ENGINEERING*.
            </td>    
        </tr>
        <tr>
            <td rowspan= 1>ClusterAnomaly</td>
            <td rowspan=1 align="center"> 
               Model based on clustering. Outliers scores are solely computed based on their distance to the closest large cluster center, kMeans is used for clustering algorithm.
            </td>
            <td>
            <a href=>
            Chawla, S., & Gionis, A. (2013, May). k-means–: A unified approach to clustering and outlier detection. In Proceedings of the 2013 SIAM international conference on data mining (pp. 189-197).
            </td>    
        </tr>
        <tr>
            <td rowspan= 1>DeepCNN_LSTM</td>
            <td rowspan=1 align="center"> 
            Neural network model for time series and static data including convolutional and recurrent architecture.
            </td>
            <td>
            <a href=>
            Aguilera-Martos, I., García-Vico, Á. M., Luengo, J., Damas, S., Melero, F. J., Valle-Alonso, J. J., & Herrera, F. (2022). TSFEDL: A Python Library for Time Series Spatio-Temporal Feature Extraction and Prediction using Deep Learning (with Appendices on Detailed Network Architectures and Experimental Cases of Study). arXiv preprint arXiv:2206.03179.
            </td>    
        </tr>
        <tr>
            <td rowspan= 1>AutoEncoder</td>
            <td rowspan=1 align="center"> 
             Fully connected AutoEncoder for time series and static data. Neural network for learning useful data   representations unsupervisedly. detect  anomalies in the data by calculating the reconstruction.
            </td>
            <td>
            <a href=>
            Aggarwal, C.C., 2015. Outlier analysis. In Data mining (pp. 237-263), Ch.3. Springer, Cham. Ch.3
            </td>    
        </tr>
    </tbody>

    
</table>

## Installation

FLEX-Anomalies is available on the PyPi repository and can be easily installed using: 

``` pip: pip install flex-anomalies ```

Install the necessary dependencies:

```pip install -r requirements.txt```

## Citation

If you use this repository in your research work, please cite the Flexible paper: 


