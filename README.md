<img src="Anomaly.png" width="100">

# flex-anomalies 
flex-anomalies is a Python library dedicated to anomaly detection in machine learning. It offers a wide range of algorithms and techniques, including models based on distance, density, trees, and neural networks such as convolutional and recurrent architectures. The library also provides aggregators, anomaly score processing techniques, and pre-processing techniques for data. 

Anomaly detection involves examining data and detecting deviations or anomalies present in the data, with the goal of purifying data sets and identifying anomalies for further analysis.


### Details

Anomaly Detection with <a href=https://github.com/FLEXible-FL/FLEXible/tree/main>FLEXible</a> Federated Learning: This repository contains implementations of anomaly detection algorithms using the Flexible Federated Learning library. <a href=https://github.com/FLEXible-FL/FLEXible/tree/main>FLEXible</a> is a Python library for realizing federated learning in an efficient and scalable manner. 
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
- **AnomalyDetection_AutoEncoder_FLEX_ts.ipynb**: Notebook showing a step-by-step example of how to use the Auto Encoder model for anomaly detection with federated learning for time series.The structure of the sliding window, data federation, federated training and model evaluation at the server and client level.
- **AnomalyDetection_PCA_FLEX.ipynb**: A notebook demonstrating the application of PCA_Anomaly for anomaly detection with federated learning for a static dataset.
- **AnomalyDetection_Cluster_FLEX.ipynb**: Notebook showing a step-by-step example of how to use the ClusterAnomaly model for anomaly detection with federated learning for static data and evaluating the model on test sets. 
- **AnomalyDetection_IsolationForest_FLEX.ipynb**: Notebook showing an example of how to use the IsolationForest  model with federated learning for an example set of static data.  From data federation and training to model evaluation on a test set.
- **AnomalyDetection_CNNN_LSTM_FLEX_ts.ipynb**: Notebook showing the use of the DeepCNN_LSTM model with federated learning for anomaly detection in time series. The structure of the sliding window, data federation, federated training and model evaluation at server and client level.

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
            <a href=https://ieeexplore.ieee.org/document/4781136>
            Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. In *International Conference on Data Mining*\ , pp. 413-422. IEEE.
            </td>    
        </tr>
        <tr>
            <td rowspan= 1>PCA_Anomaly</td>
            <td rowspan=1 align="center"> 
            Principal component analysis (PCA), algorithm for detecting outlier.Outlier scores can be obtained as  the sum of weighted euclidean distance between each sample to the hyperplane constructed by the selected eigenvectors
            </td>
            <td>
            <a href=https://www.researchgate.net/publication/228709094_A_Novel_Anomaly_Detection_Scheme_Based_on_Principal_Component_Classifier>
            Shyu, M.L., Chen, S.C., Sarinnapakorn, K. and Chang, L., 2003. A novel anomaly detection scheme based on principal component classifier. *MIAMI UNIV CORAL GABLES FL DEPT OF ELECTRICAL AND COMPUTER ENGINEERING*.
            </td>    
        </tr>
        <tr>
            <td rowspan= 1>ClusterAnomaly</td>
            <td rowspan=1 align="center"> 
               Model based on clustering. Outliers scores are solely computed based on their distance to the closest large cluster center, kMeans is used for clustering algorithm.
            </td>
            <td>
            <a href=https://epubs.siam.org/doi/10.1137/1.9781611972832.21>
            Chawla, S., & Gionis, A. (2013, May). k-means–: A unified approach to clustering and outlier detection. In Proceedings of the 2013 SIAM international conference on data mining (pp. 189-197).
            </td>    
        </tr>
        <tr>
            <td rowspan= 1>DeepCNN_LSTM</td>
            <td rowspan=1 align="center"> 
            Neural network model for time series and static data including convolutional and recurrent architecture.
            </td>
            <td>
            <a href=https://arxiv.org/abs/2206.03179>
            Aguilera-Martos, I., García-Vico, Á. M., Luengo, J., Damas, S., Melero, F. J., Valle-Alonso, J. J., & Herrera, F. (2022). TSFEDL: A Python Library for Time Series Spatio-Temporal Feature Extraction and Prediction using Deep Learning (with Appendices on Detailed Network Architectures and Experimental Cases of Study). arXiv preprint arXiv:2206.03179.
            </td>    
        </tr>
        <tr>
            <td rowspan= 1>AutoEncoder</td>
            <td rowspan=1 align="center"> 
             Fully connected AutoEncoder for time series and static data. Neural network for learning useful data   representations unsupervisedly. detect  anomalies in the data by calculating the reconstruction.
            </td>
            <td>
            <a href=https://link.springer.com/chapter/10.1007/978-3-319-14142-8_8>
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


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this repository in your research work, please cite the Flexible paper: 
