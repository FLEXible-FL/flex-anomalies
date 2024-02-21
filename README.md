# flex-anomalies 
flex-anomalies is a Python library dedicated to anomaly detection in machine learning. It offers a wide range of algorithms and techniques, including models based on distance, density, trees, and neural networks such as convolutional and recurrent architectures. The library also provides aggregators, anomaly score processing techniques, and pre-processing techniques for data. 

Anomaly detection involves examining data and detecting deviations or anomalies present in the data, with the goal of purifying data sets and identifying anomalies for further analysis.

## Features
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
