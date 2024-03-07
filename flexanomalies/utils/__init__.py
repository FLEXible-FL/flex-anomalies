#Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


from flexanomalies.utils.base_model import BaseModel
from flexanomalies.utils.autoencoder import AutoEncoder
from flexanomalies.utils.cnn_lstm import DeepCNN_LSTM
from flexanomalies.utils.cluster  import ClusterAnomaly
from flexanomalies.utils.iforest  import IsolationForest
from flexanomalies.utils.pca_anomaly  import PCA_Anomaly
from flexanomalies.utils.process_scores import process_scores,process_scores_with_percentile,process_scores_with_threshold