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


from flexanomalies.utils import BaseModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import cdist
import pickle


class PCA_Anomaly(BaseModel):

    """PCA with scikit-learn
    Parameters of sklearn's PCA model:
    n_components : int, float or 'mle', default=None
    n_selected_components : int, number of selected principal components, default=None
    random_state : int, RandomState instance or None, default None
    Others:
    preprocess : bool, If true, standardize features , optional (default=True)
    contamination : float in (0., 0.5), optional (default=0.1)
                    Contamination of the data set, the proportion of outliers in the data set.
    model_path: str optional (default= '')

    """

    def __init__(
        self,
        n_components=None,
        n_selected_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
        model_path="",
        contamination=0.1,
        preprocess=True,
        weighted=True,
    ) -> None:
        super(PCA_Anomaly, self).__init__(contamination=contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.weighted = (weighted,)
        self.preprocess = preprocess
        self.model_path = model_path
        self.scaler = StandardScaler()

        self.model = self._build_model()

    def _build_model(self):
        model = PCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )

        return model

    def fit(self, X, y=None):
        """
        X : numpy array of shape (samples, features)
        y:  Ignored in unsupervised methods
        """

        # Standardize data
        if self.preprocess:
            Xscaler = self.scaler.fit_transform(X)
        else:
            Xscaler = np.copy(X)

        self.model.fit(X=Xscaler, y=y)

    def predict(self, X, y=None):

        if self.preprocess:
            Xscaler = self.scaler.fit_transform(X)
        else:
            Xscaler = np.copy(X)

        # attributes from the sklearn PCA
        self.n_components_ = self.model.n_components_
        self.components_ = self.model.components_

        # selected num components
        if self.n_selected_components is None:
            self.n_selected_components_ = self.n_components_
        else:
            self.n_selected_components_ = self.n_selected_components

        self.w_components_ = np.ones(
            [
                self.n_components_,
            ]
        )
        if self.weighted:
            self.w_components_ = self.model.explained_variance_ratio_

        self.selected_components_ = self.components_[
            -1 * self.n_selected_components_ :, :
        ]
        self.selected_w_components_ = self.w_components_[
            -1 * self.n_selected_components_ :
        ]

        self.d_scores_ = np.sum(
            cdist(Xscaler, self.selected_components_) / self.selected_w_components_,
            axis=1,
        ).ravel()

        self.process_scores()
        return self

    def decision_function(self, X, y=None):
        """
        X : numpy array of shape (n_samples, n_features)

        Returns  anomaly scores : numpy array of shape (n_samples,)
                The anomaly score of the input samples.
        """
        return np.sum(
            cdist(X, self.selected_components_) / self.selected_w_components_, axis=1
        ).ravel()

    def load_model(self, model_path=""):
        self.model = pickle.load(
            open(
                f"{model_path}/model.pkl"
                if model_path
                else f"{self.model_path}/model.pkl",
                "rb",
            )
        )

    def save_model(self, model_path=""):
        if not model_path and not self.model_path:
            raise Exception("You must provide a path to save model")
        pickle.dump(
            self.model,
            open(
                f"{model_path}/model.pkl"
                if model_path
                else f"{self.model_path}/model.pkl",
                "wb",
            ),
        )
