"""
Module dedicated to the generation and retrieval of classical and quantum datasets.

Args:
    the_dataset_register: singleton global instance of DatasetRegister
"""


import openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
import pywt


def download_dataset_openml(the_id):
    """
    Download a dataset from OpenML platform given the ID of the dataset.

    Args:
        the_id: ID of the dataset (int)

    Returns:
        tuple (X, y) of numpy ndarray having shapes (d,n) and (n,)
    """
    metadata = openml.datasets.get_dataset(the_id)
    # get dataset
    X, y, _, attribute_names = metadata.get_data(
        dataset_format="array", target=metadata.default_target_attribute
    )
    return X, y


def download_dataset_openml_structured(the_id, format="array"):
    if Path(f"dataset_{the_id}.npy").exists():
        return np.load(f"dataset_{the_id}.npy", allow_pickle=True).item()

    metadata = openml.datasets.get_dataset(the_id)
    # get dataset
    X, y, _, attribute_names = metadata.get_data(
        dataset_format=format, target=metadata.default_target_attribute
    )
    data = {"X": X, "y": y}
    np.save(f"dataset_{the_id}.npy", data)
    return data


def download_dataset_openml_by_name(name):
    """
    Download a dataset from OpenML platform given the name of the dataset

    Args:
        name: name of the dataset (str)

    Returns:
        tuple (X, y) of numpy ndarray having shapes (d,n) and (n,)
    """
    # get the list of all datasets in OpenML platform
    openml_df = openml.datasets.list_datasets(output_format="dataframe")
    # retrieve the dataset id
    the_id = int(openml_df.query(f'name == "{name}"').iloc[0]["did"])
    return download_dataset_openml_by_name(the_id)


def get_resource(filename):
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources

    from . import resources
    return pkg_resources.open_text(resources, filename)


def get_dataset_quantum(the_id):
    """
    This function calls already preprocessed datasets with quantum labels.
    These examples are identified with a specific id.
    
    The available datasets at the moment are:

        - Fashion-MNIST with 2 features and encoding E1
        - Fashion-MNIST with 4 features and encoding E2
        - Fashion-MNIST with 8 features and encoding E3
    
    These datasets can be used to benchmark the performace of our
    classical and quantum kernels to verify the power of data.

    Args:
        the_id: parameter able to distinguish between quantum dataset

    Returns:
        tuple (X, y) of numpy ndarray having shapes (d,n) and (n,)
    """

    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources

    from . import resources

    if the_id == 0:
        X = np.loadtxt(
            pkg_resources.open_text(resources, "X_Q_Fashion-MNIST_720_2_E1"),
            delimiter=" ",
        )
        y = np.loadtxt(
            pkg_resources.open_text(resources, "y_Q_Fashion-MNIST_720_2_E1"),
            delimiter=" ",
        )
    elif the_id == 1:
        X = np.loadtxt(
            pkg_resources.open_text(resources, "X_Q_Fashion-MNIST_720_4_E2"),
            delimiter=" ",
        )
        y = np.loadtxt(
            pkg_resources.open_text(resources, "y_Q_Fashion-MNIST_720_4_E2"),
            delimiter=" ",
        )
    elif the_id == 2:
        X = np.loadtxt(
            pkg_resources.open_text(resources, "X_Q_Fashion-MNIST_720_8_E3"),
            delimiter=" ",
        )
        y = np.loadtxt(
            pkg_resources.open_text(resources, "y_Q_Fashion-MNIST_720_8_E3"),
            delimiter=" ",
        )

    return X, y


def create_gaussian_mixtures(D, noise, N):
    """
    Create the Gaussian mixture dataset
    :param D: number of dimensions: (x1, x2, 0, .., 0) in R^D
    :param noise: intensity of the random noise (mean 0)
    :param N: number of elements to generate
    :return: dataset
    """
    if N % 4 != 0:
        raise ValueError("The number of elements within the dataset must be a multiple of 4")
    if D < 2:
        raise ValueError("The number of dimensions must be at least 2")
    if noise < 0:
        raise ValueError("Signal to noise ratio must be > 0")

    X = np.zeros((N, D))
    Y = np.zeros((N,))
    centroids = np.array([(.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5)])
    for i in range(N):
        quadrant = i % 4
        Y[i] = 1 if quadrant % 2 == 0 else -1  # labels are 0 or 1
        X[i][0], X[i][1] = centroids[quadrant] + np.random.uniform(-noise, noise, size=(2,))
    return X, Y


def process_regression_dataset(dataset, n_components, n_elements=None, test_size=0.50, random_state=42):
    """
    Process dataset
    :param dataset: dict with keys 'X', 'y' as np arrays
    :param n_components: number of features to be extracted with PCA
    :param n_elements: number of elements to be randomly extracted
    :param random_state: random state for train test split
    return X_train, X_test, y_train, y_test
    """
    X, y = dataset['X'], dataset['y']
    if n_elements is not None and n_elements < len(X):
        selected_samples = np.random.choice(len(X), n_elements, replace=False)
        X = X[selected_samples]
        y = y[selected_samples]
    X = PCA(n_components=n_components).fit_transform(X)
    # y = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y)
    X = X - np.mean(X)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = y - np.mean(y)
    y = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def load_ols_cancer_dataset():
    dataset = pd.read_csv(get_resource("cancer_reg.csv"))
    dataset.binnedInc = dataset.binnedInc.apply(lambda value: eval(value.replace("[", "(").replace("]", ")"))[0])
    places = list(set(dataset.Geography))
    dataset.Geography.replace(places, range(len(places)), inplace=True)
    dataset = dataset.replace(np.nan, 0)
    target = dataset.TARGET_deathRate
    data = dataset.drop(columns=['TARGET_deathRate'], axis=1)
    X = data.to_numpy().astype(float)
    y = target.to_numpy().reshape(-1, 1)
    data = {"X": X, "y": y}
    return data


def load_fish_market_dataset():
    dataset = pd.read_csv(get_resource("Fish.csv"))
    places = list(set(dataset.Species))  # categorical -> numerical
    dataset.Species.replace(places, range(len(places)), inplace=True)
    dataset = dataset.replace(np.nan, 0)  # remove NaN
    target = dataset.Weight  # get target
    data = dataset.drop(columns=['Weight'], axis=1)
    X = data.to_numpy().astype(float)
    y = target.to_numpy().reshape(-1, 1)
    data = {"X": X, "y": y}
    return data


def load_medical_bill_dataset():
    dataset = pd.read_csv(get_resource("insurance.csv"))
    places = list(set(dataset.sex))  # categorical -> numerical
    dataset.sex.replace(places, range(len(places)), inplace=True)
    places = list(set(dataset.smoker))  # categorical -> numerical
    dataset.smoker.replace(places, range(len(places)), inplace=True)
    places = list(set(dataset.region))  # categorical -> numerical
    dataset.region.replace(places, range(len(places)), inplace=True)
    dataset = dataset.replace(np.nan, 0)  # remove NaN
    target = dataset.charges  # get target
    data = dataset.drop(columns=['charges'], axis=1)
    X = data.to_numpy().astype(float)
    y = target.to_numpy().reshape(-1, 1)
    data = {"X": X, "y": y}
    return data


def load_real_estate_dataset():
    dataset = pd.read_csv(get_resource("Real estate.csv"))
    dataset.drop(columns=['No'], axis=1)
    dataset = dataset.replace(np.nan, 0)  # remove NaN
    target = dataset['Y house price of unit area']  # get target
    data = dataset.drop(columns=['Y house price of unit area'], axis=1)
    X = data.to_numpy().astype(float)
    y = target.to_numpy().reshape(-1, 1)
    data = {"X": X, "y": y}
    return data


def load_wine_quality_dataset():
    return download_dataset_openml_structured(40498)


def load_who_life_expectancy_dataset():
    dataset = download_dataset_openml_structured(43505, format="dataframe")['X']
    places = list(set(dataset.country))  # categorical -> numerical
    dataset.country.replace(places, range(len(places)), inplace=True)
    places = list(set(dataset.country_code))  # categorical -> numerical
    dataset.country_code.replace(places, range(len(places)), inplace=True)
    places = list(set(dataset.region))  # categorical -> numerical
    dataset.region.replace(places, range(len(places)), inplace=True)
    dataset = dataset.replace(np.nan, 0)
    target = dataset.life_expect  # get target
    data = dataset.drop(columns=['life_expect'], axis=1)
    X = data.to_numpy().astype(float)
    y = target.to_numpy().reshape(-1, 1)
    data = {"X": X, "y": y}
    return data


def load_function_approximation_sin_squared():
    X = np.linspace(-1, 1, 100)
    y = np.sin(2 * X)**2
    data = {"X": np.concatenate([X, X]).reshape((100, 2)), "y": y.reshape((-1, 1))}
    return data


def load_function_approximation_step():
    X = np.linspace(-1, 1, 100)
    y = np.zeros(shape=(100,))
    y[X > 0] = 1
    y[X < 0] = -1
    X = X.reshape((-1, 1))
    data = {"X": np.concatenate([X, X]).reshape((100, 2)), "y": y.reshape((-1, 1))}
    return data


def load_function_approximation_meyer_wavelet():
    phi, psi, x = pywt.Wavelet('dmey').wavefun(level=5)
    psi_zoom = psi[(x > 25) & (x < 35)]
    x_zoom = x[(x > 25) & (x < 35)].ravel()
    data = {"X": np.concatenate([x_zoom, x_zoom]).reshape((len(x_zoom), 2)), "y": psi_zoom.reshape((-1, 1))}
    return data


class DatasetRegister:
    """
    List of datasets available in this module. The object is iterable.
    """

    def __init__(self):
        """
        Init method.

        Returns:
            None
        """
        self.datasets = []
        self.current = 0

    def register(self, dataset_name, dataset_type, information_nature, get_dataset):
        """
        Register a new dataset.

        Args:
            dataset_name: name of the dataset
            dataset_type: 'regression' or 'classification'
            information_nature: 'classical' or 'quantum'
            get_dataset: function pointer to a zero-parameter function returning (X, y)

        Returns:
            None
        """
        assert dataset_type in ["regression", "classification"]
        assert information_nature in ["classical", "quantum"]
        self.datasets.append(
            {
                "name": dataset_name,
                "type": dataset_type,
                "information": information_nature,
                "get_dataset": get_dataset,
            }
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.datasets):
            raise StopIteration
        self.current += 1
        return self.datasets[self.current - 1]

    def __len__(self):
        return len(self.datasets)


the_dataset_register = DatasetRegister()
the_dataset_register.register(
    "iris", "classification", "classical", lambda: download_dataset_openml(61)
)
the_dataset_register.register(
    "Fashion-MNIST", "classification", "classical", lambda: download_dataset_openml(40996),
)
the_dataset_register.register(
    "liver-disorders", "regression", "classical", lambda: download_dataset_openml(8)
)
the_dataset_register.register(
    "delta_elevators", "regression", "classical", lambda: download_dataset_openml(198)
)
the_dataset_register.register(
    "Q_Fashion-MNIST_2_E1", "regression", "quantum", lambda: get_dataset_quantum(0)
)
the_dataset_register.register(
    "Q_Fashion-MNIST_4_E2", "regression", "quantum", lambda: get_dataset_quantum(1)
)
the_dataset_register.register(
    "Q_Fashion-MNIST_8_E3", "regression", "quantum", lambda: get_dataset_quantum(2)
)