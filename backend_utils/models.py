from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib


from os.path import join
model_path = 'output\\models'


def build_SVC(gamma='auto'):
    return make_pipeline(StandardScaler(), SVC(gamma=gamma))


def build_KNN(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def build_RFC(n_estimators=100, max_depth=13):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)


def build_DTC(min_samples_split=3, min_samples_leaf=1):
    return DecisionTreeClassifier(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)


def save_model(model, model_name):
    joblib.dump(model, join(model_path, model_name)+'.pkl')
