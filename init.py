from random import randint
from cv2 import imshow, waitKey, destroyAllWindows, error
from csv import writer
import wx
from backend_utils import data_loader, preprocessor, feature_extractor, models

read_mode_list = ['Raw image', 'Cropped image']

acc_path = 'output\\accuracy_list.csv'


class Controller:
    def __init__(self):
        self.imgs, self.labels = [], []
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []
        self.X_features = []

        self.model = None
        self.model_name = ''
        self.acc = 0

        self.cropped = 0
        self.comment = ''
        self.feature_info = ''

        self.csv_writer = None

    def load_data(self, cropped):
        if cropped: self.cropped = 1
        self.imgs, self.labels = data_loader.load_img_label(cropped)
        return 0 if len(self.imgs) == data_loader.total_num else -1

    def cvt_2_gray(self):
        self.imgs = preprocessor.cvt_2_gray(self.imgs)
        self.comment += '-gray scale'

    def gaussian_blur(self, k=3):
        self.imgs = preprocessor.gaussian_blur(self.imgs, ksize=(k, k))
        self.comment += '-gaussian blur'

    def laplacian(self, k=3):
        self.imgs = preprocessor.laplacian(self.imgs, ksize=k)
        self.comment += '-laplacian'

    def resize(self, w, h):
        self.imgs = preprocessor.resize(self.imgs, width=w, height=h)
        self.comment += '-resize to {w}*{h}'.format(w=w, h=h)

    def value_equalize(self):
        self.imgs = preprocessor.value_equalize(self.imgs)
        self.comment += 'value equalize'

    def bilateral(self, d=9, sigmaColor=75, sigmaSpace=75):
        self.imgs = preprocessor.bilateral(self.imgs, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        self.comment += 'bilateral filter'

    def show(self):
        index = randint(0, data_loader.total_num - 1)
        imshow(str(randint), self.imgs[index])
        waitKey()
        destroyAllWindows()

    def vectorize(self):
        self.X_features = feature_extractor.vectorize(self.imgs)
        self.feature_info = 'Flatten image'

    def histogram_of_gradient(self):
        self.X_features = feature_extractor.histogram_of_gradient(self.imgs)
        self.feature_info = 'HOG'

    def local_binary_pattern(self, P=8, R=1.0):
        self.X_features = feature_extractor.local_binary_pattern(self.imgs, P=P, R=R)
        self.feature_info = 'Local binary pattern'

    def canny(self, sigma=3):
        self.X_features = feature_extractor.canny(self.imgs, sigma=sigma)
        self.feature_info = 'Canny'

    def build_SVC(self, gamma='auto'):
        self.model = models.build_SVC(gamma=gamma)
        self.model_name = 'SVC_{gamma}'.format(gamma=gamma)

    def build_KNN(self, k=5):
        self.model = models.KNeighborsClassifier(n_neighbors=k)
        self.model_name = 'KNN_{k}'.format(k=k)

    def build_RFC(self, n_estimators=100, max_depth=13):
        self.model = models.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.model_name = 'RFC_{n_estimators}_{max_depth}'.format(n_estimators=n_estimators, max_depth=max_depth)

    def build_DTC(self, min_samples_split=3, min_samples_leaf=1):
        self.model = models.DecisionTreeClassifier(min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf)
        self.model_name = 'RFC_{inner}_{leaf}'.format(inner=min_samples_split, leaf=min_samples_leaf)

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = data_loader.img_split(self.X_features, self.labels)
        self.model.fit(self.X_train, self.y_train)
        self.acc = self.model.score(self.X_test, self.y_test, sample_weight=None)
        return self.model_name, self.acc

    def save_model(self):
        models.save_model(self.model, self.model_name)

    def save_accuracy(self):
        f = open(acc_path, 'a', encoding='utf-8')
        self.csv_writer = writer(f)
        self.csv_writer.writerow([self.model_name, self.acc, self.feature_info, self.cropped, self.comment])


class StartPanel(wx.Frame):

    def __init__(self, parent=None, fid=-1):
        wx.Frame.__init__(self, parent, fid, 'Traffic sign project - Group 10', size=(340, 240),
                          style=wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        self.Center()
        self.panel = wx.Panel(self)
        self.items = []

        self.btn_start = wx.Button(self.panel, -1, 'Start!', pos=(40, 30), size=(240, 60), style=wx.EXPAND)

        self.read_mode = wx.RadioBox(self.panel, -1, "Choose your read mode", (40, 110), wx.DefaultSize,
                                     read_mode_list, 2, wx.RA_SPECIFY_COLS)

        self.btn_start.Bind(wx.EVT_LEFT_DOWN, self.start)

    def start(self, event):
        wx.StaticText(self.panel, -1, 'Loading...Please wait!', pos=(40, 180), size=(200, 50))
        val = controller.load_data(cropped=False if self.read_mode.GetStringSelection() == read_mode_list[0] else True)
        if val != -1:
            wx.MessageBox('Images & labels loaded!\nNow heading for Preprocessing Phase.', 'Prompt',
                          wx.ICON_INFORMATION)
        else:
            wx.MessageBox('Fail to load Images & labels.', 'ERROR', wx.ICON_ERROR)
            self.Destroy()
        preprocessor_panel = PreprocessPanel()
        preprocessor_panel.Show()
        self.Destroy()


class PreprocessPanel(wx.Frame):

    def __init__(self, parent=None, fid=-1):
        wx.Frame.__init__(self, parent, fid, 'Preprocessing Phase - Group 10', size=(650, 480),
                          style=wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        self.Center()
        self.panel = wx.Panel(self)

        self.have_a_look_btn = wx.Button(self.panel, -1, 'Have a look!', pos=(0, 410), size=(150, 30), style=wx.EXPAND)
        self.next_btn = wx.Button(self.panel, -1, 'Next', pos=(480, 410), size=(150, 30), style=wx.EXPAND)
        self.grey_btn = wx.Button(self.panel, -1, 'Grey scale change', pos=(50, 30), size=(150, 30), style=wx.EXPAND)
        self.gaussian_btn = wx.Button(self.panel, -1, 'Gaussian blur', pos=(250, 30), size=(150, 30), style=wx.EXPAND)
        self.laplacian_btn = wx.Button(self.panel, -1, 'Laplacian', pos=(450, 30), size=(150, 30), style=wx.EXPAND)
        self.resize_btn = wx.Button(self.panel, -1, 'Resizing', pos=(50, 80), size=(150, 30), style=wx.EXPAND)
        self.val_equ_btn = wx.Button(self.panel, -1, 'V channel equalization', pos=(250, 80), size=(150, 30),
                                     style=wx.EXPAND)
        self.bilateral_btn = wx.Button(self.panel, -1, 'Bilateral filter', pos=(450, 80), size=(150, 30),
                                       style=wx.EXPAND)

        self.info = None

        self.grey_btn.Bind(wx.EVT_LEFT_DOWN, self.change_gray_scale)
        self.have_a_look_btn.Bind(wx.EVT_LEFT_DOWN, self.have_a_look)
        self.gaussian_btn.Bind(wx.EVT_LEFT_DOWN, self.gaussian)
        self.laplacian_btn.Bind(wx.EVT_LEFT_DOWN, self.laplacian)
        self.resize_btn.Bind(wx.EVT_LEFT_DOWN, self.resize)
        self.val_equ_btn.Bind(wx.EVT_LEFT_DOWN, self.val_equ)
        self.bilateral_btn.Bind(wx.EVT_LEFT_DOWN, self.bilateral)
        self.next_btn.Bind(wx.EVT_LEFT_DOWN, self.next)

    def input_paremeter(self, name, default):
        dlg = wx.TextEntryDialog(self.panel, name, 'Input parameter')
        dlg.SetValue(str(default))
        if dlg.ShowModal() == wx.ID_OK:
            return dlg.GetValue()

    def have_a_look(self, event):
        controller.show()

    def change_gray_scale(self, event):
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        controller.cvt_2_gray()
        self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))

    def gaussian(self, event):
        k = int(self.input_paremeter('Kernel size (ODD number!)', 3))
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        controller.gaussian_blur(k)
        self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))

    def laplacian(self, event):
        k = int(self.input_paremeter('Kernel size', 3))
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        controller.laplacian(k)
        self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))

    def resize(self, event):
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        w = int(self.input_paremeter('Width', 48))
        h = int(self.input_paremeter('Height', 48))
        controller.resize(w, h)
        self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))

    def val_equ(self, event):
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        try:
            controller.value_equalize()
            self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))
        except error:
            wx.MessageBox('Please do this BEFORE gray scale change!!', 'ERROR', wx.ICON_ERROR)

    def bilateral(self, event):
        d = int(self.input_paremeter('d', 9))
        sigmaColor = int(self.input_paremeter('sigmaColor', 75))
        sigmaSpace = int(self.input_paremeter('sigmaSpace', 75))
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        controller.bilateral(d, sigmaColor, sigmaSpace)
        self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))

    def next(self, event):
        feature_extract_panel = FeatureExtractionPanel()
        feature_extract_panel.Show()
        self.Destroy()


class FeatureExtractionPanel(wx.Frame):

    def __init__(self, parent=None, fid=-1):
        wx.Frame.__init__(self, parent, fid, 'Feature Extracting Phase - Group 10', size=(650, 480),
                          style=wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        self.Center()
        self.panel = wx.Panel(self)

        self.vectorize_btn = wx.Button(self.panel, -1, 'Just flatten the image', pos=(50, 30), size=(150, 30),
                                       style=wx.EXPAND)
        self.hog_btn = wx.Button(self.panel, -1, 'HOG', pos=(250, 30), size=(150, 30), style=wx.EXPAND)
        self.lbp_btn = wx.Button(self.panel, -1, 'Local binary pattern', pos=(450, 30), size=(150, 30), style=wx.EXPAND)
        self.canny_btn = wx.Button(self.panel, -1, 'Canny', pos=(50, 80), size=(150, 30), style=wx.EXPAND)

        self.info = None

        self.vectorize_btn.Bind(wx.EVT_LEFT_DOWN, self.vectorize)
        self.hog_btn.Bind(wx.EVT_LEFT_DOWN, self.hog)
        self.lbp_btn.Bind(wx.EVT_LEFT_DOWN, self.lbp)
        self.canny_btn.Bind(wx.EVT_LEFT_DOWN, self.canny)

    def input_paremeter(self, name, default):
        dlg = wx.TextEntryDialog(self.panel, name, 'Input parameter')
        dlg.SetValue(str(default))
        if dlg.ShowModal() == wx.ID_OK:
            return dlg.GetValue()

    def vectorize(self, event):
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        controller.vectorize()
        self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))
        wx.MessageBox('Got the flattened images as vector features!\n' +
                      'Now heading for the model selection phase.', 'Prompt', wx.ICON_INFORMATION)
        self.next()

    def hog(self, event):
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        controller.histogram_of_gradient()
        self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))
        wx.MessageBox('Got the histograms of gradient as features!\n' +
                      'Now heading for the model selection phase.', 'Prompt', wx.ICON_INFORMATION)
        self.next()

    def lbp(self, event):
        P = int(self.input_paremeter('Number of circularly symmetric neighbour set points:', 8))
        R = float(self.input_paremeter('Radius of circle:', 1.0))
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        controller.local_binary_pattern(P=P, R=R)
        self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))
        wx.MessageBox('Got the local binary patterns as vector features!\n' +
                      'Now heading for the model selection phase.', 'Prompt', wx.ICON_INFORMATION)
        self.next()

    def canny(self, event):
        sigma = int(self.input_paremeter('Standard deviation of the Gaussian filter(ODD number):', 3))
        self.info = wx.StaticText(self.panel, -1, 'Processing...Please wait!', pos=(0, 0), size=(200, 25))
        controller.canny(sigma=sigma)
        self.info = wx.StaticText(self.panel, -1, 'Done!', pos=(0, 0), size=(200, 25))
        wx.MessageBox('Got the canny edges as vector features!\n' +
                      'Now heading for the model selection phase.', 'Prompt', wx.ICON_INFORMATION)
        self.next()

    def next(self):
        model_selection_panel = ModelSelectionPanel()
        model_selection_panel.Show()
        self.Destroy()


class ModelSelectionPanel(wx.Frame):

    def __init__(self, parent=None, fid=-1):
        wx.Frame.__init__(self, parent, fid, 'Model Selection Phase - Group 10', size=(650, 480),
                          style=wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        self.Center()
        self.panel = wx.Panel(self)

        self.SVC_btn = wx.Button(self.panel, -1, 'SVC', pos=(50, 30), size=(150, 30), style=wx.EXPAND)
        self.KNN_btn = wx.Button(self.panel, -1, 'KNN', pos=(250, 30), size=(150, 30), style=wx.EXPAND)
        self.RFC_btn = wx.Button(self.panel, -1, 'Random forest classifier', pos=(450, 30), size=(150, 30),
                                 style=wx.EXPAND)
        self.DTC_btn = wx.Button(self.panel, -1, 'Decision tree classifier', pos=(50, 80), size=(150, 30),
                                 style=wx.EXPAND)

        self.info = None

        self.SVC_btn.Bind(wx.EVT_LEFT_DOWN, self.SVC)
        self.KNN_btn.Bind(wx.EVT_LEFT_DOWN, self.KNN)
        self.RFC_btn.Bind(wx.EVT_LEFT_DOWN, self.RFC)
        self.DTC_btn.Bind(wx.EVT_LEFT_DOWN, self.DTC)

    def input_paremeter(self, name, default):
        dlg = wx.TextEntryDialog(self.panel, name, 'Input parameter')
        dlg.SetValue(str(default))
        if dlg.ShowModal() == wx.ID_OK:
            return dlg.GetValue()

    def SVC(self, event):
        gamma = self.input_paremeter('Gamma value of SVC:', 'auto')
        controller.build_SVC(gamma=gamma if gamma == 'auto' else float(gamma))
        wx.MessageBox('SVC successfully built.\n Let\'s train!', 'Prompt', wx.OK | wx.ICON_INFORMATION)
        self.next()

    def KNN(self, event):
        k = int(self.input_paremeter('k value of KNN:', 5))
        controller.build_KNN(k=k)
        wx.MessageBox('KNN classifier successfully built.\n Let\'s train!', 'Prompt', wx.OK | wx.ICON_INFORMATION)
        self.next()

    def RFC(self, event):
        n_estimators = int(self.input_paremeter('The number of trees in the forest:', 100))
        max_depth = int(self.input_paremeter('The maximum depth of the tree:', 13))
        controller.build_RFC(n_estimators=n_estimators, max_depth=max_depth)
        wx.MessageBox('Random forest classifier successfully built.\n Let\'s train!', 'Prompt',
                      wx.OK | wx.ICON_INFORMATION)
        self.next()

    def DTC(self, event):
        inner = int(self.input_paremeter('The minimum number of samples required to split an internal node:', 3))
        leaf = int(self.input_paremeter('The minimum number of samples required to be at a leaf node:', 1))
        controller.build_DTC(min_samples_split=inner, min_samples_leaf=leaf)
        wx.MessageBox('Decision tree classifier successfully built.\n Let\'s train!', 'Prompt', wx.OK | wx.ICON_INFORMATION)
        self.next()

    def next(self):
        train_test_panel = TrainTestPanel()
        train_test_panel.Show()
        self.Destroy()


class TrainTestPanel(wx.Frame):

    def __init__(self, parent=None, fid=-1):
        wx.Frame.__init__(self, parent, fid, 'Model Selection Phase - Group 10', size=(650, 480),
                          style=wx.CAPTION | wx.CLOSE_BOX | wx.MINIMIZE_BOX)
        self.Center()
        self.panel = wx.Panel(self)

        self.model_name = ''
        self.acc = 0

        self.info = None

        self.train_btn = wx.Button(self.panel, -1, 'Train!', pos=(50, 30), size=(550, 200), style=wx.EXPAND)
        self.save_model_btn = wx.Button(self.panel, -1, 'Save model', pos=(50, 250), size=(150, 30), style=wx.EXPAND)
        self.save_acc_btn = wx.Button(self.panel, -1, 'Save accuracy', pos=(450, 250), size=(150, 30), style=wx.EXPAND)

        self.train_btn.Bind(wx.EVT_LEFT_DOWN, self.train)
        self.save_model_btn.Bind(wx.EVT_LEFT_DOWN, self.save_model)
        self.save_acc_btn.Bind(wx.EVT_LEFT_DOWN, self.save_acc)

    def train(self, event):
        self.info = wx.StaticText(self.panel, -1, 'Training...Please wait!', pos=(0, 0), size=(500, 25))
        self.model_name, self.acc = controller.train()
        self.info = wx.StaticText(self.panel, -1, 'Complete! Model {name}\'s accuracy is {acc}'.
                                  format(name=self.model_name, acc=self.acc), pos=(0, 0), size=(500, 25))
        return

    def save_model(self, event):
        controller.save_model()
        self.info = wx.StaticText(self.panel, -1, 'Model saved!', pos=(0, 0), size=(500, 25))

    def save_acc(self, event):
        controller.save_accuracy()
        self.info = wx.StaticText(self.panel, -1, 'Model\'s accuracy saved!', pos=(0, 0), size=(500, 25))


if __name__ == '__main__':
    controller = Controller()
    app = wx.App(False)
    facade = StartPanel()
    facade.Show()
    app.MainLoop()
    # imgs, labels = data_loader.load_img_label(cropped=True)
    #
    # imgs = preprocessor.resize(imgs, width=48, height=48)
    # imgs = preprocessor.value_equalize(imgs)
    # imgs = preprocessor.cvt_2_gray(imgs)
    #
    # import cv2
    # cv2.imshow('0', imgs[0])
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # import numpy as np
    # print(np.shape(feature_extractor.canny(imgs)))
    # X_features = feature_extractor.histogram_of_gradient(imgs)
    #
    # model_list += [models.build_KNN(), models.build_RFC(), models.build_SVC(), models.build_DTC()]
    #
    # X_train, X_test, y_train, y_test = data_loader.img_split(X_features, labels)
    #
    # for model in model_list:
    #     model.fit(X_train, y_train)
    #     acc = model.score(X_test, y_test, sample_weight=None)
    #     print(acc)
