import sys
from UI.main import Ui_MainWindow
from vgg_classify import Classify
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import *


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.initialize()

    def initialize(self):
        self.setWindowTitle("Bvh Texture Classifier")
        self.img_path = ""
        self.label = ""
        self.precision = 0.0
        #self.label_showImage.setPixmap(QPixmap("UI/bg.jpg"))

        # button slot
        self.pushButton_openImage.clicked.connect(lambda: self.btn_openImage())
        self.pushButton_classify.clicked.connect(lambda: self.btn_classify())
        self.menuExit.triggered.connect(lambda: self.btn_exit())

        # load model
        self.model = Classify("checkpoint/epoch_42-best-acc_0.9312499761581421.pth", 64, 8, False)
        self.textBrowser.append("Initialization is complete!\n")
        print("Initialization is complete")

    def btn_openImage(self):
        print("openImage")
        self.img_path, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image File (*.jpg; *.png; *.Tiff)')
        print(self.img_path)
        self.textBrowser.append("Load Image： \n" + str(self.img_path))
        #self.label_openedImage.setText(str(self.img_path))
        self.label_showImage.setPixmap(QPixmap(self.img_path))

    def btn_exit(self):
        sys.exit()


    def btn_classify(self):
        if self.img_path == "":
            print("Image not imported")
            self.textBrowser.append("Image not imported")
        else:
            self.precision, self.label, self.text = self.model.classify(self.img_path)
            self.label_class.setText('label: ' + str(self.label))
            self.textBrowser.append('Label: ' + str(self.label))
            self.textBrowser.append('precision: ' + str(self.precision))
            self.textBrowser.append(str(self.text))
            self.textBrowser.append('\n')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
