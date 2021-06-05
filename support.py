

from PyQt5 import QtWidgets, QtCore, uic,QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QTableWidget,QVBoxLayout, QLineEdit, QGraphicsDropShadowEffect
from PyQt5.QtCore import QSortFilterProxyModel

counter =0
jumper =0
class Other (QMainWindow):
    def __init__(self):
        super(Other, self).__init__()
        uic.loadUi('design_2.ui',self)
        
        
        #self.shadow = QGraphicsDropShadowEffect(self)
        #shadow = QGraphicsDropShadowEffect(blurRadius=5, xOffset=4, yOffset=4)
        #self.circularBg.setGraphicsEffect(self.shadow)

        self.progressBarValue(0)
        #TEST PROGRESS BAR FUNCTION

        #self.progressBarValue(50)

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)

        self.timer.start(15)


    def progress (self):
        global counter
        global jumper

        value = counter
        
        htmlText = """<p><span style=" font-size:46pt;">{VALUE}</span><span style=" font-size:38pt; vertical-align:super;">%</span></p>"""
        newHtml = htmlText.replace("{VALUE}", str(jumper))

        if (value > jumper):

            self.labelPercentage.setText(newHtml)

            jumper +=10
        if value >=100: value = 1.000

        self.progressBarValue(value)

        # CLOSE SPLASH SCREE AND OPEN APP
        if counter > 100:
            # STOP TIMER
            self.timer.stop()

            # SHOW MAIN WINDOW
            #self.main = MainWindow()
            #self.main.show()

            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        counter += 1.5



    def progressBarValue(self,value):
    	#PORGRESSBAR STYLESHEET BASE

        styleSheet = """
        QFrame{
            border-radius: 100px;
            background-color: qconicalgradient(cx:0.5, cy:0.5, angle:90, stop:{STOP_1} rgba(255, 0, 127, 0), stop:{STOP_2} rgba(85, 170, 255, 255));
        }
        """
        
        #GET PROGRESS BAR VALUE, CONVERT TO FLOAT AND INVERT VALUE
        progress = (100 - value) /100.0

        #GET NEW VALUES
        stop_1 = str(progress - 0.001)
        stop_2 = str(progress)

        #SET VALUE TO NEW STYLESHEET

        newStylesheet = styleSheet.replace("{STOP_1}", stop_1).replace("{STOP_2}", stop_2)

        #APPLY STYLESHEET WITH NEW VALUES
        self.circularProgress.setStyleSheet(newStylesheet)


app = QApplication([])
window = Other()
window.show()
app.exec_()