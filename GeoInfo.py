
import math
from math import isnan
import string
import pandas as pd
import numpy as np
from natsort import natsorted
import random

from scipy.stats import t
from sklearn.linear_model import LinearRegression

import openpyxl
import os
from pathlib import Path
import sys
import platform

from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QTableWidget,QVBoxLayout,\
 QLineEdit, QTableView,QWidget, QMessageBox, QItemDelegate

from PyQt5.QtCore import QSortFilterProxyModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem


import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


#from define import FloatDelegate
from sklearn import preprocessing, metrics, mixture
from sklearn.mixture import GaussianMixture
from sklearn.datasets._samples_generator import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from scipy.spatial.distance import cdist



op_sys = platform.system()
if op_sys == 'Darwin':
    from Foundation import NSURL

class PandasModel(QtCore.QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None): 
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()
        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        return super(PandasModel, self).headerData(section, orientation, role)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()
        if not index.isValid():
            return QtCore.QVariant()
        if index.row() == 0:
            return QtCore.QVariant(self._df.columns.values[index.column()])
        return QtCore.QVariant(str(self._df.iloc[index.row()-1, index.column()]))

    def setData(self, index, value, role):
        if index.row() == 0:
            if isinstance(value, QtCore.QVariant):
                value = value.value()
            if hasattr(value, 'toPyObject'):
                value = value.toPyObject()
            self._df.columns.values[index.column()] = value
            self.headerDataChanged.emit(QtCore.Qt.Horizontal, index.column(), index.column())
        else:
            col = self._df.columns[index.column()]
            row = self._df.index[index.row()]
            if isinstance(value, QtCore.QVariant):
                value = value.value()
            if hasattr(value, 'toPyObject'):
                value = value.toPyObject()
            else:
                dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
                self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)+1 

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)


class FloatDelegate(QItemDelegate):
    def __init__(self, parent=None):
        super().__init__()

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setValidator(QDoubleValidator())
        return editor

    
    def updateDF(self, row, column):
        text = self.tableWidget.item(row, column).text()
        self.dataframe.iloc[row, column] = text

class About(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('about.ui',self)
        self.pushButton.clicked.connect(self.close)
        self.setFixedSize(834, 403)

    def display(self):
    	self.show()

class Ui (QMainWindow):
    
    def __init__(self):
        super(Ui, self).__init__()
        self.w = None
        uic.loadUi('Main_1.ui',self)
        self.second = About()
        self.actionOpen_2.triggered.connect(self.open)
        self.actionLoad.triggered.connect(self.read_2)
        self.actionOpen.triggered.connect(self.open)
        self.actionLoad.triggered.connect(self.read_2)
        self.actionOpen_2.triggered.connect(self.progressing)
        self.actionCalculate.triggered.connect(self.read)
        self.actionCalculate.triggered.connect(self.read_1)
        self.actionCalculate.triggered.connect(self.read_3)
        self.actionCalculate.triggered.connect(self.progressing_1)
        self.actionGraph_2.triggered.connect(self.graph)
        self.actionGraph.triggered.connect(self.graph)
        self.actionExit_2.triggered.connect(self.Exit)
        self.actionExit.triggered.connect(self.Exit)
        self.actionHelp.triggered.connect(self.help)
        self.actionExport_2.triggered.connect(self.statistic)
        self.actionExport_3.triggered.connect(self.rejected)
        self.actionHelp_2.triggered.connect(self.help)
        self.pushButton_9.clicked.connect(self.filter)
        self.actionDefine.triggered.connect(self.Define)
        self.radioButton.toggled.connect(self.onclick)
        self.radioButton_1.toggled.connect(self.onclick)
        self.pushButton_7.clicked.connect(self.print_1)
        self.pushButton_8.clicked.connect(self.Define)


        self.actionExport_2.setEnabled(False)
        self.actionExport_3.setEnabled(False)
        self.actionDefine.setEnabled(False)
        self.actionLoad.setEnabled(False)
        self.actionCalculate.setEnabled(False)
        self.actionGraph_2.setEnabled(False)
        self.actionMachine_Learning.setEnabled(False)
        self.actionCalculate_1.setEnabled(False)
        self.actionGraph.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_8.setEnabled(False)
        self.pushButton_9.setEnabled(False)


        self.setAcceptDrops(True)

    class OutlierMatrix:
        outlier_MC = np.array([])
        outlier_Density = np.array([])
        outlier_LL = np.array([])
        outlier_PL = np.array([])
        outlier_Void_Ratio = np.array([])
        outlier_Tau025 = np.array([])
        outlier_Tau050 = np.array([])
        outlier_Tau75 = np.array([])
        outlier_Tau100 = np.array([])
        outlier_Tau150 = np.array([])
        outlier_Tau200 = np.array([])
        outlier_Tau300 = np.array([])


    def help(self):
    	self.second.display()

    def progressing(self):
        from support import Other


    def progressing_1(self):
        from support_1 import Other


    def Exit(self):
        reply = QMessageBox.question(self, 'Quit', 'Are you sure you want to quit?',
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()

    def Define(self):
        self.pushButton_7.setEnabled(True)
        self.pushButton_8.setEnabled(True)
        self.radioButton.setChecked(True)
        self.label_11.setText('Data Definition')
        self.label_9.setText('Physical data')
        self.label_10.setText('Mechanical data')
        data = {
                'Name': ['LAYER','BOREHOLE','SAMPLE','WATER CONTENT','DENSITY (ƴ)','LIQUID LIMIT',
                'PLASTIC LIMIT ','VOID (е₀)'],
                'Index': ['B','C','D','X', 'Y', 'T', 'U', 'AA']
        }

        data_1 = {
                'Name':['TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8'],
                'Name_1': ['AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ'],
                'Index':[25, 50, 75, 100, 150, 200, 300,400]
        }


        dframe = pd.DataFrame(data)
        self.dframe = dframe
        self.dram = dframe

        dataframe = pd.DataFrame(data_1)
        self.dataframe = dataframe
        self.datafram = dataframe

        nRows, nColumns = self.dram.shape
        self.tableWidget_2.setColumnCount(nColumns)
        self.tableWidget_2.setRowCount(nRows)
        self.tableWidget_2.setHorizontalHeaderLabels(('Name','Column'))
        self.tableWidget_2.setItemDelegateForColumn(0, FloatDelegate())
        for i in range(self.tableWidget_2.rowCount()):
            for j in range(self.tableWidget_2.columnCount()):
                self.tableWidget_2.setItem(i, j, QTableWidgetItem(self.dram.iloc[i,j]))
        self.tableWidget_2.cellChanged[int, int].connect(self.update)
        stylesheet = "::section{Background-color:rgb(192,192,192);}"
        self.tableWidget_2.verticalHeader().setStyleSheet(stylesheet)
        self.tableWidget_2.horizontalHeader().setStyleSheet(stylesheet)

        
        nRows_1, nColumns_1 = self.datafram.shape
        self.tableWidget_3.setColumnCount(nColumns_1)
        self.tableWidget_3.setRowCount(nRows_1)
        self.tableWidget_3.setHorizontalHeaderLabels(('Shear stress', 'Column', 'Normal stress (kPa)'))
        self.tableWidget_3.setItemDelegateForColumn(0, FloatDelegate())
        for i_1 in range(self.tableWidget_3.rowCount()):
            for j_1 in range(self.tableWidget_3.columnCount()):
                self.tableWidget_3.setItem(i_1, j_1, QTableWidgetItem(str(self.datafram.iloc[i_1,j_1])))
        self.tableWidget_3.cellChanged[int, int].connect(self.update_1)
        stylesheet = "::section{Background-color:rgb(192,192,192);}"
        self.tableWidget_3.verticalHeader().setStyleSheet(stylesheet)
        self.tableWidget_3.horizontalHeader().setStyleSheet(stylesheet)
    
    def update(self, row, column):
        text = self.tableWidget_2.item(row, column).text()
        self.dframe.iloc[row, column] = text

    def update_1(self, row, column):
        text = self.tableWidget_3.item(row, column).text()
        self.dataframe.iloc[row, column] = text

    def print_1 (self):
        self.actionLoad.setEnabled(True)
        self.dframe['Sys'] = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL', 'VOID']
        self.dframe_1 = self.dframe.replace(r'^\s*$',np.nan, regex = True)
        dframe_2 = self.dframe_1.dropna(inplace=True)

        print(self.dframe_1)

        self.dataframe_2 = self.dataframe
        self.dataframe_3 = self.dataframe_2.replace(r'^\s*$',np.nan, regex = True)
        dataframe_4 = self.dataframe_3.dropna(inplace=True)
        self.dataframe_1 = self.dataframe.replace(r'^\s*$',np.nan, regex = True)

        key = []
        values = []
        for i in range(len(self.dataframe_1)):
            key.append('SIG' + str(i))
            values.append(float(self.dataframe_1['Index'][i]))
            
        dic = dict(zip(key, values))

    
        clean_dict = {k: dic[k] for k in dic if not isnan(dic[k])}
        supkey = []
        suvalues = list(clean_dict.values())
        self.TAU = []
        for j in range(len(clean_dict)):
            supkey.append('SIGMA' + str(j+1))
            self.TAU.append('TAU' + str(j+1))

        self.dataframe_3['TAU'] = self.TAU


        self.dict_last = dict(zip(supkey, suvalues))
        self.total = self.dframe_1['Sys'].values.tolist() + self.TAU
        self.total_1 = self.dframe_1['Sys'].values.tolist()[3:] + self.TAU


    def onclick(self):
        if self.radioButton.isChecked():
            self.target = 'CONSTRUCTION'

        elif self.radioButton_1.isChecked():
            self.target = 'BRIDGE'
        
    def open(self):
        self.fname = QFileDialog.getOpenFileName(self, caption = 'Open file', directory=".", filter="All Files (*.*)")[0]
        if self.fname !=(''):
            self.actionDefine.setEnabled(True)

        else:
            print('DataFrame not selected')
            QMessageBox.about(self, 'Error', 'YOU HAVE NOT SELECTED FILE')


    def dragEnterEvent(self,e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
    def dragMoveEvent(self,e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
    def dropEvent(self,e):
        if e.mimeData().hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()
            for url in e.mimeData().urls():
                if op_sys == 'Darwin':
                    fname = str(NSURL.URLWithString_(str(url.toSTring())).filePathURL().path())
                else:
                    fname = str(url.toLocalFile())
            self.fname = fname
            print(self.fname)
            self.read_2()
        else:
            e.ignore()

    def read_2(self):
        self.actionDefine.setEnabled(True)
        self.df = pd.read_excel(self.fname)
        self.actionCalculate.setEnabled(True)
        self.label_2.setText('Raw Data')
        self.data =self.df
        self.data = self.df.sort_index()
        self.model_1 = QStandardItemModel(len(self.data.axes[1]), len(self.data.axes[0]))
        self.model_1.setHorizontalHeaderLabels(self.data.columns)
        self.data = self.data.sort_index()
        self.model_1 = PandasModel(self.data)
        self.tableView.setModel(self.model_1)
        for i in range(self.model_1.columnCount()):
            ix = self.model_1.index(-1, i)
            self.tableView.openPersistentEditor(ix)

        stylesheet = "::section{Background-color:rgb(192,192,192);}"
        self.tableView.verticalHeader().setStyleSheet(stylesheet)
        self.tableView.horizontalHeader().setStyleSheet(stylesheet)

    def read(self):
        test = self.total

        def Initialize(prop_elm):

            #full physical value & edit mechanical value#
            train = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 
            'TAU3', 'TAU4']
            train_1 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 
            'TAU3', 'TAU4', 'TAU5']
            train_2 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 
            'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_3 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 
            'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_4 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 
            'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            #physical value option#
            train_5 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL', 'VOID']
            train_6 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL']
            train_7 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'VOID']
            train_8 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'VOID']
            train_9 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL','PL', 'VOID']
            train_10 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'VOID']
            train_11 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'LL']
            train_12 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL']
            train_13 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'VOID']
            train_14 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'PL']
            train_15 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'VOID']
            train_16 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'VOID']
            train_17 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'PL', 'VOID']
            train_18 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'VOID']
            train_19 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL']
            train_20 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'VOID']
            train_21 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN']
            train_22 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL']
            train_23 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'VOID']
            train_24 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'VOID']
            train_25 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL']
            train_26 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'VOID']
            train_27 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL']
            train_28 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'PL']
            train_29 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'VOID']
            train_30 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL']
            train_31 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC']
            train_32 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN']
            train_33 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL']
            train_34 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL']
            train_35 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'VOID']

            #mechanical values

            train_36 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'TAU1', 'TAU2', 'TAU3']
            train_37 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_38 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_39 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_40 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_41 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            #Editing physical value & mechanical value #

            train_42 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL','TAU1', 'TAU2', 'TAU3']
            train_43 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL','TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_44 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL','TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_45 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL','TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_46 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL','TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_47 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'PL','TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_48 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_49 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4']
            train_50 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5']
            train_51 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5', 'TAU6']
            train_52 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5', 'TAU6','TAU7']
            train_53 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5', 'TAU6','TAU7','TAU8']

            train_54 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_55 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4']
            train_56 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5']
            train_57 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5', 'TAU6']
            train_58 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5', 'TAU6','TAU7']
            train_59 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5', 'TAU6','TAU7', 'TAU8']

            train_60 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL','PL', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_61 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL','PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4']
            train_62 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL','PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5']
            train_63 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL','PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5', 'TAU6']
            train_64 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL','PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5', 'TAU6', 'TAU7']
            train_65 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL','PL', 'VOID', 'TAU1', 'TAU2', 'TAU3','TAU4','TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_66 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'VOID','TAU1', 'TAU2', 'TAU3']
            train_67 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'VOID','TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_68 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'VOID','TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_69 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'VOID','TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_70 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'VOID','TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_71 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'VOID','TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_72 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'LL', 'TAU1', 'TAU2', 'TAU3']
            train_73 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_74 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_75 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_76 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_77 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_78 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'TAU1', 'TAU2', 'TAU3']
            train_79 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_80 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_81 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_82 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_83 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_84 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_85 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_86 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_87 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_88 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_89 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC','DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']
            
            train_90 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3']
            train_91 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'PL' ,'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_92 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_93 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_94 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'PL' ,'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_95 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']
            
            train_96 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_97 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_98 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_99 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_100 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_101 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_102 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_103 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_104 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_105 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_106 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_107 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']
            
            train_108 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_109 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_110 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_111 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_112 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_113 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_114 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'VOID',  'TAU1', 'TAU2', 'TAU3']
            train_115 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'VOID',  'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_0116 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'VOID',  'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_116 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'VOID',  'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_117 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'VOID',  'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_118 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'VOID',  'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_119 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3']
            train_120 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_121 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_122 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_123 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_124 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_125 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_126 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_127 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_128 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_129 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_130 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_131 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'TAU1', 'TAU2', 'TAU3']
            train_132 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_133 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_134 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_135 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_136 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_137 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'TAU1', 'TAU2', 'TAU3']
            train_138 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_139 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_140 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_141 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_142 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_143 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_144 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_145 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_146 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_147 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_148 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_149 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_150 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_151 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_152 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_153 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_154 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_155 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3']
            train_156 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_157 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_158 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_159 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_160 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_161 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_162 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_163 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_164 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_165 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_166 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']
            
            train_167 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'TAU1', 'TAU2', 'TAU3']
            train_168 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_169 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_170 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_171 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_172 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_173 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_174 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_175 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_176 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_177 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_178 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_179 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'TAU1', 'TAU2', 'TAU3']
            train_180 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_181 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_182 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_183 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_184 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_185 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'TAU1', 'TAU2', 'TAU3']
            train_186 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_187 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_188 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_189 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_190 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'MC', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_191 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'TAU1', 'TAU2', 'TAU3']
            train_192 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_193 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_194 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_195 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_196 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'DEN', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_197 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'TAU1', 'TAU2', 'TAU3']
            train_198 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_199 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_200 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_201 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_202 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'LL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_203 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'TAU1', 'TAU2', 'TAU3']
            train_204 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_205 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_206 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_207 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_208 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'PL', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']

            train_209 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'VOID', 'TAU1', 'TAU2', 'TAU3']
            train_210 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4']
            train_211 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5']
            train_212 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6']
            train_213 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7']
            train_214 = ['LAYER', 'BOREHOLE', 'SAMPLE', 'VOID', 'TAU1', 'TAU2', 'TAU3', 'TAU4', 'TAU5', 'TAU6', 'TAU7', 'TAU8']
            


            #full physical value & edit mechanical value#
            if set(test) == set(train):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test) == set(train_1):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test) == set(train_2):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test) == set(train_3):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_4):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            #physical value option#
            elif set(test ) == set(train_5):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_6):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]

            elif set(test ) == set(train_7):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_8):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_9):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_10):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_11):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]

            elif set(test ) == set(train_12):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]

            elif set(test ) == set(train_13):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_14):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]

            elif set(test ) == set(train_15):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_16):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_17):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_18):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_19):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]

            elif set(test ) == set(train_20):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_21):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]

            elif set(test ) == set(train_22):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]

            elif set(test ) == set(train_23):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_24):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_25):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]

            elif set(test ) == set(train_26):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_27):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]

            elif set(test ) == set(train_28):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]

            elif set(test ) == set(train_29):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            elif set(test ) == set(train_30):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]

            elif set(test ) == set(train_31):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]

            elif set(test ) == set(train_32):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]

            elif set(test ) == set(train_33):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]

            elif set(test ) == set(train_34):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]

            elif set(test ) == set(train_35):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]

            #mechanical values

            elif set(test ) == set(train_36):

                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
            
            elif set(test ) == set(train_37):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_38):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_39):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_40):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_41):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]


            #Editing physical value & mechanical value #

            elif set(test ) == set(train_42):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_43):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_44):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_45):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_46):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test) == set(train_47):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]






            elif set(test ) == set(train_48):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_49):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_50):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_51):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_52):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_53):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_54):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_55):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_56):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_57):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_58):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_59):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_60):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_61):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_62):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_63):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_64):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_65):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_66):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_67):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_68):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_69):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_70):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_71):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_72):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_73):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_74):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_75):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_76):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_77):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_78):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_79):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_80):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_81):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_82):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_83):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_84):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_85):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_86):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_87):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_88):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_89):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_90):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_91):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_92):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_93):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_94):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_95):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_96):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_97):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_98):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_99):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_100):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_101):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_102):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_103):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_104):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_105):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_106):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_107):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_108):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_109):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_110):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_111):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_112):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_113):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_114):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_115):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_0116):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_116):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_117):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_118):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_119):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_120):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_121):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_122):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_123):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_124):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_125):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_126):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_127):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_128):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_129):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_130):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_131):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_132):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_133):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_134):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_135):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_136):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_137):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_138):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_139):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_140):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_141):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_142):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_143):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_144):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_145):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_146):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_147):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_148):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_149):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_150):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_151):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_152):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_153):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_154):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_155):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_156):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_157):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_158):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_159):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_160):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_161):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_162):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_163):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_164):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_165):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_166):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_167):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_168):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_169):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_170):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_171):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_172):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_173):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_174):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_175):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_176):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_177):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_178):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]
            
            elif set(test ) == set(train_179):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_180):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_181):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_182):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_183):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_184):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_185):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_186):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_187):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_188):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                
            elif set(test ) == set(train_189):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_190):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['MC'] = [self.dframe_1['Index'][3], 0, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_191):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_192):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_193):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_194):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_195):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_196):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['DEN'] = [self.dframe_1['Index'][4], 1, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_197):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_198):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_199):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_200):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_201):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_202):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['LL'] = [self.dframe_1['Index'][5], 2, 1,1]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_203):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_204):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_205):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_206):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_207):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_208):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['PL'] = [self.dframe_1['Index'][6], 3, 1,2]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]

            elif set(test ) == set(train_209):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]

            elif set(test ) == set(train_210):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]

            elif set(test ) == set(train_211):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]

            elif set(test ) == set(train_212):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]

            elif set(test ) == set(train_213):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]

            elif set(test ) == set(train_214):
                prop_elm['LAYER'] = [self.dframe_1['Index'][0], -1, 0,-1]
                prop_elm['BOREHOLE'] = [self.dframe_1['Index'][1], -1, 0,-1]
                prop_elm['SAMPLE'] = [self.dframe_1['Index'][2], -1, 0,-1]
                prop_elm['VOID'] = [self.dframe_1['Index'][7], 4, 1,3]
                prop_elm['TAU1'] = [self.dataframe_3['Name_1'][0], 5, 2,1]
                prop_elm['TAU2'] = [self.dataframe_3['Name_1'][1], 6, 2,1]
                prop_elm['TAU3'] = [self.dataframe_3['Name_1'][2], 7, 2,1]
                prop_elm['TAU4'] = [self.dataframe_3['Name_1'][3], 8, 2,1]
                prop_elm['TAU5'] = [self.dataframe_3['Name_1'][4], 9, 2,1]
                prop_elm['TAU6'] = [self.dataframe_3['Name_1'][5], 10, 2,1]
                prop_elm['TAU7'] = [self.dataframe_3['Name_1'][6], 11, 2,1]
                prop_elm['TAU8'] = [self.dataframe_3['Name_1'][7], 12, 2,1]


        def ColumnNumber(col_name):
            
            ref_alphabet = "_" + string.ascii_uppercase
            if len(col_name) == 1:
                col_name = '_' + col_name.upper()
            else:
                col_name = col_name.upper()

            col_num = 26 * \
                ref_alphabet.index(col_name[0])+ref_alphabet.index(col_name[1])
            return col_num

        def ColumnName(col_num):
            ref_alphabet = string.ascii_uppercase
            if col_num <= 26:
                return ref_alphabet[(col_num - 1) %26]
            else:
                return ref_alphabet[(col_num - 1) // 26] + ref_alphabet[(col_num - 1) %26]

        
        def Grubbs_test(n, alpha):
            # t.ppf function calculates the t-value at the siginificant level and the degree of freedom
            if n > 2:
                t_inv = abs(t.ppf((1.0-alpha)/2/n, n-2))
                sqrt_val = math.sqrt((n - 1) / (n - 2 + t_inv ** 2))
                return t_inv * sqrt_val
            else:
                return None

        def DecAngle_to_DMSAngle(decimal_degree):
            int_deg = round(int(decimal_degree),3)
            minute = round(int(60*(decimal_degree - int_deg)),3)
            return f'{str(int_deg)}°{str(minute):2s}´'


        def limit_state_calculation(layer_name, clean_df, design_df, conf_level):


            basics_stats = {'Count':np.nan, 'Mean':np.nan, 'StDev':np.nan, 'Min':np.nan,
                'Max': np.nan, 'Xtc': np.nan, 'Xtt(I)': np.nan, 'Xtt(II)': np.nan}
        
            conf_level1 = conf_level[0]
            conf_level2 = conf_level[1]

            shear_arr = np.array([], dtype ='float')
            sigma_arr = np.array([], dtype='float')

            for prop in prop_names:
                dec_digits=geo_props[prop][3]
                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Count'), prop] = round(clean_df.loc[(clean_df['LAYER'] == layer_name), prop].count(),0)
                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Mean'), prop] = round(clean_df.loc[(clean_df['LAYER'] == layer_name), prop].mean(),dec_digits)
                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'StDev'), prop] = round(clean_df.loc[(clean_df['LAYER'] == layer_name), prop].std(),2)
                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Min'), prop] = round(clean_df.loc[(clean_df['LAYER'] == layer_name), prop].min(),dec_digits)
                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Max'), prop] = round(clean_df.loc[(clean_df['LAYER'] == layer_name), prop].max(),dec_digits)

                if 'TAU' not in prop:
            # ----------------------------------#
            # Step 3: Calculate standard value #
            #----------------------------------#
                    Xtc = design_df.loc[(design_df['LAYER'] == layer_name) & (
                        design_df['STAT'] == 'Mean'), prop].item()

                    Xmax = design_df.loc[(design_df['LAYER'] == layer_name) & (
                        design_df['STAT'] == 'Max'), prop].item()

                    Xmin = design_df.loc[(design_df['LAYER'] == layer_name) & (
                        design_df['STAT'] == 'Min'), prop].item()

                    StDev = design_df.loc[(design_df['LAYER'] == layer_name) & (
                        design_df['STAT'] == 'StDev'), prop].item()
            #------------------------------------------------------------------------------#
            # Step 4: Calculate limit value for limit state I (bearing capacity condition) #
            #         Calculate limit value for limit state II (deformation condition)     #
            #------------------------------------------------------------------------------#
                    n = design_df.loc[(design_df['LAYER'] == layer_name) & (
                        design_df['STAT'] == 'Count'), prop].item()

                    plus_list = ['MC', 'LL', 'PL', 'VOID']
                    if n < 6:
                        if prop in plus_list:
                            Xtt_I = 0.5 * (Xtc + Xmax)
                        else:
                            Xtt_I = 0.5 * (Xtc - Xmin)

                        Xtt_II = Xtt_I
                    else:

                        cal_CoV = StDev/Xtc

                # limit state I
                        t_inv = abs(t.ppf(1.0 - conf_level1, n - 1))
                        rho_alpha = t_inv * cal_CoV / (n**0.5)
                        if prop in plus_list:
                            Xtt_I = Xtc * (1 + rho_alpha)
                        else:
                            Xtt_I = Xtc * (1 - rho_alpha)

                # limit state II
                        t_inv = abs(t.ppf(1.0 - conf_level2, n - 1))
                        rho_alpha = t_inv * cal_CoV / (n**0.5)
                        if prop in plus_list:
                            Xtt_II = Xtc * (1 + rho_alpha)
                        else:
                            Xtt_II = Xtc * (1 - rho_alpha)

            #-----------------------------------#
            # Step 5: Export statistical values #
            #-----------------------------------#
                    dec_digits=geo_props[prop][3]
                    design_df.loc[(design_df['LAYER'] == layer_name) & (
                        design_df['STAT'] == 'Xtc'), prop] = round(Xtc,dec_digits)
                    design_df.loc[(design_df['LAYER'] == layer_name) & (
                        design_df['STAT'] == 'Xtt(I)'), prop] = round(Xtt_I,dec_digits)
                    design_df.loc[(design_df['LAYER'] == layer_name) & (
                        design_df['STAT'] == 'Xtt(II)'), prop] = round(Xtt_II,dec_digits)
                else:
            # Create array X and array Y
                    n_obs = design_df.loc[(design_df['LAYER'] == layer_name) & (
                        design_df['STAT'] == 'Count'), prop].item()
                    if n_obs > 0:
                        shear_arr = np.append(
                            shear_arr, clean_df.loc[clean_df['LAYER'] == layer_name, prop].dropna())
                        sigma_arr = np.append(
                            sigma_arr, np.repeat(list(norm_pressure.values())[shear_prop_names.index(prop)], n_obs))

            if len(sigma_arr) > 0:
        # Calculate the friction angle, cohesion
                x_true = sigma_arr.reshape(-1, 1)
                y_true = shear_arr.reshape(-1, 1)
                lreg = LinearRegression()
                lreg.fit(x_true, y_true)
                y_pred = lreg.predict(x_true)
                x_mean = np.mean(sigma_arr)
                x_count = len(sigma_arr)

                tan_phi = lreg.coef_[0][0]
                phi = math.degrees(math.atan(tan_phi))
                cohesion = lreg.intercept_[0]

                ssr = sum((y_pred - np.mean(y_true))**2)[0]
                sse = sum((y_true - y_pred) ** 2)[0]  # sum of squared residuals
                sst = ssr + sse
                R_square = ssr / sst
                Se = np.sqrt(sse / (x_count - 2))  # standard error of estimation
                SSxx = sum(sigma_arr ** 2) - x_count * x_mean ** 2

                stdev_cohesion = Se * ((1 / x_count + x_mean ** 2 / SSxx) ** 0.5)
                stdev_tan_phi = Se / (SSxx ** 0.5)

                V_cohesion = stdev_cohesion / cohesion
                V_tan_phi = stdev_tan_phi / tan_phi

                t_inv_I = abs(t.ppf(1.0 - conf_level1, x_count - 2))
                t_inv_II = abs(t.ppf(1.0 - conf_level2, x_count - 2))

                cohesion_I = cohesion*(1 - t_inv_I * V_cohesion)
                cohesion_II = cohesion*(1 - t_inv_II * V_cohesion)

                tan_phi_I = tan_phi*(1 - t_inv_I * V_tan_phi)
                tan_phi_II = tan_phi*(1 - t_inv_II * V_tan_phi)

                phi_I = math.degrees(math.atan(tan_phi_I))
                phi_II = math.degrees(math.atan(tan_phi_II))

                dms_phi = DecAngle_to_DMSAngle(phi)
                dms_phi_I = DecAngle_to_DMSAngle(phi_I)
                dms_phi_II = DecAngle_to_DMSAngle(phi_II)

        # Output statistics of regression analysis


                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Xtc'), 'FRICTION_ANGLE'] = dms_phi
                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Xtt(I)'), 'FRICTION_ANGLE'] = dms_phi_I
                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Xtt(II)'), 'FRICTION_ANGLE'] = dms_phi_II

                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Xtc'), 'COHESION'] = round(cohesion, 3)
                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Xtt(I)'), 'COHESION'] = round(cohesion_I, 3)
                design_df.loc[(design_df['LAYER'] == layer_name) & (
                    design_df['STAT'] == 'Xtt(II)'), 'COHESION'] = round(cohesion_II, 3)
# ---End of limi_state_calculation---


        def remove_outlier(layer_name, input_df, output_df):
            

            outlier_df = pd.DataFrame(
                columns=('LAYER', 'PROPERTY', 'VALUE', 'ERROR_TYPE'))
            for prop in prop_names:
                in_col_idx = ColumnNumber(geo_props[prop][0]) -1
                prop_df = input_df.iloc[:, in_col_idx]
                #print(prop_df)

        #---------------------------------------------#
        # Step 1: Checking outlier according to Vstat #
        #---------------------------------------------#
                passed = False
                while not passed:
                    if prop_df.count() > 2:
                        check_value = prop_df.max()
                        if (check_value - prop_df.mean()) < (prop_df.mean() - prop_df.min()):
                            check_value = prop_df.min()

                        Vstat = Grubbs_test(prop_df.count(), 0.95)
                        StDevP = prop_df.std() * \
                            math.sqrt((prop_df.count()-1)/prop_df.count())

                        if abs(check_value - prop_df.mean()) > (Vstat * StDevP):
                            rejected_row = prop_df[prop_df == check_value]
                            outlier_df = outlier_df.append(
                                {'LAYER': layer_name, 'PROPERTY': prop, 'VALUE': check_value, 'ERROR_TYPE': 1}, ignore_index=True)

                    # remove rejected value
                            prop_df = prop_df.drop(rejected_row.index[0])
                            output_df.loc[output_df[prop] ==
                                        check_value, prop] = np.nan
                        else:
                            passed = True

                    else:
                        passed = True

        #----------------------------------------------------------------#
        # Step 2: Checking validity after Coefficient of variation (CoV) #
        #         Physical properties: maximum of CoV = 0.15             #
        #         Mechanical properties: maximum of CoV = 0.30           #
        #----------------------------------------------------------------#
                passed = False
                if geo_props[prop][2] == 1:
                    max_CoV = 0.15
                else:
                    max_CoV = 0.30

                while not passed:
                    cal_CoV = prop_df.std() / prop_df.mean()

                    if cal_CoV > max_CoV:
                        check_value = prop_df.max()
                        if (check_value - prop_df.mean()) < (prop_df.mean() - prop_df.min()):
                            check_value = prop_df.min()

                        rejected_row = prop_df[prop_df == check_value]
                        outlier_df = outlier_df.append(
                            {'LAYER': layer_name, 'PROPERTY': prop, 'VALUE': check_value, 'ERROR_TYPE': 2}, ignore_index=True)

                # remove rejected value
                        prop_df = prop_df.drop(rejected_row.index[0])
                        output_df.loc[output_df[prop] == check_value, prop] = np.nan
                    else:
                        passed = True

            return outlier_df
# ---End of remove_outlier---

        conf_coef = {'CONSTRUCTION': (0.95, 0.85), 'BRIDGE': (0.98, 0.90)}
        norm_pressure = self.dict_last
        prop_names = tuple(self.total_1)
        shear_prop_names = tuple(self.TAU)
        stat_names = ('Count', 'Mean', 'StDev', 'Min',
              'Max', 'Xtc', 'Xtt(I)', 'Xtt(II)')

        geo_props = dict()
        Initialize(geo_props)
        col_idxs = []
        for _prop in geo_props:
            col_idxs.append(ColumnNumber(geo_props[_prop][0])-1)

        self.data =self.df
        df = self.data
        self.clean_df = df.iloc[:, col_idxs]
        self.clean_df.columns = geo_props.keys()

        self.stat_df = pd.DataFrame(
            columns=('LAYER', 'STAT') + prop_names)


        self.rejected_df = pd.DataFrame(
            columns=('LAYER', 'PROPERTY', 'VALUE', 'ERROR_TYPE'))

        col_idx = ColumnNumber(geo_props['LAYER'][0]) - 1

        layer_names = natsorted(df.iloc[:,col_idx].unique())

        for layer in layer_names:
            dirty_df = df[df.iloc[:, col_idx] == layer]
            

            for stat in stat_names:
                self.stat_df.loc[len(self.stat_df.index)] = {
                    'LAYER':layer, 'STAT': stat}

            #Cleaning dataframe
            self.rejected_df = self.rejected_df.append(remove_outlier(layer, dirty_df, self.clean_df))
            limit_state_calculation(layer, self.clean_df, self.stat_df,
                            conf_coef[self.target])
     
    
    def read_1(self):
        self.label_3.setText('Statistics Values')
        self.model_2 = QStandardItemModel(len(self.stat_df.axes[1]), len(self.stat_df.axes[0]))
        self.model_2.setHorizontalHeaderLabels(self.stat_df.columns)
        self.stat_df = self.stat_df.sort_index()
        self.model_2 = PandasModel(self.stat_df)
        self.tableView_2.setModel(self.model_2)
        for i in range(self.model_2.columnCount()):
            ix = self.model_2.index(-1, i)
            self.tableView_2.openPersistentEditor(ix)

        stylesheet = "::section{Background-color:rgb(192,192,192);}"
        self.tableView_2.verticalHeader().setStyleSheet(stylesheet)
        self.tableView_2.horizontalHeader().setStyleSheet(stylesheet)


    def read_3(self):
        self.actionGraph_2.setEnabled(True)
        self.label_4.setText('Rejected Values')
        self.model_3 = QStandardItemModel(len(self.rejected_df.axes[1]), len(self.rejected_df.axes[0]))
        self.model_3.setHorizontalHeaderLabels(self.rejected_df.columns)
        self.rejected_df = self.rejected_df.sort_index()
        self.model_3 = PandasModel(self.rejected_df)
        self.tableView_3.setModel(self.model_3)
        for i in range(self.model_3.columnCount()):
            ix = self.model_3.index(-1, i)
            self.tableView_3.openPersistentEditor(ix)

        stylesheet = "::section{Background-color:rgb(192,192,192);}"
        self.tableView_3.verticalHeader().setStyleSheet(stylesheet)
        self.tableView_3.horizontalHeader().setStyleSheet(stylesheet)
        print(self.clean_df)


    def graph(self):
        self.df = self.clean_df
        properties = self.dframe_1['Sys'].values.tolist()[3:]
        li =  self.df['LAYER'].unique()
        self.groupby_1 =[]
        for i in range(len(li)):
            self.groupby_1.append(str(li[i]))
        self.groupby_1 = sorted(self.groupby_1)

        ds = self.df['LAYER']

        self.ds_1 = []
        for j in range(len(ds)):
            self.ds_1.append(str(ds[j]))

        self.comboBox.setEditable(True)
        self.comboBox.addItems(self.groupby_1)
        self.comboBox.activated[str].connect(self.connect)
        self.pushButton_3.clicked.connect(self.display_2)
        

        self.pushButton_2.clicked.connect(self.display_2)
        self.pushButton_2.clicked.connect(self.display_1)
        self.pushButton_2.clicked.connect(self.display_3)

        self.pushButton.clicked.connect(self.display_1)
        self.pushButton_4.clicked.connect(self.display_3)

        self.comboBox_1.addItems(properties)
        self.comboBox_1.activated[str].connect(self.viusal)
        self.comboBox_2.activated[str].connect(self.style)


        self.figure = plt.figure(figsize=(2,3), dpi=110)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.verticalLayout_1.addWidget(self.canvas)
        

        self.figure_1 = plt.figure(figsize=(3, 4), dpi=110)
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.toolbar_1 = NavigationToolbar(self.canvas_1, self)
        self.verticalLayout.addWidget(self.canvas_1)
        

    def connect(self,text):
        self.df['NewLayer'] = self.ds_1
        self.fn = self.df[self.df['NewLayer'] == text]

    def viusal(self, text):
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.pushButton_4.setEnabled(True)
        self.pushButton_9.setEnabled(True)
        print(self.fn[text])
        self.data = self.fn[text]
        if text == 'LL':
            self.label.setText('LIQUID LIMIT')
        elif text == 'PL':
            self.label.setText('PLASTIC LIMIT')
        elif text == 'DEN':
            self.label.setText('DENSITY')
        elif text == 'VOID':
            self.label.setText('VOID RATIO')
        elif text == 'MC':
            self.label.setText('WATER CONTENT')


    def style(self,text):
        sns.set_style(text)

    def display_2(self):
        print(self.lineEdit_2.text())
        self.verticalLayout_1.addWidget(self.toolbar)
        binz = self.lineEdit_2.text()
        his = self.data
        his = [incom for incom in his if str(incom) !='nan']
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.hist(his, bins= int(binz))
        self.ax.set_xlabel('Data')
        self.ax.set_ylabel('Frequency')
        self.figure.tight_layout()
        self.canvas.draw()

    def display_1(self):
        box = self.data
        box = [incom for incom in box if str(incom) !='nan']
        self.figure_1.clear()
        self.verticalLayout.addWidget(self.toolbar_1)
        self.ax = self.figure_1.add_subplot(111)
        self.ax.boxplot(box)
        self.ax.set_xlabel('Data')
        self.figure_1.tight_layout()
        self.canvas_1.draw()

    def display_3(self):
        self.actionExport_2.setEnabled(True)
        self.actionExport_3.setEnabled(True)
        self.actionMachine_Learning.setEnabled(True)
        des = round(pd.DataFrame(self.data.describe()),3)
        desc =  ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        des_1 = des.insert(0,'Name',desc)
        nRows, nColumns =des.shape
        self.tableWidget.setColumnCount(nColumns)
        self.tableWidget.setRowCount(nRows)
        self.tableWidget.setHorizontalHeaderLabels(('Statistics', 'Values'))
        for i in range(self.tableWidget.rowCount()):
            for j in range(self.tableWidget.columnCount()):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(des.iloc[i, j])))

        stylesheet = "::section{Background-color:rgb(192,192,192);}"
        self.tableWidget.horizontalHeader().setStyleSheet(stylesheet)

    def filter(self):
        filter_proxy_model = QSortFilterProxyModel()
        filter_proxy_model.setSourceModel(self.model_1)
        filter_proxy_model.setFilterKeyColumn(1)
        self.lineEdit_6.textChanged.connect(filter_proxy_model.setFilterRegExp)
        self.tableView.setModel(filter_proxy_model)
        filter_proxy_model = QSortFilterProxyModel()
        filter_proxy_model.setSourceModel(self.model_2)
        filter_proxy_model.setFilterKeyColumn(0)
        self.lineEdit_6.textChanged.connect(filter_proxy_model.setFilterRegExp)
        self.tableView_2.setModel(filter_proxy_model)
        filter_proxy_model = QSortFilterProxyModel()
        filter_proxy_model.setSourceModel(self.model_3)
        filter_proxy_model.setFilterKeyColumn(0)
        self.lineEdit_6.textChanged.connect(filter_proxy_model.setFilterRegExp)
        self.tableView_3.setModel(filter_proxy_model)

    def closeEvent(self, event):
    	reply = QMessageBox.question(self, 'Quit', 'Are you sure you want to quit?',
    		QMessageBox.Yes| QMessageBox.No, QMessageBox.No)
    	if reply == QMessageBox.Yes:
    		event.accept()
    	else:
    		event.ignore()
    def statistic(self):
        file_filter = 'Data File (*.xlsx *.csv *.dat);; Excel File (*.xlsx *.xls)'
        response = QFileDialog.getSaveFileName(
            parent=self,
            caption='Select a data file',
            directory= 'Statistics Table',
            filter=file_filter,
            initialFilter='Excel File (*.xlsx *.xls)'
        )
        if response[0] != (''):
            self.stat_df.to_excel(response[0])
        else:
            QMessageBox.about(self, 'Error','Error')

    def rejected(self):
        file_filter = 'Data File (*.xlsx *.csv *.dat);; Excel File (*.xlsx *.xls)'
        response = QFileDialog.getSaveFileName(
            parent=self,
            caption='Select a data file',
            directory= 'Rejected Table',
            filter=file_filter,
            initialFilter='Excel File (*.xlsx *.xls)'
        )
        if response[0] != (''):
            self.rejected_df.to_excel(response[0])
        else:
            QMessageBox.about(self, 'Error','Error')

app = QApplication(sys.argv)
w = Ui()
w.show()
app.exec_()
