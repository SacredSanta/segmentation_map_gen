"""
Date : unknown
Who : S.W. Leem

<24.08.20>
deprecated 된 application 으로 추정.
"""

#%% 0.
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QMainWindow, QComboBox, QGridLayout, QLabel, QLineEdit,  QCheckBox, QCalendarWidget
from PyQt5.QtWidgets import QAction, QMenu, qApp, QHBoxLayout, QLCDNumber, QSlider, QVBoxLayout, QFrame, QProgressBar
from PyQt5.QtGui import QColor, QPixmap, QDrag
from PyQt5.QtCore import QCoreApplication, Qt, QObject, pyqtSignal, QBasicTimer, QDate, QMimeData

#%% 1. packages.

class Main(QMainWindow, QWidget):
    def __init__(self):
        super().__init__()  # 상속받아오기 from Qwidget
        self.initUI()

    def initUI(self):
        # 버튼 관련 코드들
        #btn = QPushButton('Start', self)
        #btn.resize(btn.sizeHint())
        #btn.setToolTip('툴팁입니다.<b>안녕하세요.<b/>')
        #btn.move(190, 340)
        #btn.clicked.connect(QCoreApplication.instance().quit)  # 클릭 신호를 받으면 괄호 안에 있는 슬롯을 실행한다!

        okButton = QPushButton("Ok")
        cancelButton = QPushButton("Cancel")

        hbox = QHBoxLayout()  # 가로로 배열되는 레이아웃 박스 (Main window는 안되고, Qwidget을 상속받을때만 가능)
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)

        vbox = QVBoxLayout()  # 세로로 배열되는 레이아웃 박스
        #pixmap = QPixmap("ori_img3.jpeg")
        #lbl = QLabel(self)
        #lbl.setPixmap(pixmap)

        #vbox.addWidget(lbl)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # 윈도우 크기, 제목
        self.move
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Peak Finder')
        self.statusBar()  # 상태표시줄 생성
        self.statusBar().showMessage("안녕하세요")  # 상태표시줄 메세지

        menu = self.menuBar()  # 메뉴바 추가

        # 메뉴 바에 들어가는 항목들
        menu_file = menu.addMenu('File')  # 파일 그룹 추가하기
        menu_edit = menu.addMenu('Edit')  # 에딧 그룹 추가하기
        menu_view = menu.addMenu('View')  # 뷰 그룹 추가하기

        # 메뉴바에 exit을 추가하고, quit하는 동작까지 추가하기
        file_exit = QAction('Exit', self)  # 파일 메뉴 객체 생성
        file_exit.setShortcut('Ctrl+Q')
        file_exit.setStatusTip("누르면 영원히 빠이빠이")  # 여기까지는 메모리에만 객체 만든거고, GUI에는 아직 추가 안되어 있음
        file_exit.triggered.connect(QCoreApplication.instance().quit)

        # 메뉴바에 view를 추가하고, 동작을 추가하기
        view_stat = QAction('상태표시줄', self, checkable=True)  # 체크박스를 추가하는것
        view_stat.setChecked(True)
        view_stat.triggered.connect(self.tglStat)

        # 메뉴바에 여러가지 menu 만들기
        file_new = QMenu('New', self)  # file 안에 new 라는 sub 그룹이 추가되는것이야
        file_open = QMenu('Open', self)
        file_save = QMenu('Save', self)

        # new 에다가 txt를 불러올거냐, py를 불러올거냐를 보여주기
        file_new_txt = QAction("텍스트 파일", self)
        file_new_py = QAction("파이썬 파일", self)
        file_new.addAction(file_new_txt)
        file_new.addAction(file_new_py)

        # file 부분에 new, open, save 동작 추가
        menu_file.addMenu(file_new)
        menu_file.addMenu(file_open)
        menu_file.addMenu(file_save)
        menu_file.addAction(file_exit)  # file exit이라는 action을 추가하는 것
        menu_view.addAction(view_stat)

        self.resize(450,400)
        self.show()


    def tglStat(self, state):
        if state:  # state가 true 라면 status bar를 보여주세요
            self.statusBar().show()
        else:
            self.statusBar().hide()

    def contextMenuEvent(self, QContextMenuEvent):  # 우클릭 하면 메뉴가 나와요
        cm = QMenu(self)

        quit = cm.addAction('Quit')
        action = cm.exec_(self.mapToGlobal(QContextMenuEvent.pos()))
        if action == quit:
            qApp.quit()

        # 전체적인 map에서 어디를 클릭했는지를 저장해서 반환하겠다.
        # 나중에 우클릭 하는 위치가 다를 경우 다른 메뉴가 나오도록

    def closeEvent(self, QCloseEvent):  # 창 닫기를 할 때 옵션을 주는것
        ans = QMessageBox.question(self, "종료 확인", "종료 하시겠습니까?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main()
    sys.exit(app.exec_())
    
    
    
    
