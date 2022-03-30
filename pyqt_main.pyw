import traceback
import numpy as np
import rightmove_webscraper as rmw
from pathlib import Path
import sys
import rm_lib
import pandas as pd
import time
import playsound

from PyQt5.QtWidgets import (QPushButton, QApplication, QGridLayout, QWidget, QDesktopWidget, QMainWindow,
                             QLineEdit, QTextEdit, QLabel, QProgressBar, QTextBrowser, QGroupBox)
from PyQt5.QtCore import QThread, QObject, pyqtSignal


# tesseract path can be taken from PyQt
# url can be taken from PyQt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.layout = QGridLayout()
        self.window = QWidget()
        self.text_log_widget = QGroupBox()
        self.text_log = QTextBrowser(self.text_log_widget)

        self.widgets = {"button_confirm": QPushButton("Confirm"),
                        "progress_bar": QProgressBar(),
                        "textbox_url": QLineEdit(),
                        "text_url": QLabel("URL: "),
                        "text_log": self.text_log}

        self.widget_pos = {"text_url": (0, 0),
                           "textbox_url": (0, 1, 1, 3),
                           "button_confirm": (0, 4),
                           "text_log": (1, 0, 2, 4),
                           "progress_bar": (3, 0, 1, 4)}

        self.thread, self.worker = None, None
        self.url = None
        self.log = str()

        self.init_ui()

    def init_ui(self):
        # Connect signal to our methods.
        self.widgets["button_confirm"].clicked.connect(self.button_confirm_pressed)

        self.window.setWindowTitle('Rightmove HouseTrawler v2.0.0')
        qtRectangle = QWidget.frameGeometry(self)  # this code is to attempt to centre the window.
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        window_position = (qtRectangle.topLeft().x(), qtRectangle.topLeft().y())  # up to here.
        window_size = (600, 300)

        self.window.setGeometry(*window_position, *window_size)

        self.widgets["text_log"].setLineWrapMode(QTextEdit.NoWrap)

        for widget_name in self.widgets:
            self.layout.addWidget(self.widgets[widget_name], *self.widget_pos[widget_name])

        self.window.setLayout(self.layout)
        self.window.show()

    def log_print(self, text):
        self.log += str(text) + "\n"
        self.widgets["text_log"].setText(self.log)

    def progress_update(self, percentage: int):
        self.widgets["progress_bar"].setValue(percentage)

    def button_confirm_pressed(self):
        self.log = str()  # clearing log
        self.widgets["text_log"].setText(self.log)

        self.url = str(self.widgets["textbox_url"].text())  # reading value from textbox
        self.run_worker()

    def run_worker(self):
        self.thread = QThread()
        self.worker = Worker(self.url)

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.progress_update.connect(self.progress_update)
        self.worker.log_update.connect(self.log_print)
        self.thread.start()


class Worker(QObject):
    finished = pyqtSignal()  # signalling code for signalling to mother process of GUI.
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):

        pyqt_signal_dict = {"progress_bar": self.progress_update,
                            "text_log": self.log_update}

        image_types = [".jpg", ".png", ".jpeg", ".gif"]

        try:
            rm_lib.remove_files_in_directory(Path(Path.cwd(), "floorplans"), image_types)

            try:
                pyqt_signal_dict["text_log"].emit("Querying data from rightmove.com...")
                rm_data = rmw.RightmoveData(self.url)
                pyqt_signal_dict["text_log"].emit("Loaded RightmoveData!")
                pyqt_signal_dict["text_log"].emit(f"Total results: {rm_data.results_count}")
            except:
                traceback.print_exc()
                pyqt_signal_dict["text_log"].emit("Error in loading RightmoveData!")
                rm_data = None

            df = pd.DataFrame(rm_data.get_results)

            image_link_format = "https://www.rightmove.co.uk/properties/%s#/floorplan?activePlan=1&channel=RES_BUY"

            title_list = ((8, "image url", ""),
                          (9, "filename", ""),
                          (10, "Area (sqft)", np.nan),
                          (11, "Pounds/sqft", np.nan),
                          (12, "Is a flat?", ""),
                          (13, "Has a garden?", ""),
                          (14, "Results total:", ""))
            rm_lib.assign_titles(df, title_list)

            rm_lib.bs_pull_links(df, image_link_format, pyqt_signal_dict)

            image_folder = Path(Path.cwd(), "floorplans")
            image_folder.mkdir(exist_ok=True)
            excel_folder = Path(Path.cwd(), "excel files")
            excel_folder.mkdir(exist_ok=True)

            save_file = Path(excel_folder, str(time.strftime("%d-%m-%y (%H=%M)" + ".xlsx")))

            rm_lib.download_images(df, image_folder, image_types, pyqt_signal_dict)
            rm_lib.process_images(df, image_folder, image_types, pyqt_signal_dict)
            image_dict = rm_lib.images_to_text(df, image_folder, pyqt_signal_dict)

            keywords = ["total area", "total gross area", "total floor area", "total approx. floor area", "total",
                        "gross internal floor area", "gross internal area", "approximate area",
                        "approximate floor area",
                        "ground floor"]

            colour_dict = {"white": "#FFFFFF",
                           "lightgreen": "#90EE90",
                           "lemonchiffon": "#FFFACD",
                           "lightcoral": "#F08080"}

            rm_lib.find_area(df, image_dict, keywords, pyqt_signal_dict)
            styled_df = rm_lib.process_data(df, colour_dict, pyqt_signal_dict)
            rm_lib.df_to_excel(styled_df, save_file, rm_data, pyqt_signal_dict)

            playsound.playsound('notification.mp3')

        except:
            tb = traceback.format_exc()
            self.log_update.emit(tb)

        self.progress_update.emit(100)
        self.log_update.emit("Finished!")
        self.finished.emit()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
