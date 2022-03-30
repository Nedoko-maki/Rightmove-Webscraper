import collections
import itertools
import traceback
from pathlib import Path
import bs4
import numpy as np
import requests
import json
import urllib.request, urllib.error
import PIL
import pytesseract
import pandas as pd
import multiprocessing as mp


Msg = collections.namedtuple("Msg", ["event", "data"])


class ProcessManager:
    def __init__(self, pyqt_signal: dict):

        self.processes = {}
        self.queue = mp.Queue()
        self.pyqt_signal = pyqt_signal

    @staticmethod
    def _wrapper(func, pid, queue, args, kwargs):
        func(*args, pid=pid, queue=queue, **kwargs)  # function execution

    @staticmethod
    def _chunks(iterable, size):
        """Generate adjacent chunks of data"""
        it = iter(iterable)
        return iter(lambda: tuple(itertools.islice(it, size)), ())

    def run(self, pid, func, *args, **kwargs):
        """Start processes individually with user-managed resources."""
        args2 = (func, pid, self.queue, args, kwargs)
        proc = mp.Process(target=self._wrapper, args=args2)
        self.processes[pid] = {"pid": pid, "process": proc, "terminated": False}  # saving processes in a dict
        self.processes[pid]["process"].start()

    def map(self, func, work, *args, max_processes=mp.cpu_count(), **kwargs):
        """Map a function onto multiple processes, with this class managing the input."""

        work_list = self._chunks(work, max_processes)  # dividing work into smaller chunks

        for pid, work in enumerate(work_list):
            kwargs["work"] = work
            args2 = (func, pid, self.queue, args, kwargs)
            proc = mp.Process(target=self._wrapper, args=args2)
            self.processes[pid] = {"pid": pid, "process": proc, "terminated": False}
            self.processes[pid]["process"].start()

    def wait(self):  # waiting for processes to finish work, while updating the GUI
        return_list = []
        terminated = False
        while not terminated:
            for _ in self.processes:
                event, data = self.queue.get()

                if event == "return_data":  # event conditionals
                    return_list.append(data)
                elif event == "pyqt_signal":
                    self.pyqt_signal[data[0]].emit(data[1])  # can emit whatever PyQt signal depending on a list.
                elif event == "proc_terminate":
                    self.processes[data]["process"].join()  # process is terminated
                    self.processes[data]["terminated"] = True

                if all([self.processes[pid]["terminated"] for pid in self.processes]):
                    terminated = True
                    break

        return return_list


def f_type_return(file, file_type_list: list):  # takes string and file type list as input.
    for f_type in file_type_list:
        if str(file).lower().endswith(f_type):
            return str(f_type)


def remove_files_in_directory(directory: Path, file_types: list):
    files = [_ for _ in directory.glob("*") if f_type_return(_, file_types) in file_types]
    for file_path in files:
        file_path.unlink()


def get_id(temp_url: str):
    found = False
    first_index = temp_url.find("properties") + len("properties") + 1
    index = first_index
    while not found:
        if not temp_url[index].isnumeric():
            return temp_url[first_index:index]
        else:
            index += 1
        if index > 100:
            return 0


def bs_pull_links(df, image_link_format, pyqt_signal_dict):
    pyqt_signal_dict["text_log"].emit("Starting to pull html data from links...")
    for idx in range(len(df.index)):
        temp_url = df.at[idx, "url"]
        image_page = image_link_format % str(get_id(temp_url))
        page = requests.get(image_page)
        soup = bs4.BeautifulSoup(page.content, 'lxml')
        script = soup.find_all("script")
        fp_link = "Non-standard html: couldn't find window.PAGE_MODEL"
        for s in script:
            s = str(s)
            if "window.PAGE_MODEL = " in s:

                idx_1 = len("<script>     window.PAGE_MODEL = ")
                idx_2 = len(s) - len("</script>")
                json_string = s[idx_1:idx_2]
                json_object = json.loads(json_string)
                # with open(Path(Path.cwd(), "floorplans", str(idx) + ".json"), "w") as json_file:
                #     json.dump(json_object, json_file, indent=4)

                try:
                    if json_object["analyticsInfo"]["analyticsProperty"]["floorplanCount"] != 0:
                        fp_link = json_object["propertyData"]["floorplans"][0]["url"]
                    else:
                        fp_link = "No plan"
                    break
                except KeyError:
                    fp_link = "No plan"  # Occasionally bs4 doesn't return an analyticsInfo, rerunning it should work.

        df.at[idx, "image url"] = fp_link
        pyqt_signal_dict["progress_bar"].emit(round(100 * idx / len(df.index)))


def assign_titles(df, title_list):
    for position, title, value in title_list:
        df.insert(position, title, value)
        # reminder: variable value will fill the column with itself, and the column will
        # allow only one type. BE CAREFUL.


def download_images(df, image_folder: Path, image_types: list, pyqt_signal_dict):
    pyqt_signal_dict["text_log"].emit("Downloading images...")
    for idx in range(len(df.index)):
        if df.at[idx, "image url"] != "No plan":
            try:
                urllib.request.urlretrieve(df.at[idx, "image url"],
                                           Path(image_folder,
                                                str(idx) + f_type_return(df.at[idx, "image url"].lower(),
                                                                         image_types)).absolute())
                df.at[idx, "filename"] = str(idx) + f_type_return(df.at[idx, "image url"].lower(), image_types)
            except urllib.error.HTTPError:
                df.at[idx, "Area (sqft)"] = "HTTP Retrieval Error"
            except TypeError:
                print(f"{df.at[idx, 'image url']}, {f_type_return(df.at[idx, 'image url'].lower(), image_types)}, "
                      f"{type(f_type_return(df.at[idx, 'image url'].lower(), image_types))}, type error")
        pyqt_signal_dict["progress_bar"].emit(round(100 * idx / len(df.index)))
    pyqt_signal_dict["text_log"].emit("Done!")


def process_images(df, image_folder: Path, image_types: list, pyqt_signal_dict):
    pyqt_signal_dict["text_log"].emit("Processing images...")
    for idx in range(len(df.index)):
        if df.at[idx, "image url"] != "No plan":
            try:
                image = PIL.Image.open(Path(image_folder, df.at[idx, "filename"])).convert("L")
                nx, ny = image.size
                if int(nx) < 900 or int(ny) < 900:
                    image = image.resize((int(nx * 2), int(ny * 2)), PIL.Image.LANCZOS)
                image.save(Path(image_folder, str(idx) + ".jpg"))
                image.close()
                if f_type_return(df.at[idx, "filename"], image_types) != ".jpg":
                    Path(image_folder, df.at[idx, "filename"]).unlink()
                df.at[idx, "filename"] = str(idx) + ".jpg"
            except Exception:
                traceback.print_exc()
        pyqt_signal_dict["progress_bar"].emit(round(100 * idx / len(df.index)))
    pyqt_signal_dict["text_log"].emit("Done!")


def images_to_text(df: pd.DataFrame, image_folder: Path, pyqt_signal_dict):
    pyqt_signal_dict["text_log"].emit("Converting images to text...")

    image_list = list()
    for idx in range(len(df.index)):
        if df.at[idx, "image url"] != "No plan":
            image_list.append((idx, df.at[idx, "filename"]))

    proc_manager = ProcessManager(pyqt_signal_dict)
    proc_manager.map(_images_to_text, image_list, image_folder)
    list_of_dicts = proc_manager.wait()
    pyqt_signal_dict["text_log"].emit("Done!")
    return merge_dicts(list_of_dicts)


def _images_to_text(image_folder: Path, pid, queue, work=None):

    """
    Any map function param must have a pid, queue and a work keyword argument. There can be multiple positional
    arguments prepending pid and queue, and there can be keyword arguments in any order as long as the work kwarg
    exists.
    """

    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    image_list = work
    object_dict = dict()
    for idx, filename in image_list:
        try:
            image = PIL.Image.open(Path(image_folder, filename))
            text = pytesseract.image_to_string(image)
            image.close()
            object_dict[idx] = {"filename": filename, "text": text.lower()}
        except PermissionError:
            print(f"Permission error, filename: {filename}")
        except Exception:
            traceback.print_exc()
        queue.put(Msg("pyqt_signal", ["progress_bar", round(100 * idx / len(image_list))]))
    queue.put(Msg("return_data", object_dict))
    queue.put(Msg("proc_terminate", pid))


def merge_dicts(dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries. Python 3.9+
    """
    result = {}
    for dictionary in dict_args:
        result = result | dictionary
    return result


def find_number(text: str):
    _ = 0
    while True:
        if text[_].isnumeric():
            first_idx = _
            break
        else:
            _ += 1
        if _ > len(text) - 1:
            first_idx = 0
            break

    _ = first_idx
    dot_count = 0
    while True:
        if text[_] == ".":
            dot_count += 1
            if dot_count > 1:
                if text[_ - 1] == ".":  # in the case of 123..
                    second_idx = _ - 2
                else:  # in the case of 1.23.
                    second_idx = _ - 1
                break
        if not text[_].isnumeric() and text[_] != ".":
            second_idx = _
            break
        else:
            _ += 1
        if _ > len(text) - 1:
            second_idx = 0
            break
    return first_idx, second_idx


def find_number_reverse(text: str, start=None):
    _ = len(text) - 1 if start is None else start

    while True:
        if text[_].isnumeric():
            second_idx = _ + 1
            break
        else:
            _ -= 1
        if _ < 0:
            second_idx = 0
            break

    dot_count = 0
    while True:
        if text[_] == ".":
            dot_count += 1
            if dot_count > 1:
                if text[_ + 1] == ".":
                    first_idx = _ + 2  # in the case of ..123
                else:
                    first_idx = _ + 1  # in the case of .1.23
                break
        if not text[_].isnumeric() and text[_] != ".":
            first_idx = _ + 1
            break
        else:
            _ -= 1
        if _ < 0:
            first_idx = 0
            break
    return first_idx, second_idx


def clean_text(text, replace_set, allowed_char_set):
    for _ in text:  # removing any unrelated text
        if _ not in allowed_char_set and not _.isnumeric():
            replace_set.add(_)

    for string in replace_set:  # removing extra fluff from the text
        text = text.replace(string, "")

    return text


def find_area(df, image_dict: dict, keywords, pyqt_signal_dict: dict):  # The bulk of the text recognition logic.
    # a bona-fide nightmare.

    pyqt_signal_dict["text_log"].emit("Processing text...")

    unit_list = ["sqft", "sq.ft", "sqm", "sq.m", "ft", "m2"]
    sqft_unit_list = ["sqft", "sq.ft", "ft"]
    sqm_unit_list = ["sqm", "sq.m", "m2"]
    replace_set = {"\n", " ", ":", "(approx.)", "approx"}
    allowed_character_set = {"s", "q", "m", "f", "t", "."}

    for idx in image_dict:
        area_text = str()  # reset area_text for the next loop.
        for kw in keywords:  # First stage recognition. If np.nan is still in the ws cell, then move on to second stage
            # recognition logic.
            if image_dict[idx]["text"].find(kw) != -1:
                find_idx = image_dict[idx]["text"].find(kw)

                if len(image_dict[idx]["text"]) < find_idx + len(kw) + 60:
                    area_text = image_dict[idx]["text"][find_idx + len(kw):]
                else:
                    area_text = image_dict[idx]["text"][find_idx + len(kw):find_idx + len(kw) + 59]

                area_text = clean_text(area_text, replace_set, allowed_character_set)
                image_dict[idx]["text"] = area_text

                if not any(map(str.isnumeric, area_text)):
                    break

                s1_idx_1, s1_idx_2 = find_number(area_text)  # stage 1 index 1 & 2: s1_idx_1/2

                for unit in unit_list:
                    if area_text[s1_idx_2:s1_idx_2 + len(unit)] == unit:
                        area = np.nan
                        try:
                            area = float(area_text[s1_idx_1:s1_idx_2])
                        except ValueError:
                            print(f"Value error, ({s1_idx_1}:{s1_idx_2}): {area_text}\n{area_text[s1_idx_1:s1_idx_2]}")

                        if unit in ["sq.m", "sqm", "m2"]:
                            if area > 200:  # if unusually big house, find the next number which is a sqft num
                                s1_2_idx_1, s1_2_idx_2 = find_number(area_text[s1_idx_2 + 1:])
                                area = float(area_text[s1_idx_2 + 1 + s1_2_idx_1:s1_idx_2 + 1 + s1_2_idx_2])
                                image_dict[idx]["exit code"] = "sqm > 200, stage 1"
                                # setting var area knowing this is a sqft value, since the other one is sqm.
                                # reverse can be done, ie if area > 2000 sqft, find sqm backup measurement.
                            else:
                                image_dict[idx]["exit code"] = "sqft, stage 1, no complications."
                                area = round(10.76 * float(area), 1)  # conversion to sqft
                        df.at[idx, "Area (sqft)"] = area
                        break
                break

        if len(area_text) == 0:  # if there are no kw in the text, assign the whole area_text from image_dict.
            area_text = image_dict[idx]["text"]

        if np.isnan(df.at[idx, "Area (sqft)"]) and any(map(str.isnumeric, area_text)):
            # second stage, where we have to try identify the area via unit
            # key words / str.find().

            area_text = clean_text(area_text, replace_set, allowed_character_set)
            image_dict[idx]["text"] = area_text

            for unit in unit_list:
                if area_text.find(unit) != -1:
                    s2_idx_1, s2_idx_2 = find_number_reverse(area_text, start=area_text.find(unit))
                    area = np.nan
                    try:
                        area = float(area_text[s2_idx_1:s2_idx_2])
                    except ValueError:
                        print(f"Value error, ({s2_idx_1}:{s2_idx_2}): {area_text}\n{area_text[s2_idx_1:s2_idx_2]}")

                    if unit in ["sq.m", "sqm", "m2"]:
                        area = round(10.76 * float(area), 1)
                        image_dict[idx]["exit code"] = "sqft, stage 2, no complications."
                    if area > 2000 or area < 1:
                        if unit in ["sq.m", "sqm", "m2"]:  # in the case that area is larger than 2000sqft.
                            for unit_2 in sqft_unit_list:
                                if area_text.find(unit_2) != -1:
                                    s2_idx_1, s2_idx_2 = find_number_reverse(area_text, start=area_text.find(unit_2))
                                    area = float(area_text[s2_idx_1:s2_idx_2])
                                    image_dict[idx]["exit code"] = "sqm > 200, stage 2"
                                    break
                        else:
                            for unit_2 in sqm_unit_list:
                                if area_text.find(unit_2) != -1:
                                    s2_idx_1, s2_idx_2 = find_number_reverse(area_text, start=area_text.find(unit_2))
                                    area = float(area_text[s2_idx_1:s2_idx_2])
                                    area = round(10.76 * float(area), 1)
                                    image_dict[idx]["exit code"] = "sqft > 2000, stage 2"
                                    break
                    df.at[idx, "Area (sqft)"] = area
                    break

        image_dict[idx]["located area"] = df.at[idx, "Area (sqft)"]

    with open(Path(Path.cwd(), "floorplans", "image_dict.json"), "w") as json_file:
        json.dump(image_dict, json_file, indent=4)

    pyqt_signal_dict["text_log"].emit("Done!")


def process_data(df, colour_dict, pyqt_signal_dict):
    count, no_plan_count = 0, 0
    for idx in range(len(df.index)):

        if not np.isnan(df.at[idx, "Area (sqft)"]):  # check yield
            count += 1
        if df.at[idx, "image url"] == "No plan":  # yield takes no-plans into account, subtracts at the end.
            no_plan_count += 1

        value, area = df.at[idx, "price"], df.at[idx, "Area (sqft)"]  # calculate ppsqft

        if not pd.isnull(value) and not np.isnan(area) and area != 0:
            # reminder, you can't do np.nan == np.nan, ret false.
            # use one of the above functions in pd or np.
            df.at[idx, "Pounds/sqft"] = round(value / area)

            df.at[idx, "Is a flat?"] = "Yes" if (
                        "apartment" in df.at[idx, "type"] or "flat" in df.at[idx, "type"]) else "No"

    df.sort_values(by=["Pounds/sqft"], inplace=True)
    df.iat[0, 14] = f"Total results: {len(df.index)}"
    df.iat[1, 14] = f"Succeeded: {count}"
    df.iat[2, 14] = f"No plans: {no_plan_count}"
    df.iat[3, 14] = f"Failed: {len(df.index) - count - no_plan_count}"
    df.iat[4, 14] = f"Yield: {round(100 * count / (len(df.index) - no_plan_count), 1)}%"
    df = df.style.applymap(apply_colour, colour_dict=colour_dict, subset=["Pounds/sqft"])
    pyqt_signal_dict["text_log"].emit(f"Yield: {round(100 * count / (len(df.index) - no_plan_count), 1)}%")
    return df


def apply_colour(val, colour_dict):
    colour = colour_dict["white"]
    if val <= 400:
        colour = colour_dict["lightgreen"]
    elif 400 < val < 500:
        colour = colour_dict["lemonchiffon"]
    elif val >= 500:
        colour = colour_dict["lightcoral"]
    return f"background-color: {colour}"


def df_to_excel(df, save_file, rm_data, pyqt_signal_dict):
    id_list = [("number_bedrooms", "BY BEDROOMS"), ("postcode", "BY POSTCODE"), ("type", "BY TYPE")]
    writer = pd.ExcelWriter(save_file, engine='xlsxwriter')

    df.to_excel(writer, sheet_name="FULL", index=False)
    pyqt_signal_dict["text_log"].emit("Saving dataframes to excel spreadsheet...")
    for internal_name, sheet_name in id_list:
        temp_df = pd.DataFrame(rm_data.summary(by=internal_name))
        temp_df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()
    pyqt_signal_dict["text_log"].emit("Done!")
