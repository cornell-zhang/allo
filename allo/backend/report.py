# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=broad-exception-caught

import os
import re
import json
import time
import xmltodict
from tabulate import tabulate
import pandas as pd


class Displayer:
    """
    Queryable report displayer.

    ...

    Attributes
    ----------
    _category: list
        List of default latency information category.

    _category_aux: list
        List of latency information category with range indicators.

    _loop_name: list
        List of loop names without loop nest indicators.

    _loop_name_aux: list
        List of loop names with loop nest indicators.

    _max_level: int
        Maximum level of loop nest in the report file.

    _data: dict
        Dictionary containing latency data.

    unit: str
        Unit of each information.

    Methods
    ----------
    __checker(lst)
        Checks the validity of frame (unit of report data) list.

    __member_init(elem)
        Extract out properties of the report file.

    __data_acquisition(elem)
        Extract out latency information from the report file.

    init_table(obj)
        Initialize the attributes given the report file.

    collect_data(obj)
        Record latency information provided by the report file.

    get_max(col)
        Sort the latency in a decreasing order for specific latency category.

    display(loops=None, level=None, cols=None)
        Display the report table with appropriate query arguments.
    """

    def __init__(self, unit):
        """
        Parameters
        ----------
        unit: str
            Unit for all numerical values in the table.
        """
        self._category = [
            "TripCount",
            "Latency",
            "IterationLatency",
            "PipelineII",
            "PipelineDepth",
        ]
        self._category_aux = []
        self._loop_name = []
        self._loop_name_aux = []
        self._max_level = 0
        self._data = {}
        self.unit = unit

    def __is_valid(self, lst):
        """True if the given list of frames with its information are not
        empty. False otherwise.

        Parameters
        ----------
        lst: list
            List that contains information about the loop in report JSON file
            and its relevant reference information.

        Returns
        ----------
        bool
            Validity of the frame list.
        """
        valid = False
        for elem in lst:
            try:
                if elem[0]:
                    valid |= True
            # Except for the last non-dict case
            # FIXME: DO NOT USE THIS KIND OF CODING STYLE!
            except Exception:
                return False
        return valid

    def __member_init(self, elem):
        """Given values to a specific loop, update the class attributes
        accordingly.

        Parameters
        ----------
        elem: tuple
            Tuple containing information about latency values of the current
            loop, the loop name, and other reference information such as the
            current loop level.

        Returns
        ----------
        list
            List containing information about loops that are next-level down.
        """
        obj, loop, ref, level, loop_aux = elem[0], elem[1], elem[2], elem[3], elem[4]

        frame = []
        inner_loops = []
        loop.append(ref)

        if level == 0:
            loop_aux.append(ref)
        else:
            loop_aux.append("+" * level + " " + ref)

        if len(obj) != 0:
            for cat in self._category:
                if cat in obj and isinstance(obj[cat], dict):
                    for index, item in enumerate(self._data):
                        itemlist = list(item)
                        if itemlist[0] == cat:
                            itemlist[1] = 1
                        item = tuple(itemlist)
                        self._data[index] = item

        for k in list(obj.keys()):
            if k not in self._category:
                inner_loops.append(k)

        for il in inner_loops:
            frame.append((obj[il], loop, il, level + 1, loop_aux))

        if len(frame) == 0:
            frame.append(({}, loop, ref, level, loop_aux))

        return frame

    def __data_acquisition(self, elem):
        """From latency values in a specific loop, extract out the latency
        values.

        Parameters
        ----------
        elem: tuple
            Single-element tuple containing information about latency values
            of the current loop.

        Returns
        ----------
        list
            List containing information about loops that are next-level down.
        """
        obj, val_dict = elem[0], elem[1]

        frame = []
        inner_loops = []

        if len(obj) != 0:
            for cat in self._category_aux:
                cat_split = cat.split(" ", 1)
                val = "N/A"

                key = cat.replace(" ", "")
                # Not a min-max value
                if key in self._category:
                    try:
                        val = obj[key]
                    except Exception:
                        pass
                else:
                    cat_split = cat.split(" ", 1)
                    minmax = cat_split[0].lower()
                    key = cat_split[1].replace(" ", "")
                    if isinstance(obj[key], dict):
                        val = obj[key]["range"][minmax]
                    else:
                        val = obj[key]

                val_dict[cat] = val

        for s in list(obj.keys()):
            if s not in self._category:
                inner_loops.append(s)

        for il in inner_loops:
            frame.append((obj[il], val_dict))

        if len(frame) == 0:
            frame.append(({}, val_dict))

        return frame

    def init_table(self, obj):
        """Initialize attributes defined above for the specific report file.

        Parameters
        ----------
        obj: dict
            Dictionary representation of the report file.
        """
        keys = list(obj.keys())

        frame_lst = []

        for k in keys:
            frame_lst.append((obj[k], [], k, 0, []))

        # FIXME: Temporarily make use of data to be a list
        # pylint: disable=redefined-variable-type
        self._data = []
        for cat in self._category:
            self._data.append((cat, 0))

        while self.__is_valid(frame_lst):
            frame_lst = [self.__member_init(x) for x in frame_lst]

            frame_lst = [item for elem in frame_lst for item in elem]

        self._max_level = max(x[3] for x in frame_lst)

        filtered = [x[1] for x in frame_lst]
        filtered_aux = [x[4] for x in frame_lst]

        self._loop_name = [item for elem in filtered for item in elem]
        self._loop_name = list(dict.fromkeys(self._loop_name))

        self._loop_name_aux = [item for elem in filtered_aux for item in elem]
        self._loop_name_aux = list(dict.fromkeys(self._loop_name_aux))

        for cat, r in self._data:
            data_cat = re.sub(r"(\w)([A-Z])", r"\1 \2", cat)
            if r == 0:
                self._category_aux.append(data_cat)
            else:
                self._category_aux.append("Min " + data_cat)
                self._category_aux.append("Max " + data_cat)

        # Re-initialize the data field
        self._data = {}

    def collect_data(self, obj):
        """Collect latency data from the given report file.

        Parameters
        ----------
        obj: dict
            Dictionary representation of the report file.
        """
        keys = list(obj.keys())

        frame_lst = []

        fin_dict = {key: [] for key in self._category_aux}
        self._data = {key: [] for key in self._category_aux}

        for k in keys:
            frame_lst.append((obj[k], {}))

        while self.__is_valid(frame_lst):
            frame_lst = [self.__data_acquisition(x) for x in frame_lst]

            for cat in self._category_aux:
                store = []
                for elem in frame_lst:
                    try:
                        store.append(elem[0][1][cat])
                    except Exception:
                        pass
                fin_dict[cat].append(store)

            new_frame_lst = []
            for elem in frame_lst:
                for item in elem:
                    new_frame_lst.append((item[0], {}))

            frame_lst = new_frame_lst

        lev_seq = [x.count("+") for x in self._loop_name_aux]
        for cat in self._category_aux:
            for lev in lev_seq:
                self._data[cat].append(fin_dict[cat][lev].pop(0))

    def get_max(self, col):
        """Form a 3-element tuple list that sorts loops in a decreasing order
        with respect to the latency information of the specified latency
        category.

        Parameters
        ----------
        col: str
            Latency category name.

        Returns
        ----------
        list
            3-element tuple list with loop names, its data corresponding to
            [col] latency category, and the loop level.
        """
        tup_lst = list(zip(self._loop_name, self._data[col], self._loop_name_aux))
        tup_lst = [(x[0], x[1], x[2].count("+")) for x in tup_lst]
        return list(reversed(sorted(tup_lst, key=lambda x: int(x[1]))))

    def display(self, loops=None, level=None, cols=None):
        """Display the report file.

        Parameters
        ----------
        loops: list, optional
            List of loop names (e.g., ['A', 'Y'])
        level: int, optional
            Maximum level of loop nest to print.
        cols: list, optional
            List of column names. (e.g., ['Trip Count'])

        Returns
        ----------
        str
            String representation of pandas dataframe being displayed.
        """
        if loops is None:
            loops = self._loop_name_aux
        if level is None:
            level = self._max_level
        if cols is None:
            cols = self._category_aux

        selected = []
        for l in loops:
            if not isinstance(l, str):
                l = str(l).split(",", maxsplit=1)[0]
                # TODO: add support for axis value specification
                # If the item is a list of tuples, then that means the axis value was specified
                # stage, axis = l[0], l[1]
                # l[0], l[1] needs to be splitted
                # l = str(l[0]) + "_" + str(l[1])

            for k in self._loop_name_aux:
                if l in k and l not in selected:
                    selected.append(k)

        rows = []
        if level > self._max_level:
            rows = selected
        else:
            for k in selected:
                lev = k.count("+")
                if lev <= level:
                    rows.append(k)

        ncols = []
        for c in cols:
            for ca in self._category_aux:
                if c in ca:
                    ncols.append(ca)

        alignment = ("left",)
        for _ in range(len(cols)):
            alignment = alignment + ("right",)

        df = pd.DataFrame(data=self._data, index=self._loop_name_aux)
        print(
            tabulate(
                df.loc[rows, cols], headers=cols, tablefmt="psql", colalign=alignment
            )
        )
        print(f"* Units in {self.unit}")
        splt = df.loc[rows, cols].to_string().split("\n")
        pd.set_option("max_colwidth", len(splt[0]) * 100)
        return df.loc[rows, cols].to_string()


def parse_js(path):
    js_file = os.path.join(path, "kernel/reports/lib/report_data.js")
    if not os.path.isfile(js_file):
        raise RuntimeError(f"Cannot find {js_file}, run csyn first")

    # TODO: parse AOCL profiling report
    with open(js_file, "r", encoding="utf-8") as fp:
        js_scripts = fp.read()
        regex = r"total_kernel_resources.*?(\d+), (\d+), (\d+), (\d+), (\d+)"
        match = re.findall(regex, js_scripts)
        print(
            f"[{time.strftime('%H:%M:%S', time.gmtime())}] Parsing AOCL HLS report... "
        )
        LUT, FF, RAM, DSP, MLAB = match[0]
        print(f"[--------] ALUT : {LUT}")
        print(f"[--------] FF   : {FF}")
        print(f"[--------] RAM  : {RAM}")
        print(f"[--------] DSP  : {DSP}")
        print(f"[--------] MLAB : {MLAB}")


def parse_xml(path, prod_name, top="top", print_flag=False):
    xml_file = os.path.join(path, "out.prj", f"solution1/syn/report/{top}_csynth.xml")
    if not os.path.isfile(xml_file):
        raise RuntimeError(f"Cannot find {xml_file}, run csyn first")
    json_file = os.path.join(path, "report.json")
    with open(json_file, "w", encoding="utf-8") as outfile:
        with open(xml_file, "r", encoding="utf-8") as xml:
            profile = xmltodict.parse(xml.read())["profile"]
            json.dump(profile, outfile, indent=2)

    user_assignment = profile["UserAssignments"]
    perf_estimate = profile["PerformanceEstimates"]
    area_estimate = profile["AreaEstimates"]
    overall_latency = perf_estimate["SummaryOfOverallLatency"]

    res = {}
    res["HLS Version"] = prod_name + " " + profile["ReportVersion"]["Version"]
    res["Product family"] = user_assignment["ProductFamily"]
    res["Target device"] = user_assignment["Part"]
    clock_unit = user_assignment["unit"]
    res["Top Model Name"] = user_assignment["TopModelName"]
    res["Target CP"] = user_assignment["TargetClockPeriod"] + " " + clock_unit
    res["Estimated CP"] = (
        perf_estimate["SummaryOfTimingAnalysis"]["EstimatedClockPeriod"]
        + " "
        + clock_unit
    )
    res["Latency (cycles)"] = (
        f"Min {overall_latency['Best-caseLatency']:<6}; "
        + f"Max {overall_latency['Worst-caseLatency']:<6}"
    )
    res["Interval (cycles)"] = (
        f"Min {overall_latency['Interval-min']:<6}; "
        + f"Max {overall_latency['Interval-max']:<6}"
    )

    est_resources = area_estimate["Resources"]
    avail_resources = area_estimate["AvailableResources"]
    resources = {}
    for name in ("BRAM_18K", "DSP48E", "DSP", "FF", "LUT", "URAM"):
        if name not in est_resources:
            continue
        item = [est_resources[name], avail_resources[name]]
        item.append(f"{round(int(item[0]) / int(item[1]) * 100)}%")
        resources[name] = item.copy()
    res["Resources"] = tabulate(
        [[key] + value for key, value in resources.items()],
        headers=["Type", "Used", "Total", "Util"],
        colalign=("left", "right", "right", "right"),
    )
    lst = list(res.items())
    tablestr = tabulate(lst, tablefmt="psql").split("\n")
    endash = tablestr[0].split("+")
    splitline = "+" + endash[1] + "+" + endash[2] + "+"
    tablestr.insert(5, splitline)
    table = "\n".join(tablestr)

    # Latency information extraction
    clock_unit = overall_latency["unit"]
    info_table = Displayer(clock_unit)
    if "SummaryOfLoopLatency" in perf_estimate:
        summary = perf_estimate["SummaryOfLoopLatency"]
        info_table.init_table(summary)
        info_table.collect_data(summary)

    if print_flag:
        print(table)
        if "SummaryOfLoopLatency" in perf_estimate:
            info_table.display()
    return info_table


def report_stats(target, folder):
    path = folder
    if target.tool.name == "vivado_hls":
        if os.path.isdir(os.path.join(path, "out.prj")):
            return parse_xml(path, "Vivado HLS", top=target.top)
        raise RuntimeError("Not found out.prj folder")

    if target.tool.name == "aocl":
        if os.path.isdir(os.path.join(path, "kernel/reports")):
            return parse_js(path)
        raise RuntimeError("Not found out.prj folder")

    if target.tool.name == "vitis":
        if os.path.isdir(os.path.join(path, "out.prj")):
            return parse_xml(path, "Vitis HLS", top=target.top)
        raise RuntimeError("Not found out.prj folder")

    raise RuntimeError(f"tool {target.tool.name} not yet supported")
