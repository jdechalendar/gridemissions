from traitlets import TraitError
import ipywidgets as widgets
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import display


class BaDataGui(object):
    def __init__(self, data):
        # Avoid annoying warning
        from pandas.plotting import register_matplotlib_converters

        register_matplotlib_converters()

        self.data = data
        self.variable = data.variable
        self.KEY = data.KEY
        if self.variable == "E":
            self.scaling = 1e-3
            self.ylabel = "GWh"
        elif self.variable == "CO2":
            self.scaling = 1e-6
            self.ylabel = "kton"
        elif self.variable == "CO2i":
            self.scaling = 1
            self.ylabel = "kg/MWh"
        else:
            raise NotImplementedError

    def make_viz(self):
        # Create controls
        # Initialize column type controller with demand
        fields = []
        for k in list(self.KEY.keys()):
            if len([el for el in self.data.df.columns if self._matches(k, el)]) > 0:
                fields.append(k)
        self.ctl_filter = widgets.Dropdown(options=fields + ["ALL"])

        self.ctl_cols = widgets.Dropdown(
            options=sorted(
                [col for col in self.data.df.columns if self._matches("D", col)]
            )
        )

        dates = pd.date_range(self.data.df.index[0], self.data.df.index[-1], freq="D")
        options = [(date.strftime(" %d/%m/%Y "), date) for date in dates]
        index = (0, len(options) - 1)
        self.ctl_date = widgets.SelectionRangeSlider(
            options=options,
            index=index,
            description="Dates",
            orientation="horizontal",
            continuous_update=False,
            layout=widgets.Layout(width="80%"),
        )

        self.ctl_filter.observe(self._ctl_filter_handler, names=["value"])
        self.ctl_cols.observe(self._ctl_cols_handler, names=["value"])
        self.ctl_date.observe(self._ctl_date_handler, names=["value"])

        self.input_widgets = widgets.VBox(
            [widgets.HBox([self.ctl_filter, self.ctl_cols]), self.ctl_date]
        )
        self.output = widgets.Output()
        # initalize plot before making it
        self._make_plot(self.ctl_cols.value, self.ctl_date.value)

    def display(self):
        display(self.input_widgets)
        display(self.output)

    def _matches(self, field, col):
        n = len(re.findall("%s", self.KEY[field]))
        return re.match(self.KEY[field] % tuple([r"\w+"] * n), col)

    def _find(self, field, col):
        n = len(re.findall("%s", self.KEY[field]))
        res = re.findall(self.KEY[field] % tuple([r"(\w+)"] * n), col)
        if len(res) == 1:
            return res[0]
        else:
            print(res)
            print(field)
            print(col)
            raise NotImplementedError

    def _ctl_filter_handler(self, change):
        if change.new == "ALL":
            self.ctl_cols.options = self.data.df.columns

        else:
            old_col = self.ctl_cols.value
            self.ctl_cols.options = sorted(
                [col for col in self.data.df.columns if self._matches(change.new, col)]
            )

            if change.old != "ALL":
                ba = self._find(change.old, old_col)
                try:
                    self.ctl_cols.value = self.KEY[change.new] % ba
                except (TypeError, TraitError):
                    # When going to or from the ID field
                    # Or when the ba doesn't have that field (e.g. for
                    # generation sources)
                    pass

    def _ctl_cols_handler(self, change):
        self._make_plot(change.new, self.ctl_date.value)

    def _ctl_date_handler(self, change):
        self._make_plot(self.ctl_cols.value, change.new)

    def _make_plot(self, col=None, date_rng=None):
        if col is None:
            col = self.ctl_cols.options[0]
        if date_rng is None:
            date_rng = self.ctl_date.value

        self.output.clear_output()
        with self.output:
            f, ax = plt.subplots(figsize=(15, 5))
            try:
                ax.plot(self.data.df.loc[date_rng[0] : date_rng[1], col] * self.scaling)
            except KeyError:
                print(col)
                raise
            ax.grid()
            ax.set_ylabel(self.ylabel)
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            f.autofmt_xdate()
            plt.show()


class BaDataGuiFlex(BaDataGui):
    def __init__(self, data, plotter):
        """

        Parameters
        ----------
        plotter : callable
            Custom function to make plots
        """
        super().__init__(data)
        self.plotter = plotter

    def _make_plot(self, col=None, date_rng=None):
        if col is None:
            col = self.ctl_cols.options[0]
        if date_rng is None:
            date_rng = self.ctl_date.value

        self.output.clear_output()
        with self.output:
            f, ax = plt.subplots(figsize=(15, 5))
            try:
                self.plotter(
                    ax, self.data.df.loc[date_rng[0] : date_rng[1], col] * self.scaling
                )
            except KeyError:
                print(col)
                raise
            ax.grid()
            ax.set_ylabel(self.ylabel)
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            f.autofmt_xdate()
            plt.show()


class BaDataGuiComp(BaDataGui):
    def __init__(self, data1, data2, labels=["data1", "data2"], diff=False):
        super().__init__(data1)
        self.data2 = data2
        self.labels = labels
        self.diff = diff

        if data2.variable != self.variable:
            raise ValueError("Both variables should be the same!")

    def _make_plot(self, col=None, date_rng=None):
        if col is None:
            col = self.ctl_cols.options[0]
        if date_rng is None:
            date_rng = self.ctl_date.value

        if col not in self.data2.df:
            # Edge case: the column was added in self.data
            # but didn't exist in self.data2
            self.data2.df.loc[:, col] = np.nan

        self.output.clear_output()
        with self.output:
            f, ax = plt.subplots(figsize=(15, 5))
            try:
                if self.diff:
                    ax.plot(
                        (
                            self.data.df.loc[date_rng[0] : date_rng[1], col]
                            - self.data2.df.loc[date_rng[0] : date_rng[1], col]
                        )
                        * self.scaling
                    )
                else:
                    ax.plot(
                        self.data.df.loc[date_rng[0] : date_rng[1], col] * self.scaling
                    )
                    ax.plot(
                        self.data2.df.loc[date_rng[0] : date_rng[1], col] * self.scaling
                    )
            except KeyError:
                print(col)
                raise
            ax.grid()
            ax.set_ylabel(self.ylabel)
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            f.autofmt_xdate()
            ax.legend(self.labels)
            plt.show()
