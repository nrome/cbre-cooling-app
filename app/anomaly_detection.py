# -*- coding: utf-8 -*-

"""
The objective of this class is to, on a time series, execute the following
tasks:
 - determine the root / decomposed trend line on a time series using
 ARIMA / exponential smoothing
 - identify, on a raw time series OR decomposed trend line, the
 outliers based on algorithms such as:
     - moving z score
     - local outlier factor
     - ...
 - plot these items in a way to easily visualize anomalies that have occurred
"""


import matplotlib.pyplot as plt
import mpld3
import numpy as np
import scipy.stats as scs
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.neighbors import LocalOutlierFactor


class TSAnomalyDetection(object):
    """
    This class is in charge of implementing time series anomaly detection
    on pandas series, and it is going to expose itself in the same fashion
    as scikit learn where two algorithms are used:
     - Moving Z Score w/ Sliding Window
     - Moving ANOVA w/ Sliding Window
     - Croston's Algorithm

    tsad = TSAnomalyDetection(a=.05, w=8046)
    tsad.fit(data)
    tsad.plot()
    """

    def __init__(self, a=.001, w=8046, decomposition=False,
                 n_neighbors=14):
        """
        Args:
            a (float): this is the alpha level for the width of the prediction
            interval
            w (int): this is the size of the sliding window in terms of number
            of periods
        """
        self.a = a
        self.w = w
        self.decomposition = decomposition
        self.n_neighbors = n_neighbors

    def _moving_z_score(self):
        """
        Args:
            None
        Returns:
            None, creates the features for the data
        """
        self.mean = self.data.rolling(self.w).mean()
        std = self.data.rolling(self.w).std()
        coefficient = scs.norm.ppf(1 - self.a / 2)
        upper_pi = self.mean + coefficient * std
        lower_pi = self.mean - coefficient * std
        self.up_out = self.data[(self.data > upper_pi)]
        self.lo_out = self.data[(self.data < lower_pi)]
        if self.decomposition is True:
            # this is information with seasonal decomposition
            decomposition = seasonal_decompose(self.data, freq=self.w)
            # this is the trend information
            self.trend = decomposition.trend
            self.trend_mean = self.trend.rolling(self.w).mean()
            std = self.trend.rolling(self.w).std()
            coefficient = scs.norm.ppf(1 - self.a / 2)
            upper_pi = self.trend_mean + coefficient * std
            lower_pi = self.trend_mean - coefficient * std
            self.trend_up_out = self.trend[(self.trend > upper_pi)]
            self.trend_lo_out = self.trend[(self.trend < lower_pi)]
            # this is the seasonal information
            self.seasonal = decomposition.seasonal
            self.seasonal_mean = self.seasonal.rolling(self.w).mean()
            std = self.seasonal.rolling(self.w).std()
            coefficient = scs.norm.ppf(1 - self.a / 2)
            upper_pi = self.seasonal_mean + coefficient * std
            lower_pi = self.seasonal_mean - coefficient * std
            self.seasonal_up_out = self.seasonal[(self.seasonal > upper_pi)]
            self.seasonal_lo_out = self.seasonal[(self.seasonal < lower_pi)]
            # this is the residual information
            self.residual = decomposition.resid
            self.residual_mean = self.residual.rolling(self.w).mean()
            std = self.data.rolling(self.w).std()
            coefficient = scs.norm.ppf(1 - self.a / 2)
            upper_pi = self.residual_mean + coefficient * std
            lower_pi = self.residual_mean - coefficient * std
            self.residual_up_out = self.residual[(self.residual > upper_pi)]
            self.residual_lo_out = self.residual[(self.residual < lower_pi)]

    def _plot_moving_z_score(self, name=None, plot_type='show',
                             data='original'):
        """
        Args:
            name (str): this is the name of the metric where anomaly
            detection is to be plotted
            figsize (tuple): This is a tuple noted by (W, H) where W is the
            width of the figure and H is the height
            plot_type (str): this is the nature of the plot which is going to
            have the following possible outputs:
                 - 'show': will show the plot
                 - 'html': will return html as a string
                 - 'json': will return the plot as json
            data (str): this is the data which is going to be trended,
            options for this can be: {'data', 'trend', 'seasonal', 'residuals'}
        Returns:
            html (str): This is the output html representation of the figure
            when web is True
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if name:
            fig.suptitle('Outlier Chart for {}'.format(name))
        else:
            fig.suptitle('Outlier Chart')
        if data == 'original':
            values = self.data
            mean = self.mean
            up_out = self.up_out
            lo_out = self.lo_out
            label = 'original'
        elif data == 'trend':
            values = self.trend
            mean = self.trend_mean
            up_out = self.trend_up_out
            lo_out = self.trend_lo_out
            label = 'trend'
        elif data == 'seasonal':
            values = self.seasonal
            mean = self.seasonal_mean
            up_out = self.seasonal_up_out
            lo_out = self.seasonal_lo_out
            label = 'seasonal'
        elif data == 'residual':
            values = self.residual
            mean = self.residual_mean
            up_out = self.residual_up_out
            lo_out = self.residual_lo_out
            label = 'residual'
        ax.plot(values.index, values, c='k', label=label)
        ax.plot(mean.index, mean, c='g', label='moving average')
        ax.plot(up_out.index, up_out, 'ro', label='upper outliers')
        ax.plot(lo_out.index, lo_out, 'bo', label='lower outliers')
        ax.legend(loc='best')
        if plot_type == 'show':
            plt.show()
        elif plot_type == 'html':
            plot_output = mpld3.fig_to_html(fig)
            return plot_output
        elif plot_type == 'json':
            plot_output = mpld3.fig_to_dict(fig)
            return plot_output

    def _local_outlier_factor(self):
        """
        Args:
            None
        Returns:
            None, creates the features for the data using LOF
        """
        cont_pct = 1 / float(len(self.data.columns))
        clf = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                 n_jobs=-1, metric='cosine',
                                 contamination=cont_pct)
        self.y_pred = clf.fit_predict(self.data.T.values)
        self.normal = np.array(self.data.columns)[self.y_pred == 1]
        self.anomalies = np.array(self.data.columns)[self.y_pred == -1]

    def _plot_local_outlier_factor(self, name=None, plot_type='show'):
        """
        Args:
            name (str): this is the name of the metric where anomaly
            detection is to be plotted
            figsize (tuple): This is a tuple noted by (W, H) where W is the
            width of the figure and H is the height
            plot_type (str): this is the nature of the plot which is going to
            have the following possible outputs:
                 - 'show': will show the plot
                 - 'html': will return html as a string
                 - 'json': will return the plot as json
            data (str): this is the data which is going to be trended,
            options for this can be: {'data', 'trend', 'seasonal', 'residuals'}
        Returns:
            html (str): This is the output html representation of the figure
            when web is True
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle('Peer Analytics Outlier Chart for {}'.format(name))
        plt.title('Detected Outliers: {}'.format(', '.join(self.anomalies)))
        for crah in self.normal:
            ax.plot(self.data[crah].index, self.data[crah],
                    alpha=.5, c='b', label=crah)
        for crah in self.anomalies:
            ax.plot(self.data[crah].index, self.data[crah],
                    alpha=.5, c='r', label=crah)
        ax.legend(loc='best')
        if plot_type == 'show':
            plt.show()
        elif plot_type == 'html':
            plot_output = mpld3.fig_to_html(fig)
            return plot_output
        elif plot_type == 'json':
            plot_output = mpld3.fig_to_dict(fig)
            return plot_output

    def plot(self, name=None, plot_type='show',
             data='original'):
        """
        Args:
            name (str): this is the name of the metric where anomaly
            detection is to be plotted
            figsize (tuple): This is a tuple noted by (W, H) where W is the
            width of the figure and H is the height
            plot_type (str): this is the nature of the plot which is going to
            have the following possible outputs:
                 - 'show': will show the plot
                 - 'html': will return html as a string
                 - 'json': will return the plot as json
            data (str): this is the data which is going to be trended,
            options for this can be: {'data', 'trend', 'seasonal', 'residuals'}
        Returns:
            html (str): This is the output html representation of the figure
            when web is True
        """
        if len(self.data.shape) == 1:
            return self._plot_moving_z_score(name, plot_type,
                                             data=data)
        else:
            return self._plot_local_outlier_factor(name, plot_type)

    def fit(self, data):
        """
        Args:
            data (pandas timeseries): this is a pandas time series where
            the index is a datetime object
        Returns:
            None, creates the features for the data
        """
        # this is information for the basic plot
        self.data = data
        if len(self.data.shape) == 1:
            self._moving_z_score()
        else:
            self._local_outlier_factor()


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('../output/sample_anova.csv', parse_dates=['date_time'])
    df.set_index('date_time', inplace=True)
    tsad = TSAnomalyDetection()
    tsad.fit(df)
    plot = tsad.plot(plot_type='html')
