# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns
import timeit
from dashdb_connect import DashdbConnect
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from scipy.spatial.distance import cosine


class CrahOptimizer(object):
    """
    This class is in charge of executing the optimization algorithms
    for crah fan speeds and set-point temperatures by reading the sensor
    information and controlling these accordingly
    """

    def __init__(self, start='2017-08-01', end='2017-10-31',
                 threshold=27, max_delta=10, crah_only=False,
                 static_params=False, w=16, model='lr', evaluation=False,
                 local=False, verbose=False,):
        """
        Args:
            crah (str): this is the code of the crah unit for which
            optimization is going to be computed
            start (str): this is the starting date/time in YYYY-MM-DD format
            for the sql query
            end (str): this is the ending date/time in YYYY-MM-DD format
            for the sql query
            threshold (int): this is the value of the maximum inlet temperature
            permitted by policy in this data center
            max_delta (float): this is the maximum number of degC that the
            supply air temp can be increased to
            crah_only (boolean): True if the data pulled will only be for
            this crah else false if it is going to include data for another
            crah as well
            static_params (boolean): True if we are going to use the
            static parameters computed by running a regression on the whole
            time series else False if only for the crahs
            w (int): This is the size of the sliding window in number of hours
            verbose (boolean): True if the seconds execution time is
            going to be printed else False if not
            model (str): this is the kind of model that is going to be used
            in the prediction
            evaluation (boolean): True if model evaluation functions are to be
            executed such as cross validation etc, else False
            verbose (boolean): True if the class is going to report progress
            on execution speed else False
        NOTE - implement, later on, changing w with a range of days and hours
        """
        self.start = start
        self.end = end
        self.verbose = verbose
        if self.verbose:
            now = timeit.default_timer()
        self.db = DashdbConnect()
        if self.verbose:
            print 'connecting to db: ', timeit.default_timer() - now
        self.energy_cost = 4.73
        self.threshold = threshold
        self.max_delta = max_delta
        if static_params:
            crah = 'DH1'
            crah_df = self.db.query_db('SELECT * FROM SAT_FAN_PARAMS;')
            crah_df.set_index('params', inplace=True)
            crah_df = crah_df.astype(float)
            self.params = crah_df[crah].to_dict()
        self.crah_units = {
            "DH10", "DH11", "DH14", "DH18", "DH20", "DH21", "DH23", "DH24",
            "DH26", "DH4", "DH5", "DH7", "DH9", "DH1", "DH2", "DH15", "DH17",
            "DH13"
        }
        self.w = w
        self.model = model
        self.evaluation = evaluation
        self.a_step = .1
        self.m_factor = 2
        self.ex_pct_threshold = .2
        self.local = local

    def load_crah_data(self, crah):
        """
        Args:
            crah (str): this is the code of the crah unit to be retrieved
        Returns:
            None, loads the crah unit onto the crah attribute of the class
        """
        if self.verbose:
            now = timeit.default_timer()
        sensor_query = \
            '''
            SELECT "sensor_id"
            FROM SENSOR_TO_CRAH
            WHERE "acu_name" = '{crah}'
            '''.format(crah=crah)
        sensors = self.db.query_db(sensor_query)
        sensors_list = \
            (['"{}"'.format(str(sensor['sensor_id']))
              for sensor in sensors])
        sensors_string = ', '.join(sensors_list)
        inlet_query = \
            '''
            SELECT "date_time", {sensors}
                FROM SERVER_INLET
            WHERE "date_time" BETWEEN '{start}' AND '{end}'
            ORDER BY "date_time" ASC
            '''.format(sensors=sensors_string, start=self.start,
                       end=self.end)
        joint_query = \
            '''
            SELECT *
            FROM
                ({subtable}) AS l JOIN {crah} AS r
                ON l."date_time" = r."date_time"
            WHERE l."date_time" BETWEEN '{start}' AND '{end}'
            ORDER BY l."date_time" ASC;
            '''.format(subtable=inlet_query,
                       crah=crah, start=self.start,
                       end=self.end)
        self.crah = \
            self.db.query_df(joint_query)
        sensor_list = sensors_string.replace('"', '').split(', ')
        self.server_inlets = self.crah[sensor_list]
        self.crah.drop(sensor_list, axis=1, inplace=True)
        self.crah = self.crah[['sat_interp', 'fan_spd_interp']]
        self.crah.columns = ['sat', 'fan_spd']
        max_server_inlet = pd.DataFrame(self.server_inlets.T.max(),
                                        columns=['server_max'])
        self.crah = self.crah.join(max_server_inlet)
        if self.verbose:
            print 'query db: ', timeit.default_timer() - now
        self.crah['hour'] = pd.DatetimeIndex(self.crah.index).hour
        self.crah['doy'] = \
            pd.DatetimeIndex(self.crah.index).dayofyear
        self.crah['day_of_year'] = \
            pd.DatetimeIndex(self.crah.index).dayofyear

    def local_crah_data(self, crah):
        """
        Args:
            crah (str): this is the code of the crah unit to be retrieved
        Returns:
            None, loads the crah unit onto the crah attribute of the class
        """
        if self.verbose:
            now = timeit.default_timer()
        self.crah = pd.read_csv('data/{}.csv'.format(crah),
                                parse_dates=['date_time'])
        self.crah.set_index('date_time', inplace=True)
        if crah == 'DH13':
            self.crah['fan_spd_interp'] = 50.0
        self.crah = self.crah[['sat_interp', 'rat_interp', 'fan_spd_interp',
                               'max_inlet_temp']]
        self.crah.columns = ['sat', 'rat', 'fan_spd', 'server_max']
        if self.verbose:
            print 'local data: ', timeit.default_timer() - now
        self.crah['hour'] = pd.DatetimeIndex(self.crah.index).hour
        self.crah['doy'] = \
            pd.DatetimeIndex(self.crah.index).dayofyear
        self.crah['day_of_year'] = \
            pd.DatetimeIndex(self.crah.index).dayofyear

    def _predict_inlet(self, sat, fan_spd, delta_sat):
        """
        Args:
            sat (float): this is the supply air temperature in degC
            fan_spd (float): this is the fan_spd in output %
            delta_sat (float): this is the incremental change to the
            supply air temperature
        Returns:
            inlet_temp (float): this is the predicted value of
            the inlet temp
       """
        inlet_temp = self.params['const'] \
            + self.params['fan_spd_interp'] * fan_spd \
            + self.params['sat_interp'] * (sat + delta_sat)
        return inlet_temp

    def _get_max_dsat(self, sat, fan_spd):
        """
        Args:
            sat (float): this is the supply air temperature in degC
            fan_spd (float): this is the fan_spd in output %
        Returns:
            delta_sat (float): this is the incremental change
            to the sat based on what is maximally permissible
            by this observation
        NOTE - this handles individual observations
        """
        inlet_temp = self._predict_inlet(sat, fan_spd, 0.1)
        if np.greater(inlet_temp, self.threshold):
            return 0.0
        else:
            dsat_array = np.linspace(.1, self.max_delta, self.max_delta * 100)
            sat_array = np.ones(len(dsat_array)) * sat
            fan_array = np.ones(len(dsat_array)) * fan_spd
            inlet_array = np.apply_along_axis(self._predict_inlet, 0,
                                              sat_array, fan_array, dsat_array)
            threshold = inlet_array < self.threshold
            delta_sat = dsat_array[np.argmax(inlet_array[threshold])]
        return delta_sat

    def _predict_max_eca(self, sat, fan_spd):
        """
        Args:
            sat (float): this is the supply air temperature in degC
            fan_spd (float): this is the fan_spd in output %
            delta_sat (float): this is the incremental change to the
            supply air temperature
        Returns:
            eca (float): this is the predicted cost to be saved given
            the change in supply air temperature
        """
        delta_sat = self._get_max_delta_sat(sat, fan_spd)
        Q = 1.006 * 1.202 * 1.88779 * fan_spd / 100.0 * delta_sat
        eca = Q * (1.0 / 12.0) * self.energy_cost
        return eca

    def predict_total_max_eca(self):
        """
        Args:
            delta_sat (float): this is the incremental change to the
            supply air temperature
        Returns:
            total_eca (float): this is the predicted TOTAL cost to be saved
            on a series of historical data given the change in
            supply air temperature
        NOTE - make sure to reflect that sub_crah has fewer total rows than
        the actual crah time series as a whole, thus the total eca is only
        for the period with available data
        """
        sub_crah = self.crah[['sat', 'fan_spd', 'server_max']].dropna().copy()
        # sub_crah['delta_sat'] = \
        #     np.apply_along_axis(self._get_max_dsat, 0,
        #                         sub_crah['sat'],
        #                         sub_crah['fan_spd'])
        sub_crah['delta_sat'] = \
            sub_crah.apply(lambda row: self._get_max_dsat(row['sat'],
                                                          row['fan_spd']),
                           axis=1)
        eca = np.apply_along_axis(self._predict_eca, 0,
                                  sub_crah['sat'],
                                  sub_crah['fan_spd'],
                                  sub_crah['delta_sat'])
        total_eca = np.sum(eca)
        return total_eca

    def estimate_total_max_eca(self):
        """
        Args:
            delta_sat (float): this is the incremental change to the
            supply air temperature
        Returns:
            total_eca (float): this is the predicted TOTAL cost to be saved
            on a series of historical data given the change in
            supply air temperature
        NOTE - make sure to reflect that sub_crah has fewer total rows than
        the actual crah time series as a whole, thus the total eca is only
        for the period with available data
        """
        sub_crah = self.crah[['sat', 'fan_spd', 'server_max']].dropna().copy()
        # sub_crah['delta_sat'] = \
        #     np.apply_along_axis(self._get_max_dsat, 0,
        #                         sub_crah['sat'],
        #                         sub_crah['fan_spd'])
        sub_crah['delta_sat'] = \
            sub_crah.apply(lambda row: self._get_max_dsat(row['sat'],
                                                          row['fan_spd']),
                           axis=1)
        eca = np.apply_along_axis(self._predict_eca, 0,
                                  sub_crah['sat'],
                                  sub_crah['fan_spd'],
                                  sub_crah['delta_sat'])
        total_eca = np.sum(eca)
        return total_eca / float(sub_crah.shape[0]) * self.crah.shape[0]

    def _predict_eca(self, sat, fan_spd, delta_sat):
        """
        Args:
            sat (float): this is the supply air temperature in degC
            fan_spd (float): this is the fan_spd in output %
            delta_sat (float): this is the incremental change to the
            supply air temperature
        Returns:
            eca (float): this is the predicted cost to be saved given
            the change in supply air temperature
        """
        # Q = 1.006 * 1.202 * 1.88779 * fan_spd / 100.0 * delta_sat
        # eca = Q * (1.0 / 12.0) * self.energy_cost
        Q = (0.2404 * 0.07393) * (fan_spd / 100.0) * 24000.0 * \
            (9.0 / 5.0) * delta_sat * self.energy_cost * 60.0
        eca = Q / (1000000.0 * 12.0)
        return eca

    def predict_total_eca(self, delta_sat):
        """
        Args:
            delta_sat (float): this is the incremental change to the
            supply air temperature
        Returns:
            total_eca (float): this is the predicted TOTAL cost to be saved
            on a series of historical data given the change in
            supply air temperature
        NOTE - make sure to reflect that sub_crah has fewer total rows than
        the actual crah time series as a whole, thus the total eca is only
        for the period with available data
        """
        sub_crah = self.crah[['sat', 'fan_spd', 'server_max']].dropna().copy()
        sub_crah['delta_sat'] = delta_sat
        eca = np.apply_along_axis(self._predict_eca, 0,
                                  sub_crah['sat'],
                                  sub_crah['fan_spd'],
                                  sub_crah['delta_sat'])
        total_eca = np.sum(eca)
        return total_eca

    def estimate_total_eca(self, delta_sat):
        """
        Args:
            delta_sat (float): this is the incremental change to the
            supply air temperature
        Returns:
            total_eca (float): this is the predicted TOTAL cost to be saved
            on a series of historical data given the change in
            supply air temperature
        NOTE - make sure to reflect that sub_crah has fewer total rows than
        the actual crah time series as a whole, thus the total eca is only
        for the period with available data
        """
        if self.verbose:
            now = timeit.default_timer()
        sub_crah = self.crah[['sat', 'fan_spd', 'server_max']].dropna().copy()
        sub_crah['delta_sat'] = delta_sat
        eca = np.apply_along_axis(self._predict_eca, 0,
                                  sub_crah['sat'],
                                  sub_crah['fan_spd'],
                                  sub_crah['delta_sat'])
        total_eca = np.sum(eca)
        if self.verbose:
            print 'run CAP computations: ', timeit.default_timer() - now
        return total_eca / float(sub_crah.shape[0]) * self.crah.shape[0]

    def _chunks(self, list_to_chunk, n_chunks):
        """
        Args:
            list_to_chunk (array-like): this is the list to be broken up into n
            chunks
            n_chunks (int): this is the size of the chunk
        Returns:
            iterator - that will yield the chunks from the list_to_chunk
        """
        for i in range(0, len(list_to_chunk), n_chunks):
            yield list_to_chunk[i:i + n_chunks]

    def _create_windows(self):
        """
        """
        new_index = {}
        self.sub_crah['windows'] = \
            self.sub_crah.apply(lambda x: str(x.doy) + '_' + str(x.hour),
                                axis=1)
        windows = pd.unique(self.sub_crah['windows'])
        for i, sub_list in enumerate(self._chunks(windows, self.w)):
            for doy_hour in sub_list:
                new_index[doy_hour] = i
        self.sub_crah['new_index'] = \
            self.sub_crah['windows'].apply(lambda window: new_index[window])

    def fit(self, crah):
        """
        Args:
            None
        Returns:
            None, this is going to add new vectors: predictions, ubound, lbound
            for the information
        """
        if self.local:
            self.local_crah_data(crah)
        else:
            self.load_crah_data(crah)
        self.sub_crah = self.crah.dropna()
        self._create_windows()
        self.windows = pd.unique(self.sub_crah['new_index'])[:-1]
        if self.verbose:
            now = timeit.default_timer()
        self.pred_list, self.ubound_list, self.lbound_list = [], [], []
        self.w_cv_score, self.w_score, self.cosine_sim = [], [], []
        self.model_dict = {}
        mse_scorer = make_scorer(mean_squared_error)
        for window in self.windows:
            if self.model == 'lr':
                model = LinearRegression(n_jobs=-1)
            elif self.model == 'svr':
                model = SVR()
            train_data = \
                self.sub_crah.query('new_index == {}'.format(window))
            test_data = \
                self.sub_crah.query('new_index == {}'.format(window + 1))
            x_train, x_test = \
                train_data[['sat', 'fan_spd']], test_data[['sat', 'fan_spd']]
            y_train, y_test = \
                train_data['server_max'], test_data['server_max']
            if self.evaluation:
                cv_score = np.mean(cross_val_score(model, x_train, y_train,
                                                   cv=3, n_jobs=-1,
                                                   scoring=mse_scorer))
                self.w_cv_score.append(cv_score)
            model.fit(x_train, y_train)
            if self.evaluation:
                self.w_score.append(mse_scorer(estimator=model,
                                    X=x_test, y_true=y_test))
            self.model_dict[window] = model
            pred = model.predict(x_test)
            self.pred_list.extend(pred)
            std_error = np.sqrt(np.sum((y_test - pred) ** 2) / len(pred))
            self.ubound_list.extend(pred + std_error)
            self.lbound_list.extend(pred - std_error)
        diff = self.sub_crah.shape[0] - len(self.pred_list)
        self.y_true = self.sub_crah['server_max'].values[diff:]
        self.std_error = np.array(self.ubound_list) - np.array(self.pred_list)
        self.mse = mean_squared_error(self.y_true, self.pred_list)
        if self.evaluation:
            self.mse_cosine_sim = 1 - cosine(self.w_cv_score, self.w_score)
        if self.verbose:
            print 'model fitting took: ', timeit.default_timer() - now

    def plot_model(self, plot_type='show'):
        """
        Args:
            plot_type (str): this is the nature of the plot which is going to
            have the following possible outputs:
                 - 'show': will show the plot
                 - 'html': will return html as a string
                 - 'json': will return the plot as json
                 - 'mpld3': will show the plot in the browser
        Returns:
            plot_output (any of {str, dict}): returns an html string, None,
            or a, plot dict, depending on the requested output
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        diff = self.sub_crah.shape[0] - len(self.pred_list)
        ax.scatter(self.sub_crah.index[diff:],
                   self.sub_crah['server_max'].values[diff:], c='k', alpha=.1)
        ax.plot(self.sub_crah.index[diff:], self.ubound_list, c='y', alpha=.5)
        ax.plot(self.sub_crah.index[diff:], self.lbound_list, c='y', alpha=.5)
        ax.plot(self.sub_crah.index[diff:], self.pred_list, c='b',
                label='prediction using raw data')
        ax.hlines(y=self.threshold, xmin=self.sub_crah.index[diff:][0],
                  xmax=self.sub_crah.index[diff:][-1], colors='r',
                  linestyles='--',
                  label='temperature threshold')
        plt.legend(loc='best')
        if plot_type == 'show':
            plt.show()
        elif plot_type == 'html':
            plot_output = mpld3.fig_to_html(fig)
            return plot_output
        elif plot_type == 'json':
            plot_output = mpld3.fig_to_dict(fig)
            return plot_output
        elif plot_type == 'mpld3':
            mpld3.show()

    def predict(self, delta_sat):
        """
        Args:
            delta_sat (float): this is the change in supply air temperature
        Returns:
            None, produces the excursion count and delta_pred_list
            which shows the impact of delta_sat on the series
        """
        self.delta_sat = delta_sat
        if self.verbose:
            now = timeit.default_timer()
        self.delta_pred_list = []
        self.excursions = 0
        for window in self.windows:
            test_data = \
                self.sub_crah.query('new_index == {}'.format(window + 1))
            x_test = test_data[['sat', 'fan_spd']]
            model = self.model_dict[window]
            x_test['sat'] = x_test['sat'] + self.delta_sat
            delta_pred = model.predict(x_test)
            self.excursions += np.sum(delta_pred > self.threshold)
            self.delta_pred_list.extend(delta_pred)
        self.excursion_pct = float(self.excursions) / self.sub_crah.shape[0]
        if self.verbose:
            print 'prediction took: ', timeit.default_timer() - now

    def aimd(self):
        """
        Args:
            None
        Returns:
            None, executes AIMD on the time series using the historical and
            predicted values
        """
        # self.delta_sat_series = []
        # for window in self.windows:
        #     lr = self.model_dict[window]
        #     print window

    def plot_predictions(self, plot_type='show'):
        """
        Args:
            plot_type (str): this is the nature of the plot which is going to
            have the following possible outputs:
                 - 'show': will show the plot
                 - 'html': will return html as a string
                 - 'json': will return the plot as json
                 - 'mpld3': will show the plot in the browser
        Returns:
            plot_output (any of {str, dict}): returns an html string, None,
            or a, plot dict, depending on the requested output
        """
        total_eca = self.estimate_total_eca(self.delta_sat)
        fig = plt.figure()
        plt.xticks(rotation=70)
        ax = fig.add_subplot(111)
        diff = self.sub_crah.shape[0] - len(self.pred_list)
        title = 'Predicted Inlet Temp w/ Delta SAT = {}'
        subtitle = 'Cost Avoidance: CAD${}, Excursions: {}%'
        fig.suptitle(title.format(self.delta_sat))
        plt.title(subtitle.format(int(total_eca),
                                  np.round(100 * self.excursion_pct, 2)))
        ax.scatter(self.sub_crah.index[diff:],
                   self.sub_crah['server_max'].values[diff:], c='k', alpha=.1)
        ax.plot(self.sub_crah.index[diff:],
                self.delta_pred_list + self.std_error, c='y')
        ax.plot(self.sub_crah.index[diff:],
                self.delta_pred_list - self.std_error, c='y')
        ax.plot(self.sub_crah.index[diff:], self.delta_pred_list, c='b',
                label='predicted inlet temp')
        ax.hlines(y=self.threshold, xmin=self.sub_crah.index[diff:][0],
                  xmax=self.sub_crah.index[diff:][-1], colors='r',
                  linestyles='--',
                  label='temperature threshold')
        plt.legend(loc='best')
        if plot_type == 'show':
            plt.show()
        elif plot_type == 'html':
            plot_output = mpld3.fig_to_html(fig)
            return plot_output
        elif plot_type == 'json':
            plot_output = mpld3.fig_to_dict(fig)
            return plot_output
        elif plot_type == 'mpld3':
            mpld3.show()

    def plot_dashboard(self, opt_dash_df, delta_sat, plot_type='show'):
        """
        Args:
            opt_dash_df (dataframe): this is the optimization dash df
            delta_sat (float): this is the change in SAT
            plot_type (str): this is the nature of the plot which is going to
            have the following possible outputs:
                 - 'show': will show the plot
                 - 'html': will return html as a string
                 - 'json': will return the plot as json
                 - 'mpld3': will show the plot in the browser
        Returns:
            plot_output (any of {str, dict}): returns an html string, None,
            or a, plot dict, depending on the requested output
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        title = 'Quarterly Cost Avoidance '
        title += 'when Delta SAT = {} degC'.format(delta_sat)
        plt.title(title)
        opt_dash_df['cap'].plot.barh(ax=ax)
        plt.ylabel('CRAH Unit')
        plt.xlabel('Amount in CAD')
        if plot_type == 'show':
            plt.show()
        elif plot_type == 'html':
            plot_output = mpld3.fig_to_html(fig)
            return plot_output
        elif plot_type == 'json':
            plot_output = mpld3.fig_to_dict(fig)
            return plot_output
        elif plot_type == 'mpld3':
            mpld3.show()


if __name__ == "__main__":
    import os
    crah_list = filter(lambda val: 'DH' in val, os.listdir('data/'))
    crah_list = [val.replace('.csv', '') for val in crah_list]
    dsat_list = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, .5]
    for crah in crah_list:
        opt = CrahOptimizer(local=True, w=144)
        opt.fit(crah)
        excursions = []
        for delta_sat in dsat_list:
            opt.predict(delta_sat)
            eca = np.round(opt.estimate_total_eca(delta_sat), 2)
            excursion = str(np.round(opt.excursion_pct, 3))[:6]
            excursions.append(excursion)
        print crah, excursions
        excursions.append(excursion)
