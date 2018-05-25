# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns


class OperationalIssues(object):
    """
    The objective of this class is to, on a time series, determine the periods
    that violate a best practice and mark them in a plot in order to make them
    analytically visible

    Fundamentally, this module receives a dataframe and then returns a plot
    """

    def __init__(self, dataframe):
        """
        Args:
            dataframe (df): this is the dataframe on which the rules
            will be analyzed and marked as violated or cleared where the index
            is the timestamp in the date_time column
        """
        # at init
        self.df = dataframe
        # constants
        self.min_per_period = 5.0
        # High Zone Temp
        self.hzt_limit = 30
        self.hzt_min_duration = self._seconds_to_periods(30 * 60)
        # Low Zone Temp
        self.lzt_limit = 24
        self.lzt_operator = '<'
        self.lzt_min_duration = self._seconds_to_periods(0)
        # Excess Load Discharge
        self.eld_fan_limit = 48
        self.eld_temp_limit = 25
        self.eld_min_duration = self._seconds_to_periods(120 * 60)
        # OAT Sensor Verification
        self.oat_tolerance = 1
        self.oat_min_duration = self._seconds_to_periods(60 * 60)
        # Cooling Valve Leak
        self.clg_vlv_min = 0
        self.cvl_limit = 21
        self.cvl_min_duration = self._seconds_to_periods(30 * 60)
        # Cooling Not Attained
        self.cna_tolerance = 1.5
        self.cna_sat_setpoint = 21.0
        self.clg_vlv_max = 95
        self.cna_min_duration = self._seconds_to_periods(60 * 60)
        # Cooling Valve Cycling
        self.cvc_max_position = 50
        self.cvc_min_position = 10
        self.cvc_min_fault = self._seconds_to_periods(15 * 60)
        self.cvc_min_incidents = 8
        self.cvc_min_duration = self._seconds_to_periods(24 * 60 * 60)
        # Underfloor Pressure Not Attained
        self.pna_tolerance = .5
        self.pna_dp_setpoint = 7.5
        self.pna_min_duration = self._seconds_to_periods(60 * 60)
        # Dashboard Priority Scores
        self.ps = {
            "hzt_window_flag": 3,
            "lzt_window_flag": 2,
            "eld_window_flag": 2,
            "oat_window_flag": 2,
            "cvl_window_flag": 2,
            "cna_window_flag": 3,
            "cvc_window_flag": 1,
            "pna_window_flag": 3
        }
        self.col_rule = {
            "hzt_window_flag": "High Zone Temp",
            "lzt_window_flag": "Low Zone Temp",
            "eld_window_flag": "Excess Load Discharge",
            "oat_window_flag": "Outside Air Temperature",
            "cvl_window_flag": "Cooling Valve Leak",
            "cna_window_flag": "Cooling Not Attained",
            "cvc_window_flag": "Cooling Valve Cycling",
            "pna_window_flag": "Pressure Not Attained"
        }
        self.image_d = (1587, 638)
        self.energy_unit_cost = 0.1628 / (60 / self.min_per_period)
        self.cooling_cost = 4.73
        self.opt_fan_spd = 45
        self.suggested_actions = {
            "hzt": ["1. Check CHDCCU control strategy/setpoint",
                    "2. Check CHDCCU chilled water valve",
                    "3. Check Chilled water supply temperature",
                    "4. Check operation of associated racks"],
            "lzt": ["1. Check CHDCCU control strategy/setpoint",
                    "2. Check CHDCCU chilled water valve",
                    "3. Check Chilled water supply temperature",
                    "4. Check operation of associated racks"],
            "eld": ["1. Check CHDCCU control strategy / setpoint",
                    "2. Check static pressure sensor(s)"],
            "oat": ["1. Check Outside air temperature sensor"],
            "cvl": ["1. Check CHDCCU discharge air temperature sensors",
                    "2. Check CHDCCU chilled water valve"],
            "cna": ["1. Check CHDCCU control strategy",
                    "2. Check CHDCCU chilled water valve",
                    "3. Check Chilled water supply temperature",
                    "4. Check operation of associated RACKS"],
            "cvc": ["1. Check CHDCCU control strategy",
                    "2. Check CHDCCU chilled water valve"],
            "pna": ["1. Check CHDCCU control strategy / speed setpoint"
                    "2. Check static pressure sensor(s)"]
        }

    def _seconds_to_periods(self, seconds):
        """
        converts number of seconds for a time filter into a number of periods
        Args:
            seconds (int): this is the number of seconds that the time filter
            needs to be checked for
        Returns:
            periods (int): this is the number of periods that the time filter
            needs to be checked for
        """
        # currently hardcoded at 5.0 since our time periods are split by
        periods_w_decimal = seconds / 60.0 / self.min_per_period
        periods = int(seconds / 60.0 / self.min_per_period)
        if periods_w_decimal - periods > 0:
            return periods + 1
        if periods == 0:
            return 1
        return periods

    def _hzt_rule(self, stat_tuple):
        """
        Implements high zone temp rule on individual data point
        Args:
            stat_tuple (tuple): this is a tuple where the content are:
                rat (float): return air temperature in degC
                fan_spd (int): this is the percent of max capacity the fan
                is blowing at
        Returns:
            flag (int): 1 if the observation violates the high zone temp
            violation rule, and 0 if it does not
        """
        rat, fan_spd = stat_tuple
        flag = np.nan
        if rat is not np.nan and fan_spd is not np.nan:
            if fan_spd > 0 and rat > self.hzt_limit:
                flag = 1
            else:
                flag = 0
        return flag

    def _high_zone_temp(self):
        """
        Implements high zone temp rule on dataframe
        Args:
            None
        Returns:
            None
        """
        print 'HZT LIMIT IS: ', self.hzt_limit
        w = self.hzt_min_duration
        self.df['target'] = \
            list(zip(self.df['rat_interp'], self.df['fan_spd_interp']))
        self.df['hzt_flag'] = \
            self.df['target'].apply(lambda target: self._hzt_rule(target))
        self.df['hzt_window'] = self.df['hzt_flag'].rolling(w).mean()
        self.df['hzt_window_flag'] = \
            self.df['hzt_window'].apply(lambda window: 1 if window == 1
                                        else np.nan)
        self.df['high_temp_violation'] = \
            self.df['rat_interp'] * self.df['hzt_window_flag']

    def _lzt_rule(self, stat_tuple):
        """
        Implements low zone temp rule on individual data point
        Args:
            stat_tuple (tuple): this is a tuple where the content are:
                rat (float): return air temperature in degC
                fan_spd (int): this is the percent of max capacity the fan
                is blowing at
        Returns:
            flag (int): 1 if the observation violates the low zone temp
            violation rule, and 0 if it does not
        """
        rat, fan_spd = stat_tuple
        flag = np.nan
        if rat is not np.nan and fan_spd is not np.nan:
            if fan_spd > 0 and rat < self.lzt_limit:
                flag = 1
            else:
                flag = 0
        return flag

    def _low_zone_temp(self):
        """
        Implements low zone temp rule on dataframe
        Args:
            None
        Returns:
            None
        """
        w = self.lzt_min_duration
        self.df['target'] = \
            list(zip(self.df['rat_interp'], self.df['fan_spd_interp']))
        self.df['lzt_flag'] = \
            self.df['target'].apply(lambda target: self._lzt_rule(target))
        self.df['lzt_window'] = self.df['lzt_flag'].rolling(w).mean()
        self.df['lzt_window_flag'] = \
            self.df['lzt_window'].apply(lambda window: 1 if window == 1
                                        else np.nan)
        self.df['low_temp_violation'] = \
            self.df['rat_interp'] * self.df['lzt_window_flag']

    def crah_rat_plot(self, plot_type='show'):
        """
        Description:
            Executes high zone temp and low zone temp rule violation checks
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
        self._high_zone_temp()
        self._low_zone_temp()
        total_lzt = np.sum(self.df['lzt_window_flag'])
        total_hzt = np.sum(self.df['hzt_window_flag'])
        if total_lzt > 0 or total_hzt > 0:
            self.rat_action = self.suggested_actions['hzt']
        else:
            self.rat_action = [None]
        date_range = pd.Series(self.df.index)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle('CRAH Return Temperature Issues Analysis')
        ax.plot(date_range, self.df['rat_interp'],
                label='Return Air Temp', c='k')
        ax.plot(date_range, self.df['high_temp_violation'],
                label='High Zone Temp', c='r')
        ax.plot(date_range, self.df['low_temp_violation'],
                label='Low Zone Temp', c='b')
        ax.hlines(y=self.hzt_limit, xmin=date_range.iloc[0],
                  xmax=date_range.iloc[date_range.shape[0] - 1], colors='y',
                  linestyle='--', label='Temp Threshold')
        plt.xlabel('Date/Time @ 5 minute intervals')
        plt.ylabel('Temperature in degC')
        ax.legend(loc='best')
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

    def _eld_rule(self, stat_tuple):
        """
        Implements excess load discharge rule on individual data point
        Args:
            stat_tuple (tuple): this is a tuple where the content are:
                rat (float): return air temperature in degC
                fan_spd (int): this is the percent of max capacity the fan
                is blowing at
        Returns:
            flag (int): 1 if the observation violates the excess load discharge
            violation rule, and 0 if it does not
        """
        rat, fan_spd = stat_tuple
        flag = np.nan
        if rat is not np.nan and fan_spd is not np.nan:
            if fan_spd > self.eld_fan_limit and rat < self.eld_temp_limit:
                flag = 1
            else:
                flag = 0
        return flag

    def _excess_load_discharge(self):
        """
        Implements excess load discharge rule on dataframe
        Args:
            None
        Returns:
            None
        """
        w = self.eld_min_duration
        self.df['target'] = \
            list(zip(self.df['rat_interp'], self.df['fan_spd_interp']))
        self.df['eld_flag'] = \
            self.df['target'].apply(lambda target: self._eld_rule(target))
        self.df['eld_window'] = self.df['eld_flag'].rolling(w).mean()
        self.df['eld_window_flag'] = \
            self.df['eld_window'].apply(lambda window: 1 if window == 1
                                        else np.nan)
        self.df['high_fan_violation'] = \
            self.df['rat_interp'] * self.df['eld_window_flag']

    def fan_speed_plot(self, plot_type='show'):
        """
        Description:
            Executes excess load discharge checks
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
        self._excess_load_discharge()
        if np.sum(self.df['eld_window_flag']) > 0:
            self.eld_action = self.suggested_actions['eld']
        else:
            self.eld_action = None
        date_range = pd.Series(self.df.index)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle('CRAH Supply Fan Issues Analysis')
        ax.plot(date_range, self.df['fan_spd_interp'],
                label='Fan Speed', c='k')
        ax.plot(date_range, self.df['high_fan_violation'],
                label='Excess Load Discharge', c='r')
        ax.hlines(y=self.eld_fan_limit, xmin=date_range.iloc[0],
                  xmax=date_range.iloc[date_range.shape[0] - 1], colors='y',
                  linestyle='--', label='Fan Speed Max Limit')
        plt.xlabel('Date/Time @ 5 minute intervals')
        plt.ylabel('Fan Speed in Output %')
        ax.legend(loc='best')
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

    def _oat_rule(self, stat_tuple):
        """
        Implements excess load discharge rule on individual data point
        Args:
            stat_tuple (tuple): this is a tuple where the content are:
                oat_sensor (float): outside air temperature in degC from
                data center oat sensor
                twc_data (int): outside air temperature in degC from
                the weather company data
        Returns:
            flag (int): 1 if the observation violates the outside air temp
            violation rule, and 0 if it does not
        """
        oat_sensor, twc_data = stat_tuple
        flag = np.nan
        if oat_sensor is not np.nan and twc_data is not np.nan:
            if np.abs(oat_sensor - twc_data) > self.oat_tolerance:
                flag = 1
            else:
                flag = 0
        return flag

    def _oat_sensor_verification(self):
        """
        Implements excess load discharge rule on dataframe
        Args:
            None
        Returns:
            None
        """
        w = self.oat_min_duration
        self.df['temp_interp'] = (self.df['temp_interp'] - 32) * 5 / 9.
        self.df['target'] = \
            list(zip(self.df['oa_t_interp'], self.df['temp_interp']))
        self.df['oat_flag'] = \
            self.df['target'].apply(lambda target: self._oat_rule(target))
        self.df['oat_window'] = self.df['oat_flag'].rolling(w).mean()
        self.df['oat_window_flag'] = \
            self.df['oat_window'].apply(lambda window: 1 if window == 1
                                        else np.nan)
        self.df['high_difference_violation'] = \
            self.df['oa_t_interp'] * self.df['oat_window_flag']

    def oat_sensor_plot(self, plot_type='show'):
        """
        Description:
            Triggers if the absolute difference between the individual CHDCCU
            Outside Air Temperature  and TWC Outside Air Temperature is greater
            than the Temperature Sensor Tolerance for longer than the Minimum
            Fault Duration Filter.
        Args:
            tolerance (float): this is the number of temperature units that
            the OAT sensor is different from the twc readings for temp
            min_duration (int): this is the number of seconds that the oat
            sensor is different from twc readings for such that there would be
            a flag
        """
        self._oat_sensor_verification()
        if np.sum(self.df['oat_window_flag']) > 0:
            self.oat_action = self.suggested_actions['oat']
        else:
            self.oat_action = [None]
        date_range = pd.Series(self.df.index)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle('Outside Air Temperature Sensor Calibration Issues')
        ax.plot(date_range, self.df['oa_t_interp'],
                label='OAT Sensor', c='k', alpha=.75)
        ax.plot(date_range, self.df['temp_interp'],
                label='The Weather Company Data', c='b', alpha=.75)
        tolerance_label = 'High Difference Detected'
        tolerance_label += '(> {} degC)'.format(str(self.oat_tolerance))
        ax.plot(date_range, self.df['high_difference_violation'],
                label=tolerance_label, c='r')
        plt.xlabel('Date/Time @ 5 minute intervals')
        plt.ylabel('Outside Air Temperature in degC')
        ax.legend(loc='best')
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

    def _cvl_rule(self, stat_tuple):
        """
        Implements cooling valve leaking rule on individual data point
        Args:
            stat_tuple (tuple): this is a tuple where the content are:
                sat (float): supply air temperature in degC
                fan_spd (int): this is the percent of max capacity the fan
                is blowing at
                clg_vlv (float): this is the current setting of the cooling
                valve
        Returns:
            flag (int): 1 if the observation violates the cooling valve leak
            violation rule, and 0 if it does not
        """
        sat, fan_spd, clg_vlv = stat_tuple
        flag = np.nan
        if sat is not np.nan:
            if fan_spd is not np.nan and clg_vlv is not np.nan:
                if clg_vlv == self.clg_vlv_min and sat < self.cvl_limit:
                    flag = 1
                else:
                    flag = 0
        return flag

    def _cooling_valve_leaking(self):
        """
        Implements cooling valve leaking rule on dataframe
        Args:
            None
        Returns:
            None
        """
        w = self.cvl_min_duration
        self.df['target'] = list(zip(self.df['sat_interp'],
                                     self.df['fan_spd_interp'],
                                     self.df['clg_vlv_interp']))
        self.df['cvl_flag'] = \
            self.df['target'].apply(lambda target: self._cvl_rule(target))
        self.df['cvl_window'] = self.df['cvl_flag'].rolling(w).mean()
        self.df['cvl_window_flag'] = \
            self.df['cvl_window'].apply(lambda window: 1 if window == 1
                                        else np.nan)
        self.df['cooling_valve_leak_alert'] = \
            self.df['clg_vlv_interp'] * self.df['cvl_window_flag']

    def _cvc_rule(self, stat_tuple):
        """
        Implements cooling valve leaking rule on individual data point
        Args:
            stat_tuple (tuple): this is a tuple where the content are:
                rat (float): return air temperature in degC
                fan_spd (int): this is the percent of max capacity the fan
                is blowing at
        Returns:
            flag (int): 1 if the observation violates the cooling valve cycling
            violation rule, and 0 if it does not
        """
        rat, fan_spd = stat_tuple
        flag = np.nan
        if rat is not np.nan and fan_spd is not np.nan:
            if fan_spd > self.cvl_fan_limit and rat < self.cvl_temp_limit:
                flag = 1
            else:
                flag = 0
        return flag

    def _cooling_valve_cycle(self):
        """
        Implements cooling valve cycling rule on dataframe
        Args:
            None
        Returns:
            None
        """
        w = self.cvc_min_duration
        self.df['target'] = \
            list(zip(self.df['oa_t_interp'], self.df['temp_interp']))
        self.df['cvc_flag'] = \
            self.df['target'].apply(lambda target: self._cvc_rule(target))
        self.df['cvc_window'] = self.df['cvc_flag'].rolling(w).mean()
        self.df['cvc_window_flag'] = \
            self.df['cvc_window'].apply(lambda window: 1 if window == 1
                                        else np.nan)
        self.df['high_difference_violation'] = \
            self.df['oa_t_interp'] * self.df['cvc_window_flag']

    def cooling_valve_plot(self, plot_type='show'):
        """
        Description:
            Executes cooling valve leaking and cooling valve cycling checks
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
        self._cooling_valve_leaking()
        # self._cooling_valve_cycle()
        if np.sum(self.df['cvl_window_flag']) > 0:
            self.cvl_action = self.suggested_actions['cvl']
        else:
            self.cvl_action = [None]
        date_range = pd.Series(self.df.index)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle('CRAH Cooling Value Operational Issues')
        ax.plot(date_range, self.df['clg_vlv_interp'],
                label='Cooling Valve Position', c='k')
        ax.plot(date_range, self.df['cooling_valve_leak_alert'],
                label='Cooling Valve Leak Alert', c='b')
        plt.xlabel('Date/Time @ 5 minute intervals')
        plt.ylabel('Cooling Valve Position in %')
        ax.legend(loc='best')
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

    def _cna_rule(self, stat_tuple):
        """
        Implements cooling not attained rule on individual data point
        Args:
            stat_tuple (tuple): this is a tuple where the content are:
                clg_vlv (float): this is the cooling valve position
                sat (float): this is supply air temp in degC
        Returns:
            flag (int): 1 if the observation violates the cooling not attained
            violation rule, and 0 if it does not
        """
        clg_vlv, sat = stat_tuple
        flag = np.nan
        if clg_vlv is not np.nan and sat is not np.nan:
            if clg_vlv > self.clg_vlv_max:
                if np.abs(sat - self.cna_sat_setpoint) > self.cna_tolerance:
                    flag = 1
            else:
                flag = 0
        return flag

    def _cooling_not_attained(self):
        """
        Implements cooling not attained rule on dataframe
        Args:
            None
        Returns:
            None
        """
        w = self.cna_min_duration
        self.df['target'] = \
            list(zip(self.df['clg_vlv_interp'], self.df['sat_interp']))
        self.df['cna_flag'] = \
            self.df['target'].apply(lambda target: self._cna_rule(target))
        self.df['cna_window'] = self.df['cna_flag'].rolling(w).mean()
        self.df['cna_window_flag'] = \
            self.df['cna_window'].apply(lambda window: 1 if window == 1
                                        else np.nan)
        self.df['cooling_not_attained'] = \
            self.df['sat_interp'] * self.df['cna_window_flag']

    def crah_sat_plot(self, plot_type='show'):
        """
        Description:
            Executes cooling not attained rule violation checks
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
        self._cooling_not_attained()
        if np.sum(self.df['cna_window_flag']) > 0:
            self.cna_action = self.suggested_actions['cna']
        else:
            self.cna_action = [None]
        date_range = pd.Series(self.df.index)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle('CRAH Supply Temperature Issues Analysis')
        ax.plot(date_range, self.df['sat_interp'],
                label='Supply Air Temp', c='k')
        ax.plot(date_range, self.df['cooling_not_attained'],
                label='Target Temp Not Attained', c='b')
        ax.hlines(y=self.cna_sat_setpoint, xmin=date_range.iloc[0],
                  xmax=date_range.iloc[date_range.shape[0] - 1], colors='y',
                  linestyle='--', label='SAT Set Point')
        plt.xlabel('Date/Time @ 5 minute intervals')
        plt.ylabel('Temperature in degC')
        ax.legend(loc='best')
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

    def _pna_rule(self, stat_tuple):
        """
        Implements pressure not attained rule on individual data point
        Args:
            stat_tuple (tuple): this is a tuple where the content are:
                adp (float): this is the average underfloor pressure
        Returns:
            flag (int): 1 if the observation violates the pressure not attained
            violation rule, and 0 if it does not
        """
        adp = stat_tuple
        flag = np.nan
        if adp is not np.nan:
            if np.abs(adp - self.pna_dp_setpoint) > self.pna_tolerance:
                flag = 1
            else:
                flag = 0
        return flag

    def _pressure_not_attained(self):
        """
        Implements cooling not attained rule on dataframe
        Args:
            None
        Returns:
            None
        """
        # prepare the dataframe
        columns = filter(lambda colname: '_interp' in colname, self.df.columns)
        self.df = self.df[columns]
        self.df['mean_dp'] = self.df.T.mean()
        array = self.df['mean_dp'].values
        w = self.pna_min_duration
        self.df['pna_flag'] = \
            np.apply_along_axis(self._pna_rule, 1, array.reshape(-1, 1))
        self.df['pna_window'] = self.df['pna_flag'].rolling(w).mean()
        self.df['pna_window_flag'] = \
            self.df['pna_window'].apply(lambda window: 1 if window == 1
                                        else np.nan)
        self.df['pressure_not_attained'] = \
            self.df['mean_dp'] * self.df['pna_window_flag']

    def underfloor_pressure_plot(self, plot_type='show'):
        """
        Description:
            Executes pressure not attained rule violation checks
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
        self._pressure_not_attained()
        if np.sum(self.df['pna_window_flag']) > 0:
            self.pna_action = self.suggested_actions['pna']
        else:
            self.pna_action = [None]
        date_range = pd.Series(self.df.index)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle('Underfloor Pressure Issues Analysis')
        ax.plot(date_range, self.df['mean_dp'],
                label='Differential Pressure', c='k')
        ax.plot(date_range, self.df['pressure_not_attained'],
                label='Pressure Not Attained', c='r')
        ax.hlines(y=self.pna_dp_setpoint, xmin=date_range.iloc[0],
                  xmax=date_range.iloc[date_range.shape[0] - 1], colors='y',
                  linestyle='--', label='DP Set Point')
        plt.xlabel('Date/Time @ 5 minute intervals')
        plt.ylabel('Pressure in Pascals')
        ax.legend(loc='best')
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

    def crah_dashboard(self):
        """
        This function applies all CRAH rules to produce the dashboard
        Args:
            None
        Returns:
            None
        """
        self._cooling_not_attained()
        self._cooling_valve_leaking()
        # self._cooling_valve_cycle()
        self._excess_load_discharge()
        self._high_zone_temp()
        self._low_zone_temp()
        flag_cols = filter(lambda colname: 'window_flag' in colname,
                           self.df.columns)
        self.dashboard = pd.DataFrame(self.df[flag_cols].sum())
        self.dashboard = self.dashboard.reset_index()
        self.dashboard.columns = ['rule', 'periods']
        self.dashboard['p_score'] = self.dashboard['periods'] \
            * self.dashboard['rule'].apply(lambda rule: self.ps[rule])
        self.dashboard.set_index('rule', inplace=True)

    def pressure_dashboard(self):
        """
        This function applies all CRAH rules to produce the dashboard
        Args:
            None
        Returns:
            None
        """
        self._pressure_not_attained()
        flag_cols = filter(lambda colname: 'window_flag' in colname,
                           self.df.columns)
        self.dashboard = pd.DataFrame(self.df[flag_cols].sum())
        self.dashboard = self.dashboard.reset_index()
        self.dashboard.columns = ['rule', 'periods']
        self.dashboard['p_score'] = self.dashboard['periods'] \
            * self.dashboard['rule'].apply(lambda rule: self.ps[rule])
        self.dashboard.set_index('rule', inplace=True)

    def oat_dashboard(self):
        """
        This function applies all CRAH rules to produce the dashboard
        Args:
            None
        Returns:
            None
        """
        self._oat_sensor_verification()
        flag_cols = filter(lambda colname: 'window_flag' in colname,
                           self.df.columns)
        self.dashboard = pd.DataFrame(self.df[flag_cols].sum())
        self.dashboard = self.dashboard.reset_index()
        self.dashboard.columns = ['rule', 'periods']
        self.dashboard['p_score'] = self.dashboard['periods'] \
            * self.dashboard['rule'].apply(lambda rule: self.ps[rule])
        self.dashboard.set_index('rule', inplace=True)

    def dashboard_plot(self, plot_type='show'):
        """
        Description:
            creates the dashboard for the different issue frequencies
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
        plot_df = self.df.T.copy()
        plot_df['total'] = self.df.sum()
        plot_df = plot_df.sort_values(by='total', ascending=False)
        plot_df.drop('total', axis=1, inplace=True)
        plot_df.columns = [self.col_rule[column] for column in plot_df.columns]
        plot_df = plot_df[['High Zone Temp', 'Low Zone Temp',
                           'Excess Load Discharge', 'Cooling Valve Cycling',
                           'Pressure Not Attained', 'Low Zone Temp',
                           'Cooling Not Attained', 'Outside Air Temperature']]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle('Infrastructure Health Dashboard')
        colors = sns.color_palette("Set1", 8)
        plot_df.plot.barh(stacked=True, colors=colors, ax=ax, legend=False)
        plt.xlabel('Priority Score = Priority Value (1-3) X Occurrence Hours')
        plt.ylabel('Equipment Name')
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

    def heatmap_plot(self, plot_type='show'):
        """
        Description:
            creates the current or forecasted heatmaps
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
        w, h = self.image_d
        cold = self.df[self.df['temp'] < 22.2222]
        hot = self.df[self.df['temp'] > 27]
        norm = self.df[self.df['temp'] <= 27][self.df['temp'] >= 22.2222]
        size, lsize, alpha = 200, 20000, .1
        fig, ax = plt.subplots(figsize=(30, 10))
        ax.axis('off')
        # ax.imshow(flip[:, :, 1], cmap="gray", interpolation="nearest")
        ax.scatter(norm['x'], norm['y'], c='g', s=lsize, alpha=alpha)
        ax.scatter(hot['x'], hot['y'], c='r', s=lsize, alpha=alpha * 2)
        ax.scatter(self.df['x'], self.df['y'], c='k', label='sensor', s=size)
        ax.scatter(hot['x'], hot['y'], c='r', label='temp > 27', s=size)
        ax.scatter(cold['x'], cold['y'], c='b',
                   label='temp < 22.2222', s=size)
        ax.scatter(norm['x'], norm['y'], c='g',
                   label='22.22 =< temp =< 27', s=size)
        # plt.legend(loc='best')
        plt.xlim(xmin=0, xmax=w)
        plt.ylim(ymin=0, ymax=h)
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
