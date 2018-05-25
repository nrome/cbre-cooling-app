# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import timeit
from dashdb_connect import DashdbConnect


class FanOptimization(object):
    """
    This class is in charge of loading and computing the fan optimization
    savings that can be achieved by manipulating fan speed for this datacenter
    """
    def __init__(self, desired_opf=1.5, verbose=False):
        self.hours_in_csv = 2544
        self.max_rated_cooling = 2917.608
        self.desired_opf = desired_opf
        self.crah_power_at_50pct = 1.5
        self.op_fan_spd = 0.5
        self.time_increment = 5.0
        self.total_energy_saved = 0
        self.cost_energy = 0.1628
        self.lb_temp_slope = 8
        self.ub_temp_slope = 10
        self.verbose = verbose
        self.inlet_threshold = 27

    def load_data(self):
        """ Loads data from the db
        """
        if self.verbose:
            start = timeit.default_timer()
        self.db = DashdbConnect()
        util_query = \
            '''
            SELECT * FROM UTIL_TABLE ORDER BY "date_time" ASC;
            '''
        inlet_query = \
            '''
            SELECT * FROM INLET_TABLE ORDER BY "date_time" ASC;
            '''
        self.util_df = self.db.query_df(util_query)
        self.inlet_df = self.db.query_df(inlet_query)
        self.crah_ids = list(self.util_df.columns[:18])
        if self.verbose:
            print "loading data took: ", timeit.default_timer() - start

    def _run_computations(self, timestamp):
        """
        Args:
            timestamp (datetime): this is the timestamp on which these
            comptuations are going to be run
        Returns:
            power_saved (dict): this dict has the keys as the crah units
            and the values as the power saved for these crah
            lb_temp (dict): this dict has the keys as the crah units
            and the values as the lower bound temp for these crahs
            ub_temp (dict): this dict has the keys as the crah units
            and the values as the upper bound temp for these crahs
        """
        self.util_slice = self.util_df[self.crah_ids].loc[timestamp]
        self.inlet_slice = self.inlet_df.loc[timestamp]
        # crahIds = self.util_slice.sort_values().index
        # meanHourlyInletTemp = self.inlet_slice.mean()
        sorted_util_slice = self.util_slice.sort_values()
        current_opf = self.max_rated_cooling / self.util_slice.sum()
        diff_opf = current_opf - self.desired_opf
        capacity_to_reduce = self.util_slice.sum() * diff_opf
        # crahIds = []
        # capacity_saved, power_saved = 0, 0
        power_saved = 0
        lb_temp = self.inlet_slice + \
            self.lb_temp_slope * \
            (1 - self.min_fan_speed - self.op_fan_spd)
        ub_temp = self.inlet_slice + \
            self.ub_temp_slope * \
            (1 - self.min_fan_speed - self.op_fan_spd)
        if self.min_fan_speed == 0.0:
            machines_to_turn_off = \
                np.cumsum(sorted_util_slice) < capacity_to_reduce
            # capacity_reduced = sorted_util_slice[machines_to_turn_off]
            # capacity_saved = capacity_reduced.sum()
            # power_saved = capacity_reduced.shape[0] * self.savings
            power_saved = (machines_to_turn_off * self.savings).to_dict()
            # crahIds = list(capacity_reduced.index)
        else:
            # capacity_saved = capacity_to_reduce
            power_saved = sorted_util_slice.shape[0] * self.savings
            savings_vect = np.ones(len(sorted_util_slice.index)) * self.savings
            power_saved = dict(zip(sorted_util_slice.index,
                                   savings_vect))
        return power_saved, lb_temp.to_dict(), ub_temp.to_dict()

    def compute_savings(self, min_fan_speed=0.0):
        """ Runs savings computations for all units
        Args:
            None
        Returns:
            None, creates the power_df, lb_df, and ub_df attributes
        """
        self.min_fan_speed = min_fan_speed
        self.savings = (self.crah_power_at_50pct -
                        (self.min_fan_speed / self.op_fan_spd) ** 3 *
                        self.crah_power_at_50pct)
        power_df, lb_df, ub_df = [], [], []
        if self.verbose:
            start = timeit.default_timer()
        for timestamp in self.util_df.index:
            power_saved, lb_temp, ub_temp = self._run_computations(timestamp)
            power_df.append(power_saved)
            lb_df.append(lb_temp)
            ub_df.append(ub_temp)
        if self.verbose:
            print "computations took: ", timeit.default_timer() - start
        power_df = pd.DataFrame(power_df) * self.cost_energy
        power_df['date_time'] = self.util_df.index
        power_df.set_index('date_time', inplace=True)
        lb_df = pd.DataFrame(lb_df)
        lb_df['date_time'] = self.util_df.index
        lb_df.set_index('date_time', inplace=True)
        ub_df = pd.DataFrame(ub_df)
        ub_df['date_time'] = self.util_df.index
        ub_df.set_index('date_time', inplace=True)
        power_df.columns = ['DH{}'.format(num) for num in power_df.columns]
        lb_df.columns = ['DH{}'.format(num) for num in lb_df.columns]
        ub_df.columns = ['DH{}'.format(num) for num in ub_df.columns]
        self.power_df = power_df
        self.lb_df = lb_df
        self.ub_df = ub_df


if __name__ == "__main__":
    fanopt = FanOptimization(verbose=True)
    fanopt.load_data()
    fanopt.compute_savings(min_fan_speed=.2)
