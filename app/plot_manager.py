# -*- coding: utf-8 -*-

from dashdb_connect import DashdbConnect
import pandas as pd
from operational_issues import OperationalIssues
from crah_optimizer import CrahOptimizer
from anomaly_detection import TSAnomalyDetection
from fan_optimization import FanOptimization
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mpld3
import copy


class PlotManager(object):
    """
    This class is going to be in charge of returning the appropriate
    plot responses for each of the different API methods
    """
    def __init__(self):
        self.crah_query = \
            '''
            SELECT *
                FROM {crah}
            WHERE "date_time" BETWEEN '{start}' AND '{end}'
            ORDER BY "date_time" ASC;
            '''
        self.oat_query = \
            '''
            SELECT l."oa_t_interp", r."temp_interp", l."date_time"
                FROM OA_HOUR AS l LEFT JOIN TWC_HOUR AS r
                ON l."date_time" = r."date_time"
            WHERE l."date_time" BETWEEN '{start}' AND '{end}'
            ORDER BY l."date_time" ASC;
            '''
        self.dp_query = \
            '''
            SELECT *
                FROM DH_STATIC
            WHERE "date_time" BETWEEN '{start}' AND '{end}'
            ORDER BY "date_time" ASC;
            '''
        self.server_query = \
            '''
            SELECT * FROM SERVER_INLET WHERE "date_time" = '{start}';
            '''
        self.inlet_dim_query = \
            '''
            SELECT * FROM SERVER_INLET_DIM
            WHERE "io" != 'TD'
            AND "x_pos" >= 20;
            '''
        self.all_crah_metric = \
            '''
            SELECT
                l."date_time", l."{metric}" as DH1, r."{metric}" as DH2,
                g."{metric}" as DH4, w."{metric}" as DH5,
                p."{metric}" as DH9, d."{metric}" as DH10,
                q."{metric}" as DH11, k."{metric}" as DH14,
                j."{metric}" as DH15, u."{metric}" as DH17,
                y."{metric}" as DH18, z."{metric}" as DH21,
                n."{metric}" as DH23, b."{metric}" as DH24,
                v."{metric}" as DH26
            FROM
                DH1 as l LEFT JOIN DH2 as r on l."date_time" = r."date_time"
                LEFT JOIN DH4 as g on l."date_time" = g."date_time"
                LEFT JOIN DH5 as w on l."date_time" = w."date_time"
                LEFT JOIN DH9 as p on l."date_time" = p."date_time"
                LEFT JOIN DH10 as d on l."date_time" = d."date_time"
                LEFT JOIN DH11 as q on l."date_time" = q."date_time"
                LEFT JOIN DH14 as k on l."date_time" = k."date_time"
                LEFT JOIN DH15 as j on l."date_time" = j."date_time"
                LEFT JOIN DH17 as u on l."date_time" = u."date_time"
                LEFT JOIN DH18 as y on l."date_time" = y."date_time"
                LEFT JOIN DH21 as z on l."date_time" = z."date_time"
                LEFT JOIN DH23 as n on l."date_time" = n."date_time"
                LEFT JOIN DH24 as b on l."date_time" = b."date_time"
                LEFT JOIN DH26 as v on l."date_time" = v."date_time"
            WHERE l."date_time" BETWEEN '{start}' AND '{end}'
            ORDER BY l."date_time" ASC;
            '''
        self.crah = {
            "DH1", "DH2", "DH4", "DH5", "DH9", "DH10", "DH11", "DH14", "DH15",
            "DH17", "DH18", "DH21", "DH23", "DH24", "DH26"
        }
        self.equipments = [
            "DH1", "DH2", "DH4", "DH5", "DH9", "DH10", "DH11", "DH14", "DH15",
            "DH17", "DH18", "DH21", "DH23", "DH24", "DH26", "DP", "OAT"
        ]
        self.rules = [
            "hzt_window_flag", "lzt_window_flag", "eld_window_flag",
            "oat_window_flag", "cvl_window_flag", "cna_window_flag",
            "cvc_window_flag", "pna_window_flag"
        ]
        self.db = DashdbConnect()
        self.image_d = (1587, 638)
        self.metric_codes = {
            "SAT": "sat_interp",
            "RAT": "rat_interp",
            "FANSPD": "fan_spd_interp",
            "CLGVLV": "clg_vlv_interp"
        }
        self.optimal_delta_sat = {
            "DH10": 3,
            "DH11": 0,
            "DH14": 4,
            "DH18": .5,
            "DH20": 0,
            "DH21": 0,
            "DH23": 2.5,
            "DH24": 4,
            "DH26": 0,
            "DH4": 1,
            "DH5": 0,
            "DH7": 3,
            "DH9": 0,
            "DH1": 0.5,
            "DH2": 1,
            "DH15": 4,
            "DH17": .5,
            "DH13": 0
        }
        self.fanopt = FanOptimization(verbose=True)
        self.fanopt.load_data()
        self.fan_df = pd.read_csv('data/simulation_results.csv')
        self.fan_df.set_index('crah', inplace=True)
        self.fan_df = self.fan_df.loc[self.optimal_delta_sat.keys()]

    def _crah_equipment(self):
        """ Runs the crah equipment plots
        Args:
            None
        Returns:
            None
        """
        oid = OperationalIssues(self.df)
        plots = []
        plots.append({
            "plot": oid.crah_rat_plot(plot_type=self.p_format),
            "plot_name": "Return Air Temp Plot",
            "suggested_action": oid.rat_action
        })
        plots.append({
            "plot": oid.fan_speed_plot(plot_type=self.p_format),
            "plot_name": "Fan Speed Plot",
            "suggested_action": oid.eld_action
        })
        plots.append({
            "plot": oid.cooling_valve_plot(plot_type=self.p_format),
            "plot_name": "Cooling Valve Plot",
            "suggested_action": oid.cvl_action
        })
        plots.append({
            "plot": oid.crah_sat_plot(plot_type=self.p_format),
            "plot_name": "Supply Air Temp Plot",
            "suggested_action": oid.cna_action
        })
        return plots

    def _dp_sensor(self):
        """ Runs the dp sensor plot
        """
        oid = OperationalIssues(self.df)
        plots = []
        plots.append({
            "plot": oid.underfloor_pressure_plot(plot_type=self.p_format),
            "plot_name": "Differential Pressure Plot",
            "suggested_action": oid.pna_action
        })
        return plots

    def _oat_sensor(self):
        """ Runs the oat sensor plot
        """
        oid = OperationalIssues(self.df)
        plots = []
        plots.append({
            "plot": oid.oat_sensor_plot(plot_type=self.p_format),
            "plot_name": "Outside Air Temperature Plot",
            "suggested_action": oid.oat_action
        })
        return plots

    def _crah_sat(self):
        """ Runs the crah equipment plots
        Args:
            None
        Returns:
            None
        """
        oid = OperationalIssues(self.df)
        plots = []
        plots.append({
            "plot": oid.crah_sat_plot(plot_type=self.p_format),
            "plot_name": "Supply Air Temp Plot",
            "suggested_action": oid.cna_action
        })
        return plots

    def _crah_rat(self):
        """ Runs the crah equipment plots
        Args:
            None
        Returns:
            None
        """
        oid = OperationalIssues(self.df)
        plots = []
        plots.append({
            "plot": oid.crah_rat_plot(plot_type=self.p_format),
            "plot_name": "Return Air Temp Plot",
            "suggested_action": oid.rat_action
        })
        return plots

    def _crah_clv(self):
        """ Runs the crah equipment plots
        Args:
            None
        Returns:
            None
        """
        oid = OperationalIssues(self.df)
        plots = []
        plots.append({
            "plot": oid.cooling_valve_plot(plot_type=self.p_format),
            "plot_name": "Cooling Valve Plot",
            "suggested_action": oid.cvl_action
        })
        return plots

    def _crah_fanspd(self):
        """ Runs the crah equipment plots
        Args:
            None
        Returns:
            None
        """
        oid = OperationalIssues(self.df)
        plots = []
        plots.append({
            "plot": oid.fan_speed_plot(plot_type=self.p_format),
            "plot_name": "Fan Speed Plot",
            "suggested_action": oid.eld_action
        })
        return plots

    def create_plot(self, args):
        """
        Args:
            args (dict): this is the payload that the viz engine is going
            to receive from the front end and has the ff schema
                equipment (str): equipment_name
                start (str): YYYY-MM-DD
                end (str): YYYY-MM-DD
                plot_format (str): 'html' or 'json'
        Returns:
            response (dict): this is the response object which has the ff
            schema:
                plots (list): list of plot objects {
                    plot (str or dict): html or dict of plots
                    plot_name (str): name of plot
                    suggested_action (str): name of suggested actions
                }
        """
        eqmt, start, end = args["equipment"], args["start"], args["end"]
        self.p_format = args["plot_format"]
        if eqmt in self.crah:
            query = self.crah_query.format(crah=eqmt, start=start, end=end)
            self.df = self.db.query_df(query)
            plots = self._crah_equipment()
        elif eqmt == "DP":
            query = self.dp_query.format(start=start, end=end)
            self.df = self.db.query_df(query)
            plots = self._dp_sensor()
        elif eqmt == "OAT":
            query = self.oat_query.format(start=start, end=end)
            self.df = self.db.query_df(query)
            plots = self._oat_sensor()
        elif '-' in eqmt:
            crah, plot_type = eqmt.split('-')
            query = self.crah_query.format(crah=crah, start=start, end=end)
            self.df = self.db.query_df(query)
            if plot_type == "SAT":
                plots = self._crah_sat()
            elif plot_type == "RAT":
                plots = self._crah_rat()
            elif plot_type == "CLV":
                plots = self._crah_clv()
            elif plot_type == "FANSPD":
                plots = self._crah_fanspd()
        response = {
            "plots": plots
        }
        return response

    def _create_dash_df(self, args):
        """
        Creates the dashboard df
        Args:
            args (dict): this is the payload that the viz engine is going
            to receive from the front end and has the ff schema
                start (str): YYYY-MM-DD
                end (str): YYYY-MM-DD
                plot_format (str): 'html' or 'json'
        Returns:
            None
        """
        start, end = args["start"], args["end"]
        self.p_format = args["plot_format"]
        self.dash_df = pd.DataFrame(self.rules, columns=['rule'])
        self.dash_df.set_index('rule', inplace=True)
        for equipment in self.equipments:
            if equipment == "OAT":
                query = self.oat_query.format(start=start, end=end)
                self.df = self.db.query_df(query)
                oid = OperationalIssues(self.df)
                oid.oat_dashboard()
                self.dash_df = self.dash_df.join(oid.dashboard['p_score'],
                                                 rsuffix=equipment)
            elif equipment in self.crah:
                query = self.crah_query.format(crah=equipment, start=start,
                                               end=end)
                self.df = self.db.query_df(query)
                oid = OperationalIssues(self.df)
                oid.crah_dashboard()
                self.dash_df = self.dash_df.join(oid.dashboard['p_score'],
                                                 rsuffix=equipment)
            else:
                query = self.dp_query.format(start=start, end=end)
                self.df = self.db.query_df(query)
                oid = OperationalIssues(self.df)
                oid.pressure_dashboard()
                self.dash_df = self.dash_df.join(oid.dashboard['p_score'],
                                                 rsuffix=equipment)
        self.dash_df.columns = self.equipments

    def create_dashboard(self, args):
        """
        Args:
            args (dict): this is the payload that the viz engine is going
            to receive from the front end and has the ff schema
                start (str): YYYY-MM-DD
                end (str): YYYY-MM-DD
                plot_format (str): 'html' or 'json'
        Returns:
            response (dict): this is the response object which has the ff
            schema:
                plots (list): list of plot objects {
                    plot (str or dict): html or dict of plots
                    plot_name (str): name of plot
                    suggested_action (str): name of suggested actions
                }
        """
        self._create_dash_df(args)
        oid = OperationalIssues(self.dash_df)
        plots = []
        plots.append({
            "plot": oid.dashboard_plot(plot_type=self.p_format),
            "plot_name": "Health Dashboard"
        })
        response = {
            "plots": plots
        }
        return response

    def _create_heatmap_df(self, args):
        """
        Creates the heatmap df
        Args:
            args (dict): this is the payload that the viz engine is going
            to receive from the front end and has the ff schema
                start (str): YYYY-MM-DD
                plot_format (str): 'html' or 'json'
                "forecasted":
                     - "YYYY-MM-DD hh:00:00" this is the forecasted time
                period ahead of the current start period (since a baseline of
                now needs to be detremined in creating the forecasts)
                     - OR None if we only need the current heatmap at that time
        Returns:
        """
        w, h = self.image_d
        start, forecasted = args['start'], args['forecasted']
        self.p_format = args["plot_format"]
        query = self.server_query.format(start=start)
        self.heat_df = self.db.query_df(query)
        server_dim = self.db.query_db(self.inlet_dim_query)
        server_dim = pd.DataFrame(server_dim)
        server_dim.set_index('id', inplace=True)
        server_dim['x_pos'] = server_dim['x_pos'] - 37
        server_dim['y_pos'] = server_dim['y_pos'] - 1
        self.heat_df = server_dim.join(self.heat_df.T)
        self.heat_df.columns = ['sensor_type', 'x', 'y', 'temp']
        self.heat_df['x'] = \
            ((1.0 * self.heat_df['x']) / self.heat_df['x'].max()) * w
        self.heat_df['y'] = \
            ((1.0 * self.heat_df['y']) / self.heat_df['y'].max()) * h

    def create_heatmap(self, args):
        """
        Args:
            args (dict): this is the payload that the viz engine is going
            to receive from the front end and has the ff schema
                start (str): YYYY-MM-DD hh:00:00
                plot_format (str): 'html' or 'json'
                "forecasted":
                     - "YYYY-MM-DD hh:00:00" this is the forecasted time
                    period ahead of the current start period (since a baseline
                    of now needs to be detremined in creating the forecasts)
                     - OR None if we only need the current heatmap at that time
        Returns:
            response (dict): this is the response object which has the ff
            schema:
                plots (list): list of plot objects {
                    plot (str or dict): html or dict of plots
                    plot_name (str): name of plot
                    suggested_action (str): name of suggested actions
                }
        NOTE - this is going to ignore the forecasted items for now and will
        """
        self._create_heatmap_df(args)
        oid = OperationalIssues(self.heat_df)
        self.p_format = args["plot_format"]
        plots = []
        plots.append({
            "plot": oid.heatmap_plot(plot_type=self.p_format),
            "plot_name": "Heatmap"
        })
        response = {
            "plots": plots
        }
        return response

    def load_sat_optimization_models(self):
        """
        Args:
            None
        Returns:
            None, creates the opt_dict attribute which is a dictionary
            containing different CrahOptimizer objects modeled to fit
            the crah unit there
        """
        self.opt_dict = {}
        opt = CrahOptimizer(verbose=True, model='lr', w=144)
        for crah in opt.crah_units:
            opt.fit(crah)
            self.opt_dict[crah] = copy.copy(opt)

    def crah_inlet_predictions(self, args):
        """
        Args:
            args (dict): this is the payload that the viz engine is going
            to receive from the front end and has the ff schema
                crah (str): this is the code of the crah unit
                delta_sat (float): this is the change in supply air temperature
                plot_format (str): 'html' or 'json'
        Returns:
            response (dict): this is the response object which has the ff
            schema:
                plots (list): list of plot objects {
                    plot (str or dict): html or dict of plots
                    plot_name (str): name of plot
                    suggested_action (str): name of suggested actions
                }
        """
        crah = args['crah']
        delta_sat = args['delta_sat']
        plot_format = args['plot_format']
        opt = self.opt_dict[crah]
        opt.predict(delta_sat)
        plots = []
        plots.append({
            "plot": opt.plot_predictions(plot_type=plot_format),
            "plot_name": "Delta Sat Prediction"
        })
        response = {
            "plots": plots
        }
        return response

    def sat_optimization_dashboard(self, args):
        """
        Args:
            args (dict): this is the payload that the viz engine is going
            to receive from the front end and has the ff schema
                delta_sat (float): this is the change in supply air temperature
                plot_format (str): 'html' or 'json'
        Returns:
            response (dict): this is the response object which has the ff
            schema:
                plots (list): list of plot objects {
                    plot (str or dict): html or dict of plots
                    plot_name (str): name of plot
                    suggested_action (str): name of suggested actions
                }

        NOTE - add argument here (optimal) to use the optimal setting values
        """
        dashboard = []
        self.delta_sat = args["delta_sat"]
        self.p_format = args["plot_format"]
        with_excursions = args["with_excursions"]
        if with_excursions:
            for crah in self.opt_dict.keys():
                opt = self.opt_dict[crah]
                if self.delta_sat == 999.0:
                    delta_sat = self.optimal_delta_sat[crah]
                    opt.predict(delta_sat)
                    cap = opt.estimate_total_eca(delta_sat)
                else:
                    opt.predict(self.delta_sat)
                    cap = opt.estimate_total_eca(self.delta_sat)
                row = (
                    crah,
                    opt.excursions,
                    opt.excursion_pct,
                    cap
                )
                dashboard.append(row)
            self.opt_dash_df = pd.DataFrame(dashboard)
            self.opt_dash_df.columns = ['crah', 'num_excursions',
                                        'excursion_pct', 'cap']
            self.opt_dash_df.set_index('crah', inplace=True)
        else:
            for crah in self.opt_dict.keys():
                opt = self.opt_dict[crah]
                if self.delta_sat == 999.0:
                    delta_sat = self.optimal_delta_sat[crah]
                    cap = opt.estimate_total_eca(delta_sat)
                else:
                    cap = opt.estimate_total_eca(self.delta_sat)
                row = (
                    crah,
                    cap
                )
                dashboard.append(row)
            self.opt_dash_df = pd.DataFrame(dashboard)
            self.opt_dash_df.columns = ['crah', 'cap']
            self.opt_dash_df.set_index('crah', inplace=True)
        plots = []
        plots.append({
            "plot": opt.plot_dashboard(self.opt_dash_df,
                                       self.delta_sat,
                                       plot_type=self.p_format),
            "plot_name": "Optimization Dashboard"
        })
        response = {
            "plots": plots
        }
        return response

    def outlier_detection(self, args):
        """
        Args:
            args (dict): this is the payload that the viz engine is going
            to receive from the front end and has the ff schema
                unit (str): this is the equipment unit to have outliers
                checked on "ALL" if peer analytics will be executed else
                "DH11" for example if only crah unit 11 data will be pulled
                metric (str): this is the metric that will be evaluated
                start (str): YYYY-MM-DD
                end (str): YYYY-MM-DD
                plot_format (str): 'html' or 'json'
        Returns:
            response (dict): this is the response object which has the ff
            schema:
                plots (list): list of plot objects {
                    plot (str or dict): html or dict of plots
                    plot_name (str): name of plot
                    suggested_action (str): name of suggested actions
                }
        """
        start, end = args['start'], args['end']
        unit, metric = args['unit'], args['metric']
        self.p_format = args['plot_format']
        if metric in self.metric_codes:
            new_metric = self.metric_codes[metric]
        if metric == "DP":
            query = self.dp_query.format(start=start, end=end)
            df = self.db.query_df(query)
            df = df[filter(lambda col: '_interp' in col, df.columns)]
            df.columns = [col.replace('_interp', '') for col in df.columns]
            df.drop('dp_15', axis=1, inplace=True)
            tsad = TSAnomalyDetection(n_neighbors=2)
        elif unit == "ALL":
            query = self.all_crah_metric.format(metric=new_metric,
                                                start=start, end=end)
            df = self.db.query_df(query)
            tsad = TSAnomalyDetection()
        elif unit in self.crah:
            query = self.crah_query.format(crah=unit, start=start, end=end)
            df = self.db.query_df(query)
            df = df[new_metric]
            tsad = TSAnomalyDetection(w=int(.05 * df.shape[0]))
        tsad.fit(df.dropna())
        plots = []
        plots.append({
            "plot": tsad.plot(plot_type=self.p_format,
                              name=metric),
            "plot_name": "Outlier Plot"
        })
        response = {
            "plots": plots
        }
        return response

    def _plot_total_dashboard(self, plot_type='show'):
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
        fig = plt.figure(figsize=(10, 4))
        total_savings = int(self.total_df.sum().sum())
        if self.delta_sat == 999.0:
            delta_sat = 'Optimized'
        else:
            delta_sat = self.delta_sat
        if self.min_fan_spd == 999.0:
            min_fan_spd = 'Optimized'
        else:
            min_fan_spd = self.min_fan_spd
        title = 'Annual Cost Avoidance Potential'
        title += ' of CAD$ {} '.format("{:,}".format(total_savings))
        if self.delta_sat == 999.0 and self.min_fan_spd == 999.0:
            title += 'under optimal SAT and Fan-Spd Conditions'
        elif self.delta_sat == 999.0:
            title += 'under optimal SAT conditions '
            title += 'and Min Fan Spd = {}'.format(min_fan_spd)
        elif self.min_fan_spd == 999.0:
            title += 'when delta SAT = {} '.format(delta_sat)
            title += 'and when Fan Spd is optimized'
        else:
            title += 'when delta SAT = {} '.format(delta_sat)
            title += 'and Min Fan Spd = {}'.format(min_fan_spd)
        plt.title(title)
        ax1 = fig.add_subplot(1, 2, 1)
        self.total_df.sum().plot(kind='pie', ax=ax1, subplots=True,
                                 autopct='%1.1f%%')

        ax1.yaxis.set_label_text('')
        ax1.yaxis.label.set_visible(False)
        ax2 = fig.add_subplot(1, 2, 2)
        self.total_df.plot.barh(stacked=True, ax=ax2)
        ax2.yaxis.set_label_text('CRAH Units')
        ax2.xaxis.set_label_text('CAD$')
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

    def total_optimization_dashboard(self, args):
        """
        Args:
            args (dict): this is the payload that the viz engine is going
            to receive from the front end and has the ff schema
                delta_sat (float): this is the change in supply air temperature
                plot_format (str): 'html' or 'json'
        Returns:
            response (dict): this is the response object which has the ff
            schema:
                plots (list): list of plot objects {
                    plot (str or dict): html or dict of plots
                    plot_name (str): name of plot
                    suggested_action (str): name of suggested actions
                }

        NOTE - add argument here (optimal) to use the optimal setting values
        """
        # run sat optimization
        dashboard = []
        self.min_fan_spd = args["min_fan_spd"]
        self.delta_sat = args["delta_sat"]
        self.p_format = args["plot_format"]
        for crah in self.opt_dict.keys():
            opt = self.opt_dict[crah]
            if self.delta_sat == 999.0:
                delta_sat = self.optimal_delta_sat[crah]
                cap = opt.estimate_total_eca(delta_sat)
                print self.delta_sat, crah, delta_sat, cap
            else:
                cap = opt.estimate_total_eca(self.delta_sat)
            row = (
                crah,
                cap
            )
            dashboard.append(row)
        self.total_df = pd.DataFrame(dashboard)
        self.total_df.columns = ['crah', 'sat']
        self.total_df.set_index('crah', inplace=True)
        # run fan optimization
        if self.min_fan_spd == 999.0:
            fan_df = self.fan_df
        else:
            self.fanopt.compute_savings(min_fan_speed=self.min_fan_spd)
            fan_df = pd.DataFrame(self.fanopt.power_df.sum(),
                                  columns=['fan_spd'])
            fan_df = fan_df.loc[self.optimal_delta_sat.keys()]
        self.total_df = fan_df.join(self.total_df)
        # converting it into an annual
        self.total_df = self.total_df * 4
        self.total_df = self.total_df.assign(total=self.total_df['fan_spd'] +
                                             self.total_df['sat'])
        self.total_df = self.total_df.sort_values('total', ascending=False)
        self.total_df.drop('total', axis=1, inplace=True)
        self.total_df = self.total_df.fillna(0).astype(int)
        print 'this is the total df'
        print self.total_df
        print 'this is the total per type of saving'
        print self.total_df.sum()
        print '\n'
        print 'this is the total savings'
        print self.total_df.sum().sum()
        print '\n'
        plots = []
        plots.append({
            "plot": self._plot_total_dashboard(plot_type=self.p_format),
            "plot_name": "Total Optimization Dashboard"
        })
        response = {
            "plots": plots
        }
        return response


if __name__ == "__main__":
    plotter = PlotManager()
    plotter.load_sat_optimization_models()
    payload = {
        "min_fan_spd": 999.0,
        "plot_format": "show",
        "delta_sat": 999.0
    }
    plotter.total_optimization_dashboard(payload)

