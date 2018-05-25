# -*- coding: utf-8 -*-
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
from flask import Flask, render_template, render_template_string, request, jsonify, flash, redirect, session, abort, url_for
from plot_manager import PlotManager
from flask_wtf import Form
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
port = int(os.getenv('PORT', 8080))
plotter = PlotManager()

if len(sys.argv) > 1:
    status = sys.argv[1]
else:
    status = "prod"
if status == "prod":
    print 'begin loading machine learning models'
    plotter.load_sat_optimization_models()
    print '=================================================================='
    print '======================done loading models========================='
    print '=================================================================='


def format_date(date_string):
    """
    Args:
        date_string (str): this is a date in YYYYMMDD format
    Returns:
        date_string (str): this is a date in YYYY-MM-DD format
    """
    year = date_string[:4]
    month = date_string[4:6]
    date = date_string[6:8]
    date_string = '{}-{}-{}'.format(year, month, date)
    return date_string

# route for handling the login page logic
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'cooling@cbre.com' or request.form['password'] != 'Dcost@':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('heatmap'))
    return render_template('login.html', error=error)

#@app.route('/', methods=['GET'])
#def login():
#    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/datacenter")
def datacenter():
    return render_template("datacenter.html")

@app.route("/heatmap")
def heatmap():
    return render_template("heatmap.html")

@app.route("/unit-operation-issues")
def unitOpIssues():
    return render_template("unit-op-issues.html")

@app.route("/setpoint-optimization")
def setpointOpt():
    return render_template("setpoint-opt.html")

@app.route("/sim-optimization")
def simOpt():
    return render_template("sim-opt.html")

# Dynamic Routes

@app.route('/ops_issues/<equipment_start_end>', methods=['GET'])
def ops_issues_plot(equipment_start_end):
    """
    Args:
        equipment_start_end (str): this is a EQUIPMENT_YYYYDDMM_YYYYDDMM
        code which signifies the two start and end dates to be used
    """
    equipment, start, end = equipment_start_end.split('_')
    payload = {
        "equipment": equipment,
        "start": "{}".format(format_date(start)),
        "end": "{}".format(format_date(end)),
        "plot_format": "html"
    }
    response = plotter.create_plot(payload)
    plot = response['plots'][0]['plot']
    return render_template_string(plot)


@app.route('/dashboard/<start_end>/', methods=['GET'])
def dashboard_plot(start_end):
    """
    NOTE - this is just a sample of the kind of html that is going to be
    returned by the UI
    """
    start, end = start_end.split('_')
    payload = {
        "start": "{}".format(format_date(start)),
        "end": "{}".format(format_date(end)),
        "plot_format": "html"
    }
    response = plotter.create_dashboard(payload)
    plot = response['plots'][0]['plot']
    return render_template_string(plot)


@app.route('/heatmap/<start>/', methods=['GET'])
def heatmap_plot(start):
    """
    NOTE - this is just a sample of the kind of html that is going to be
    returned by the UI
    """
    payload = {
        "start": "{} 12:00:00".format(format_date(start)),
        "plot_format": "html",
        "forecasted": None
    }
    response = plotter.create_heatmap(payload)
    plot = response['plots'][0]['plot']
    return render_template_string(plot)


@app.route('/inlet_prediction/<crah_deltasat>', methods=['GET'])
def inlet_prediction(crah_deltasat):
    """
    Args:
        crah_deltasat (str): This is the crah unit with the following
        choices "DH10", "DH11", "DH14", "DH18", "DH20", "DH21",
        "DH23", "DH24", "DH26", "DH4", "DH5", "DH7", "DH9", and
        then the deltasat which is a float value for delta sat
    """
    crah, deltasat = crah_deltasat.split('_')
    payload = {
        "crah": crah,
        "delta_sat": float(deltasat),
        "plot_format": "html"
    }
    response = plotter.crah_inlet_predictions(payload)
    plot = response['plots'][0]['plot']
    return render_template_string(plot)


@app.route('/sat_opt_dashboard/<deltasat>', methods=['GET'])
def sat_opt_dashboard(deltasat):
    """
    Args:
        deltasat (float): this is a float value, decimals can be handled
    """
    payload = {
        "delta_sat": float(deltasat),
        "plot_format": "html"
    }
    response = plotter.sat_optimization_dashboard(payload)
    plot = response['plots'][0]['plot']
    return render_template_string(plot)


@app.route('/outlier_detection/<unit_metric_start_end>', methods=['GET'])
def outlier_detection(unit_metric_start_end):
    """
    Args:
        unit_metric_start_end (str): this is a UNIT_METRIC_YYYYDDMM_YYYYDDMM
        code which signifies the two start and end dates to be used as well
        as the unit and equipment that the data will be drawn from
    """
    unit, metric, start, end = unit_metric_start_end.split('_')
    payload = {
        "unit": unit,
        "metric": metric,
        "start": "{}".format(format_date(start)),
        "end": "{}".format(format_date(end)),
        "plot_format": "html"
    }
    response = plotter.outlier_detection(payload)
    plot = response['plots'][0]['plot']
    return render_template_string(plot)


@app.route('/full_opt_dashboard/<deltasat_minfanspd>', methods=['GET'])
def full_opt_dashboard(deltasat_minfanspd):
    """
    Args:
        deltasat_minfanspd (str): This is both the deltasat as well as
        the minfanspd
    """
    delta_sat, min_fan_spd = deltasat_minfanspd.split('_')
    payload = {
        "min_fan_spd": float(min_fan_spd),
        "delta_sat": float(delta_sat),
        "with_excursions": False,
        "plot_format": "html"
    }
    response = plotter.total_optimization_dashboard(payload)
    plot = response['plots'][0]['plot']
    return render_template_string(plot)


# REST APIs

@app.route('/api/v1/query/', methods=['POST'])
def get_query():
    """
    Payload:
        requests.post(url + /api/v1/query/, json={"query": "SQL QUERY HERE"})
    Returns:
        {
            "results": results (list) - where this is the list of dict objects
            where a dict is each row that is going to be a result of the
            SQL query on dashdb
        }
    """
    query = request.json['query']
    results = plotter.db.query_db(query)
    return jsonify({'results': results})


@app.route('/api/v1/analytics/', methods=['POST'])
def plot_analytics():
    """
    Payload:
        payload (dict): {
                equipment (str): equipment_name
                start (str): YYYY-MM-DD
                end (str): YYYY-MM-DD
                plot_format (str): 'html' or 'json'
            }
    Returns:
        response (dict): this is the response object which has the ff
        schema:
            plots (list): list of plot objects {
                plot (str or dict): html or dict of plots
                plot_name (str): name of plot
                suggested_action (str): name of suggested actions
            }
    """
    payload = request.json
    response = plotter.create_plot(payload)
    return jsonify(response)


@app.route('/api/v1/dashboard/', methods=['POST'])
def plot_dashboard():
    """
    Payload:
        payload (dict): {
                start (str): YYYY-MM-DD
                end (str): YYYY-MM-DD
                plot_format (str): 'html' or 'json'
            }
    Returns:
        response (dict): this is the response object which has the ff
        schema:
            plots (list): list of plot objects {
                plot (str or dict): html or dict of plots
                plot_name (str): name of plot
                suggested_action (str): name of suggested actions
            }
    """
    payload = request.json
    response = plotter.create_dashboard(payload)
    return jsonify(response)


@app.route('/api/v1/heatmap/', methods=['POST'])
def plot_heatmap():
    """
    Payload:
        payload (dict): {
                "start": "YYYY-MM-DD hh:mm:ss",
                "plot_format": "html" or "json" (html if html string else json
                if object)
                "forecasted": "YYYY-MM-DD hh:00:00" this is the forecasted
                time period ahead of the current start period (since a baseline
                of now needs to be detremined in creating the forecasts)
            }
    Returns:
        response (dict): this is the response object which has the ff
        schema:
            plots (list): list of plot objects {
                plot (str or dict): html or dict of plots
                plot_name (str): name of plot
                suggested_action (str): name of suggested actions
            }
    """
    payload = request.json
    response = plotter.create_heatmap(payload)
    return jsonify(response)


if __name__ == "__main__":
    if status == "prod":
        print 'loading prod mode'
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    else:
        print 'loading dev mode'
        app.run(host='0.0.0.0', port=port, debug=True)
