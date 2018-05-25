// MDN Tools and Docs > Web Storage API
// https://developer.mozilla.org/en-US/docs/Web/API/Web_Storage_API

/*  design pattern in lieu of REST API calls ...
    store form element parameters to construct dynamic urls 
    said urls parse routes of rendered visualizations via mpld3  */

//  Navigation Scheme and Label Terminology Subject to Change



// #################### DASHBOARD(DB)
// #################### Title: Cooling Unit Health
// #################### File: dashboard.html

    // collect parameters
    function db_storeDates() {

        // find date-from
        var db_inputDateFrom = document.getElementById("db-date-from");
        // set date-from value
        localStorage.setItem("db-date-from", db_inputDateFrom.value);
        // get date-from value
        var db_dateFromValue = localStorage.getItem("db-date-from");
            // log date-from value
            console.log("date-from: " + db_dateFromValue + " stored in local cache");

        // find date-to
        var db_inputDateTo = document.getElementById("db-date-to");
        // set date-to value
        localStorage.setItem("db-date-to", db_inputDateTo.value);
        // get date-to value
        var db_dateToValue = localStorage.getItem("db-date-to");
            // log date-to value
            console.log("date-to: " + db_dateToValue + " stored in local cache");

        // build dynamic query string
        var db_base_url = "http://cbre-cooling-optimization.mybluemix.net/dashboard/";
        var db_url = db_base_url + db_dateFromValue + "_" + db_dateToValue;
            console.log(db_url);

        // pass url to iframe
        document.getElementById('dashboard').src = db_url;

    }

    // reset default state
    function db_reset() {
        // set default url(s) for reset
        var db_default_url = "http://cbre-cooling-optimization.mybluemix.net/dashboard/20171021_20171031/";

        // pass default url to iframe
        document.getElementById('dashboard').src = db_default_url;
            console.log("reset default @ " + db_default_url);
    }



// #################### DATA CENTER(DC)
// #################### Title: Ambient Temp Plenum Pressure
// #################### File: datacenter.html

    // collect parameters
    function dc_storeDates() {

        // find date-from
        var dc_inputDateFrom = document.getElementById("dc-date-from");
        // set date-from value
        localStorage.setItem("dc-date-from", dc_inputDateFrom.value);
        // get date-from value
        var dc_dateFromValue = localStorage.getItem("dc-date-from");
            // log date-from value
            console.log("From: " + dc_dateFromValue + " stored in local cache");

        // find date-to
        var dc_inputDateTo = document.getElementById("dc-date-to");
        // set date-to
        localStorage.setItem("dc-date-to", dc_inputDateTo.value);
        // get date-to
        var dc_dateToValue = localStorage.getItem("dc-date-to");
            // log date-to
            console.log("To: " + dc_dateToValue + " stored in local cache");


        // build dynamic query string
        var dc_base_url = "http://cbre-cooling-optimization.mybluemix.net/ops_issues/";
        var oat_url = dc_base_url + "OAT_" + dc_dateFromValue + "_" + dc_dateToValue;
        var dp_url = dc_base_url + "DP_" + dc_dateFromValue + "_" + dc_dateToValue;
            console.log(oat_url);
            console.log(dp_url);

        // pass url(s) to iframe(s)
        document.getElementById('outside-air').src = oat_url;
        document.getElementById('diff-pressure').src = dp_url;
    }

    // reset default state
    function dc_reset() {
        // set default url(s) for reset
        var oat_default = "http://cbre-cooling-optimization.mybluemix.net/ops_issues/OAT_20171021_20171031";
        var dp_default = "http://cbre-cooling-optimization.mybluemix.net/ops_issues/DP_20171021_20171031";

        // pass default url(s) to iframe(s)
        document.getElementById('outside-air').src = oat_default;
            console.log("reset default OAT @ " + oat_default);
        document.getElementById('diff-pressure').src = dp_default;
            console.log("reset default DP @ " + dp_default);
    }



// #################### HEATMAP(HM)
// #################### Title: Data Center Heatmap
// #################### File: heatmap.html

    // collect parameters
    function hm_storeDate() {

        // find date
        var hm_inputDate = document.getElementById("heatmap-date");
        // set date value
        localStorage.setItem("heatmap-date", hm_inputDate.value);
        // get date value
        var hm_dateValue = localStorage.getItem("heatmap-date");
            // log date value
            console.log("Date Value: " + hm_dateValue + " stored in local cache");

        // build dynamic query string
        var hm_base_url = "http://cbre-cooling-optimization.mybluemix.net/heatmap/"
        var hm_url = hm_base_url + hm_dateValue;
            console.log(hm_url);

        // pass url to iframe
        document.getElementById('heatmap').src = hm_url;
    }

    // reset default state
    function hm_reset() {
        // set default url for reset
        var hm_default_url = "http://cbre-cooling-optimization.mybluemix.net/heatmap/20170405/";
        // pass default url to iframe
        document.getElementById('heatmap').src = hm_default_url;
            console.log("reset default @ " + hm_default_url);
    }



// #################### UNIT OPTIMIZATION ISSUES(UOI)
// #################### Title: Cooling Unit Optimization Issues
// #################### File: unit-op-issue.html

    // collect parameters
    function uoi_storeData() {

        // find unit
        var uoi_selectUnit = document.getElementById("uoi-unit");
        // set unit value
        localStorage.setItem("uoi-unit", uoi_selectUnit.value);
        // get unit value
        var uoi_unitValue = localStorage.getItem("uoi-unit");
            // log unit value
            console.log("CRAH Unit: " + uoi_unitValue + " stored in local cache");

        // find date from
        var uoi_inputDateFrom = document.getElementById("uoi-date-from");
        // set date-from value
        localStorage.setItem("uoi-date-from", uoi_inputDateFrom.value);
        // get date-from value
        var uoi_dateFromValue = localStorage.getItem("uoi-date-from");
            // log date-from value
            console.log("From: " + uoi_dateFromValue + " stored in local cache");

        // find date to
        var uoi_inputDateTo = document.getElementById("uoi-date-to");
        // set date-to value
        localStorage.setItem("uoi-date-to", uoi_inputDateTo.value);
        // get date-to value
        var uoi_dateToValue = localStorage.getItem("uoi-date-to");
            // log date-to value
            console.log("To: " + uoi_dateToValue + " stored in local cache");

        // build dynamic query string(s)
        var uoi_base_url = "http://cbre-cooling-optimization.mybluemix.net/ops_issues/"
        var uoi_sat_url = uoi_base_url + uoi_unitValue + "-SAT_" + uoi_dateFromValue + "_" + uoi_dateToValue;
            console.log(uoi_sat_url);
        var uoi_rat_url = uoi_base_url + uoi_unitValue + "-RAT_" + uoi_dateFromValue + "_" + uoi_dateToValue;
            console.log(uoi_rat_url);
        var uoi_clv_url = uoi_base_url + uoi_unitValue + "-CLV_" + uoi_dateFromValue + "_" + uoi_dateToValue;
            console.log(uoi_clv_url);
        var uoi_fanspd_url = uoi_base_url + uoi_unitValue + "-FANSPD_" + uoi_dateFromValue + "_" + uoi_dateToValue;
            console.log(uoi_fanspd_url);

        // pass url(s) to iframe(s)
        document.getElementById('met-sat').src = uoi_sat_url;
        document.getElementById('met-rat').src = uoi_rat_url;
        document.getElementById('met-clv').src = uoi_clv_url;
        document.getElementById('met-fanspd').src = uoi_fanspd_url;
    }

    // reset default state
    function uoi_reset() {
        // set default url for reset
        var uoi_default_sat_url = "http://cbre-cooling-optimization.mybluemix.net/ops_issues/DH1-SAT_20170901_20171030";
        var uoi_default_rat_url = "http://cbre-cooling-optimization.mybluemix.net/ops_issues/DH1-RAT_20170901_20171030";
        var uoi_default_clv_url = "http://cbre-cooling-optimization.mybluemix.net/ops_issues/DH1-CLV_20170901_20171030";
        var uoi_default_fanspd_url = "http://cbre-cooling-optimization.mybluemix.net/ops_issues/DH1-FANSPD_20170901_20171030";
        
        // pass default url to iframe(s)
        document.getElementById('met-sat').src = uoi_default_sat_url;
            console.log("reset default @ " + uoi_default_sat_url);
        document.getElementById('met-rat').src = uoi_default_rat_url;
            console.log("reset default @ " + uoi_default_rat_url);
        document.getElementById('met-clv').src = uoi_default_clv_url;
            console.log("reset default @ " + uoi_default_clv_url);
        document.getElementById('met-fanspd').src = uoi_default_fanspd_url;
            console.log("reset default @ " + uoi_default_fanspd_url);
    }



// #################### SETPOINT OPTIMIZATION(SPO)
// #################### Title: Cooling Unit Optimization
// #################### File: setpoint-opt.html

    // collect parameters
    function spo_storeData() {

        // find degree 
        var spo_selectDegree = document.getElementById("spo-degree");
        // set degree value
        localStorage.setItem("spo-degree", spo_selectDegree.value);
        // get degree value
        var spo_degreeValue = localStorage.getItem("spo-degree");
            // log degree value
            console.log("Degree Increase: " + spo_degreeValue + " stored in local cache");

        // find percentage
        var spo_selectPercentage = document.getElementById("spo-percentage");
        // set percent value
        localStorage.setItem("spo-percentage", spo_selectPercentage.value);
        // get percent value
        var spo_percentageValue = localStorage.getItem("spo-percentage");
            // log percent value
            console.log("Fan Speed: -" + spo_percentageValue + " stored in local cache");

        // build dynamic query string
        var spo_base_url = "http://cbre-cooling-optimization.mybluemix.net/full_opt_dashboard/";
        var spo_url = spo_base_url + spo_degreeValue + "_" + spo_percentageValue;
            console.log(spo_url);

        // pass url to iframe(s)
        document.getElementById('setpoint-opt').src = spo_url;

    }

    // reset default state
    function spo_reset() {
        // set default url for reset
        var spo_default_url = "http://cbre-cooling-optimization.mybluemix.net/full_opt_dashboard/999_999";
        // pass default url to iframe(s)
        document.getElementById('setpoint-opt').src = spo_default_url;
            console.log("reset default @ " + spo_default_url);
    }



// #################### SIMULATION OPTIMIZATION(SO)
// #################### Title: What If Analysis
// #################### File: sim-opt.html

    // collect parameters
    function so_storeData() {

        // find unit 
        var so_selectUnit = document.getElementById("so-unit");
        // set unit value
        localStorage.setItem("so-unit", so_selectUnit.value);
        // get unit value
        var so_unitValue = localStorage.getItem("so-unit");
            // log unit value
            console.log("CRAH Unit: " + so_unitValue + " stored in local cache");

        // find degree 
        var so_selectDegree = document.getElementById("so-degree");
        // set degree value
        localStorage.setItem("so-degree", so_selectDegree.value);
        // get degree value
        var so_degreeValue = localStorage.getItem("so-degree");
            // log degree value
            console.log("Degree Increase: " + so_degreeValue + " stored in local cache");

        // build dynamic query string
        var so_base_url = "http://cbre-cooling-optimization.mybluemix.net/inlet_prediction/";
        var so_url = so_base_url + so_unitValue + "_" + so_degreeValue;
            console.log(so_url);

        // pass url to iframe(s)
        document.getElementById('sim-opt').src = so_url;
 
    }

    // reset default state
    function so_reset() {
        // set default url for reset
        var so_default_url = "http://cbre-cooling-optimization.mybluemix.net/inlet_prediction/DH4_1";

        // pass default url to iframe(s)
        document.getElementById('sim-opt').src = so_default_url;
            console.log("reset default @ " + so_default_url);

    }
