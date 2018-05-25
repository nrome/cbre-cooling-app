# -*- coding: utf-8 -*-

import json
import ibm_db
import pandas as pd


class DashdbConnect(object):
    """
    This class is responsible for all matters related to the retrieval
    of data from dashdb as a generic SQL query based interface

    This class is going to have a single method on it which is 'query db'
    which is going to execute the sql statement on the database
    """
    def __init__(self):
        with open('env/dash_db.json') as config:
            self.cred = json.load(config)
        self.conn = ibm_db.pconnect(self.cred["dsn"],
                                    self.cred["username"],
                                    self.cred["password"])

    def query_db(self, query):
        """
        Args:
            query (str): this is the SQL query statement which is being used
            to get data from the database
        Returns:
            results (list): this is the list of result rows where
            each result row is a dictionary that has the keys as the columns
            of the response, and the values as the value of that item
            in the database
        """
        results = []
        stmt = ibm_db.exec_immediate(self.conn, query)
        row_dict = ibm_db.fetch_assoc(stmt)
        results.append(row_dict)
        while row_dict is not False:
            row_dict = ibm_db.fetch_assoc(stmt)
            if row_dict is not False:
                results.append(row_dict)
        return results

    def query_df(self, query):
        """
        Args:
            query (str): this is the SQL query statement which is being used
            to get data from the database
        Returns:
            df (pandas dataframe): this is the dataframe version of the
            returned table from the sql query
        """
        df = pd.DataFrame(self.query_db(query))
        if 'date_time' in df.columns:
            df.set_index('date_time', inplace=True)
        df = df.astype(float)
        return df
