#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pprint

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))



PRENTICE_JAMES = enron_data["PRENTICE JAMES"]
COLWELL_WESLEY=enron_data['COLWELL WESLEY']
SKILLING_JEFFREY_K = enron_data['SKILLING JEFFREY K']

pprint.pprint(SKILLING_JEFFREY_K)
#
#
# count =0
# for x in enron_data:
#     if enron_data[x]["poi"] == 1:
#         count +=1
# print count
#

