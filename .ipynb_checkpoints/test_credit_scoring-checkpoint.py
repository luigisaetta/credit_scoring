#
# OCI Data Science Model Deployment Client
# Author: L. Saetta
# version 2.0
# date: 14/03/2022
#
import numpy as np
import pandas as pd
import json
import requests
import time

import oci
from oci.config import validate_config
from oci.signer import Signer

# OCI access Configuration
# you need to put your ids and d path-to-key here
#
config = {
    "user": "ocid1.user.oc1..aaaaaaaay2uji3bd57h7mdldpmd7hzvnej54woyv3t6vt4ascty23khdr7sq",
    "key_file": "/Users/lsaetta/Progetti/credit_scoring/api_keys/priv.pem",
    "fingerprint": "22:77:05:e7:57:b6:9d:22:93:57:b9:b3:28:56:e1:99",
    "tenancy": "ocid1.tenancy.oc1..aaaaaaaa2g6eprhmvggcfkafgur3c65unb4lknevw2sukxywlxw7oggo4uua",
    # region is not really requested because you use the endpoint, but better to have home region in config
    "region": "eu-frankfurt-1",
    "log_requests": False
}
# Config for Data Science
# this is the url of our Model Deployment
url = "https://modeldeployment.us-ashburn-1.oci.customer-oci.com/"\
      + "ocid1.datasciencemodeldeployment.oc1.iad.amaaaaaa3mg5kzaaapwt6jn5xov2v4ye35jbxdqcc332x6e2vfo7bzice3yq/predict"


# inspired from OCI Python SDK GitHub
def make_prediction(p_auth, input_body):
    print('')
    print("*** Making predictions ...")
    print('')
    print('Making predictions on ', len(input_data), 'samples')
    print('')

    try:
        if config['log_requests']:
            print('The body request is:', json.dumps(input_body, indent=1))
            print('')

        # be careful: here you have to use the **json** parameter to pass input data, not the **data** parameter
        # at least for my service... check with other implementation, use service logs for debug
        prediction = requests.post(url, json=json.dumps(input_body), auth=p_auth)

        if prediction.status_code == 200:
            print('Model predictions:')

            # extract vector
            results = prediction.json()['prediction']
            print(results)
            print('')
        else:
            print("Making predictions failed !!!")
            print('HTTP Status code:', prediction.status_code)

            # if error exit, no reason to continue
            return

    except Exception as e:
        print("Making predictions failed!!!")
        raise e

    return results


# take data for test from cs-test.csv file
# the two parameters are start and end index for selecting which rows to use
def prepare_input_data(p_start, p_end):
    file_test = './cs-test.csv'

    features = ['RevolvingUtilizationOfUnsecuredLines', 'age',
                'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfDependents']

    # last parameter to avoid too many decimal digits
    # it's a problems with float in Pandas
    dati_test = pd.read_csv(file_test, float_precision='round_trip')

    dati_test = dati_test[features]

    # check and fix missing values here
    # NAN is substituted with the MODE (0, 5400)
    condition = dati_test.isna()['NumberOfDependents']
    dati_test.loc[condition, 'NumberOfDependents'] = 0

    condition = dati_test.isna()['MonthlyIncome']
    dati_test.loc[condition, 'MonthlyIncome'] = 5400

    return dati_test[p_start:p_end].values


if __name__ == "__main__":

    print('')
    print('*** Test calls to OCI Model Deployment endpoint...')
    print('')

    tStart = time.time()

    validate_config(config)

    print('Validate Config OK')

    # prepare for signing the API request
    auth = Signer(tenancy=config['tenancy'], user=config['user'], fingerprint=config['fingerprint'],
                  private_key_file_location=config['key_file'])

    # if you enable HTTP log (see config)
    if config['log_requests']:
        oci.base_client.is_http_log_enabled(True)

    # prepare input data
    # the two parameters are start and stop positions in the numpy array containing test data
    start = 200
    end = 400
    input_data = prepare_input_data(start, end)

    # invoke the deployed model and prints output
    preds = make_prediction(auth, input_data.tolist())

    if preds is not None:
        # let's see what cases are positive
        print('*** Highligthing only POSITIVE cases:')
        print('')

        # extract only positive cases
        list_pos = [i for i, val in enumerate(preds) if val == 1]

        np.set_printoptions(suppress=True, formatter={'float_kind': '{:10.7f}'.format},
                            linewidth=120)

        print('Found', len(list_pos), 'positive cases')
        print('')

        for i in list_pos:
            print('Case number: ', start + i)
            print('Features are:', input_data[i])
            print('')
        print('')
    else:
        exit_status = -1

    tEla = time.time() - tStart
    print('Total elapsed time:', round(tEla, 1), 'sec.')

    exit(-1)