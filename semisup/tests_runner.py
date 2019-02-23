"""
This is our test runner file for learning to learn by association. It reads all the Json files in the param folder and
runs our training and evaluation according to each of them. It support adding new Json while the run is active as long
as it has a different name (it will run it after it finished the old Jsons).
by Amit Henig and Paz Ilan.
"""

import train_eval
import os
import json
import time
from tensorflow.python.platform import app

def main(_):
    startTime = time.time()
    params_path = './params'

    # Find all Json parameter files in params folder and run
    files_ran = []
    file_list = [f for f in os.listdir(params_path) if f.endswith('.json')]
    while len(file_list) > 0:
        for f in file_list:
            with open(params_path + '/' + f) as json_file:
                print('WORKING ON ' + f)
                # Get default param struct
                params = train_eval.getDefaultParams()

                # Overide params that are in the Json into run params
                badKeys = []
                json_params = json.load(json_file)
                if 'run_name' not in json_params.keys():
                    json_params['run_name'] = os.path.splitext(f)[0]
                for key in json_params.keys():
                    if key not in params.keys():
                        badKeys = badKeys + [key]
                    params[key] = json_params[key]

                # Run
                if len(badKeys) == 0:
                    train_eval.runM(params)
                else:
                    print('for ' + f + ', bad keys: ')
                    for key in badKeys:
                        print(key)
        files_ran += file_list
        file_list2 = [f for f in os.listdir(params_path) if f.endswith('.json')]
        file_list = set(file_list2)-set(files_ran)

    endTime = time.time()
    print('~~~~~~~~~~~~~ FINISHED ~~~~~~~~~~~~~  run time: ' + str(round(endTime-startTime)) + 'sec')



if __name__ == '__main__':
  app.run()