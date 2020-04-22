import numpy as np
import random
import pandas as pd
import requests
import json

# Linearly takes x on [a,b] to [c,d]
def linearmap(ab, cd, x):
    a,b = ab[0],ab[1]
    c,d = cd[0],cd[1]
    return ((d-c)*x +(b*c - d*a))/(b - a)

# Gathering data from NASA Insight weather API
# Pass value of True for line graph data; False for windrose data
def gather_data(choice):
    api_key = "AkPFZ2oNwdefoiJ9R94G6JglHEj79MrqLPWBihVf"
    nasa_data = requests.get("https://api.nasa.gov/insight_weather/?api_key=" + api_key + "&feedtype=json&ver=1.0").json()
    sols = nasa_data["sol_keys"]

    # Data for line charts
    line_time = []
    line_temp = []
    line_pressure = []
    line_windspeed = []

    # Data for windrose chart
    rose_sol = []
    rose_solHour = []
    rose_windcount = []
    rose_winddirection = []

    # Loop through each sol to gather data for said sol
    for i in range(len(nasa_data["sol_keys"])-1):
        line_time.append(int(nasa_data["sol_keys"][i]))
        line_temp.append(float(nasa_data[str(line_time[i])]['AT']['av']))
        line_pressure.append(float(nasa_data[str(line_time[i])]['PRE']['av']))
        line_windspeed.append(float(nasa_data[str(line_time[i])]['HWS']['av']))
        # Loop through hourly data of sol for windrose data
        for j in range(len(nasa_data[str(line_time[i])]['WD'])-1):
            tempHours = list(nasa_data[str(line_time[i])]['WD'].keys()) # Gathers keys of hours from sol to use later for access
            rose_sol.append(int(nasa_data["sol_keys"][i]))
            rose_solHour.append(tempHours[j])
            rose_windcount.append(int(nasa_data[str(line_time[i])]['WD'][tempHours[j]]['ct']))
            rose_winddirection.append(int(nasa_data[str(line_time[i])]['WD'][tempHours[j]]['compass_degrees']))

    new_line_data = pd.DataFrame(data={'time':line_time,
                                  'temp':line_temp,
                                  'pressure':line_pressure,
                                  'windspeed':line_windspeed})
    new_rose_data = pd.DataFrame(data={'sol':rose_sol,
                                  'solHour':rose_solHour,
                                  'windcount':rose_windcount,
                                  'winddirection':rose_winddirection})

    old_line_data = pd.read_csv('line.csv').round(3)
    old_rose_data = pd.read_csv('rose.csv').round(3)

    join_line_data = pd.concat([new_line_data, old_line_data]).drop_duplicates().round(3).sort_values(by=['time']).reset_index(drop=True)
    join_rose_data = pd.concat([new_rose_data, old_rose_data]).drop_duplicates().round(3).sort_values(by=['sol']).reset_index(drop=True)

    join_line_data.to_csv(r'line.csv', index = False)
    join_rose_data.to_csv(r'rose.csv', index = False)

    if choice == True:
        return join_line_data
    else:
        return join_rose_data


# Generating temporary placeholder data; to be replaced!
def placeholder_data():
    time = np.linspace(5, 25, 200)
    temp = 200. + 30.*np.sin(time) + np.random.normal(0, .8, 200) + 0.04*time
    pressure = .015 + 0.01*np.sin(time + 1.5) + np.random.normal(0, .0008, 200) + 0.0004*time
    windspeed = np.random.normal(0, 1.6, len(time))
    for i in range(np.random.randint(2,6)):
        windspeed += random.randrange(3, 50)*np.exp(-(time - random.randrange(0, 20))**2/4.)

    winddirection = np.zeros_like(time)
    d = 0.
    for i in range(0, 200):
        winddirection[i] += d
        d += np.random.normal(0, 1.5)

    max_dir = np.max(winddirection) # a
    min_dir = np.min(winddirection) # b
    max_angle = 360.
    min_angle = 0.

    winddirection = linearmap([min_dir,max_dir],[0., 360], winddirection)

    data = pd.DataFrame(data={'time':time,
                              'temp':temp,
                              'pressure':pressure,
                              'windspeed':windspeed,
                              'winddirection':winddirection})
    return data
