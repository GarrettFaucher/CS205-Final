import numpy as np
import random
import pandas as pd
import requests
import json
import datetime

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
    for i in range(len(nasa_data["sol_keys"])):
        line_time.append(int(nasa_data["sol_keys"][i]))
        line_temp.append(float(nasa_data[str(line_time[i])]['AT']['av']))
        line_pressure.append(float(nasa_data[str(line_time[i])]['PRE']['av']))
        line_windspeed.append(float(nasa_data[str(line_time[i])]['HWS']['av']))
        # Loop through hourly data of sol for windrose data
        for j in range(len(nasa_data[str(line_time[i])]['WD'])-1):
            tempHours = list(nasa_data[str(line_time[i])]['WD'].keys()) # Gathers keys of hours from sol to use later for access
            rose_sol.append(int(nasa_data["sol_keys"][i]))
            rose_solHour.append(int(tempHours[j]))
            rose_windcount.append(int(nasa_data[str(line_time[i])]['WD'][tempHours[j]]['ct']))
            rose_winddirection.append(int(nasa_data[str(line_time[i])]['WD'][tempHours[j]]['compass_degrees']))

    new_line_data = pd.DataFrame(data={'time':      line_time,
                                       'temp':      line_temp,
                                       'pressure':  line_pressure,
                                       'windspeed': line_windspeed})
    new_rose_data = pd.DataFrame(data={'sol':       rose_sol,
                                       'solHour':   rose_solHour,
                                       'windcount': rose_windcount,
                                       'winddirection':rose_winddirection})

    old_line_data = pd.read_csv('/users/t/s/tsheboy/www-root/CS205-Final/line.csv').round(3)
    old_rose_data = pd.read_csv('/users/t/s/tsheboy/www-root/CS205-Final/rose.csv').round(3)

    # Sorting data and dropping dupes for usage
    join_line_data = pd.concat([new_line_data, old_line_data]).drop_duplicates(subset=['time']).round(3).sort_values(by=['time']).reset_index(drop=True)
    join_rose_data = pd.concat([new_rose_data, old_rose_data]).sort_values(by=['sol', 'solHour']).reset_index(drop=True).drop_duplicates(subset=['sol','solHour']).reset_index(drop=True)

    # Storing as csv for plot generation and historical data
    join_line_data.to_csv(r'/users/t/s/tsheboy/www-root/CS205-Final/line.csv', index = False)
    join_rose_data.to_csv(r'/users/t/s/tsheboy/www-root/CS205-Final/rose.csv', index = False)

    # Store current season for website purposes
    season = str(nasa_data[str(line_time[0])]['Season'])
    seasonFile = open("/users/t/s/tsheboy/www-root/CS205-Final/season.txt", "w")
    seasonFile.write(season[0].upper() + season[1:])
    seasonFile.close()

    # Store todays date for website purposes
    date = datetime.datetime.today()
    timeFile = open("/users/t/s/tsheboy/www-root/CS205-Final/time.txt", "w")
    timeFile.write(str(date)[0:10] + " at " + str(date)[11:16])
    timeFile.close()

    # Store latest sol for website purposes
    sol = str(str(line_time[6]))
    solFile = open("/users/t/s/tsheboy/www-root/CS205-Final/sol.txt", "w")
    solFile.write(sol)
    solFile.close()

    if choice == True:
        return join_line_data
    else:
        return join_rose_data


# Generating temporary placeholder data; to be replaced!
def placeholder_data(choice):
    if choice:
        time = np.linspace(5, 25, 200)
        temp = 200. + 30.*np.sin(time) + np.random.normal(0, .8, 200) + 0.04*time
        pressure = .015 + 0.01*np.sin(time + 1.5) + np.random.normal(0, .0008, 200) + 0.0004*time
        windspeed = np.random.normal(0, 1.6, len(time))
        for i in range(np.random.randint(2,6)):
            windspeed += random.randrange(3, 50)*np.exp(-(time - random.randrange(0, 20))**2/4.)
        data = pd.Dataframe(data={'time':time,
                                  'temp':temp,
                                  'pressure':pressure,
                                  'windspeed':windspeed})
        return data
    else:

        n_sol = 20
        n_hour = 23

        winddirection = np.zeros(n_sol*n_hour)
        d = 0.
        for i in range(0, len(winddirection)):
            winddirection[i] += d
            d += np.random.normal(0, 1.5)

        sol = np.zeros_like(winddirection)
        solHour = np.zeros_like(winddirection)
        windcount = np.zeros_like(winddirection)
        for i in range(n_sol):
            sol[i*n_hour:(i+1)*n_hour] = i*np.ones(n_hour)
            solHour[i*n_hour:(i+1)*n_hour] = np.arange(0, n_hour)
            windcount[i*n_hour:(i+1)*n_hour] = np.random.randint(0,5000,n_hour)

        max_dir = np.max(winddirection)
        min_dir = np.min(winddirection)
        max_angle = 360.
        min_angle = 0.

        winddirection = linearmap([min_dir,max_dir],[0., 360], winddirection)

        data = pd.DataFrame(data={'sol':sol,
                                  'solHour':solHour,
                                  'windcount':windcount,
                                  'winddirection':winddirection})

        return data
