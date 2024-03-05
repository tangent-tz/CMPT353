import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dd

import statsmodels.api as sm
from pykalman import KalmanFilter


def read_file(readfile):
    readfile = pd.read_csv(readfile)
    return readfile


def loess_smoothing(cpu_data):
    lowess = sm.nonparametric.lowess
    plt.figure(figsize=(12, 4))
    plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)
    timestamps = pd.to_datetime(cpu_data['timestamp'])
    loess_smoothed = lowess(cpu_data['temperature'], timestamps, frac=0.02)
    plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-')


def kalman_smoothing(cpu_data):
    kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([0.09, 0.09, 0.09, 0.09]) ** 2
    transition_covariance = np.diag([0.001, 0.001, 0.001, 0.001]) ** 2
    transition = [[0.97, 0.5, 0.2, -0.001], [0.1, 0.4, 2.2, 0], [0, 0, 0.95, 0], [0, 0, 0, 1]]
    kf = KalmanFilter(initial_state_mean=initial_state,
                      initial_state_covariance=observation_covariance,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance,
                      transition_matrices=transition)
    kalman_smoothed, _ = kf.smooth(kalman_data)
    plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')


def graphMisc(cpu_data):
    plt.legend(['CPU temperature', 'LOESS smoothing', 'Kalman smoothing'])
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('CPU temperature over time')
    plt.xticks(rotation=30)
    plt.grid()
    # remove milliseconds from x-axis
    plt.gca().xaxis.set_major_formatter(dd.DateFormatter('%Y-%m-%d'))
    # plot every 100 x-axis label
    plt.xticks(cpu_data['timestamp'][::400])
    # size of the x-axis text font
    plt.xticks(fontsize=6)
    plt.savefig('cpu.svg')

def main():
    file = sys.argv[1]
    cpu_data = read_file(file)
    loess_smoothing(cpu_data)
    kalman_smoothing(cpu_data)
    graphMisc(cpu_data)

if __name__ == '__main__':
    main()

