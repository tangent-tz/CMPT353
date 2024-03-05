import pandas as pd
import numpy as np
import sys
from pykalman import KalmanFilter
import xml.etree.ElementTree as ET


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def read_file(filename1):
    tree = ET.parse(filename1)
    return tree.getroot()

def create_dataframe_from_xml(root):
    data = []
    for trkpt in root.iter('{http://www.topografix.com/GPX/1/0}trkpt'):
        lat = float(trkpt.attrib['lat'])
        lon = float(trkpt.attrib['lon'])
        data.append({'lat': lat, 'lon': lon})
    return pd.DataFrame(data)


"""ref:https://stackoverflow.com/questions/69284105/distance-between-two-places-based-on-longitude-and-latitude"""
def distance(points):
    earth_radius = 6371000 #in meters
    lat1 = np.radians(points['lat'])
    lon1 = np.radians(points['lon'])
    lat2 = lat1.shift(-1)
    lon2 = lon1.shift(-1)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c.sum()


def smooth(points):
    initial_state = points.iloc[0]
    observation_covariance = np.diag([2.4, 2.4]) ** 2
    transition_covariance = np.diag([1.15, 1.15]) ** 2
    transition = [[1, 0], [0, 1]]
    kf = KalmanFilter(initial_state_mean=initial_state,
                      initial_state_covariance=observation_covariance,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance,
                      transition_matrices=transition)
    kalman_smoothed, _ = kf.smooth(points)
    return pd.DataFrame(kalman_smoothed, columns=points.columns)


def main():
    root = read_file(sys.argv[1])
    points = create_dataframe_from_xml(root)
    print('Unfiltered distance: %0.2f' % (distance(points),))

    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
