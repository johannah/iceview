import argparse
import sys
import os
import gpxpy
# had to clone the latest github version of gpxfile to get it to work:
# pip install git+https://github.com/tkrajina/gpxpy.git
import numpy as np
import pandas as pd

def plot_latlon(lat, lon, alt):
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    #from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12,8))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')
    #ax.set_axis_off()

    #fig.add_axes(ax)
    print('Extents', max(lat),
          max(lon),
          min(lat),
          min(lon))
    mapb = Basemap(projection='gall', resolution='l',
                   urcrnrlat=max(lat)+.01,
                   llcrnrlon=max(lon)+.01,
                   llcrnrlat=min(lat)-.01,
                   urcrnrlon=min(lon)-.01,
                   )

    yu, xu = mapb(np.asarray(lon),
               np.asarray(lat))
    mapb.scatter(yu, xu, c=alt, s=3, edgecolor="None", alpha=0.5)
    parallels = np.linspace(min(lat),max(lat),5)
    mapb.drawparallels(parallels,labels=[True, False, False, False],fontsize=10)

    meridians = np.linspace(min(lon),max(lon),5)
    # not sure why meridians is not showing up in plot
    mapb.drawmeridians(meridians,labels=[False, True, False, False],fontsize=10)
    plt.colorbar()
    plt.show()

def parse_gpx(gpxfile):
    gpxfo = open(gpxfile, 'r')
    gpx = gpxpy.parse(gpxfo)
    lat = []
    lon = []
    t = []
    alt = []
    yaw = []
    pitch = []
    roll = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                lat.append(point.latitude)
                lon.append(point.longitude)
                alt.append(point.elevation)
    plot_latlon(lat,lon,alt)

if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description="Read gpx")
    aparser.add_argument('input_file', help="file to read data from")
    #parser.add_argument('output_file', help="file to write data to")
    # parse command line
    try:
        args = aparser.parse_args()
    except :
        aparser.print_help()
        sys.exit()

    ifile = args.input_file

    if not os.path.exists(ifile):
        print("Error: input file, %s does not exist" %ifile)
        sys.exit()
    gpxdata = parse_gpx(ifile)
