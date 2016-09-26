import argparse
import sys
import os
import gpxpy
# had to clone the latest github version of gpxfile to get it to work:
# pip install git+https://github.com/tkrajina/gpxpy.git
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D

def plot_files(lats, lons, alts, names):
    # make plot
    lat = [item for sublist in lats for item in sublist]
    lon = [item for sublist in lats for item in sublist]
    mapb = Basemap(projection='gall', resolution='l',
                   urcrnrlat=max(lat)+.01,
                   llcrnrlon=max(lon)+.01,
                   llcrnrlat=min(lat)-.01,
                   urcrnrlon=min(lon)-.01,
                   )
    colors = plt.cm.rainbow(np.linspace(0, 1, len(lats)))
    fig = plt.figure(figsize=(12,8))
    ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection='3d')
    for xx in range(len(lats)):
        lat = lats[xx]
        lon = lons[xx]
        alt = alts[xx]
        name = names[xx]
        yu, xu = mapb(np.asarray(lon),
                   np.asarray(lat))
        #mapb.scatter(yu, xu, c=alt, s=3, edgecolor="None", alpha=0.5)
        #parallels = np.linspace(min(lat),max(lat),5)
        #mapb.drawparallels(parallels,labels=[True, False, False, False],fontsize=10)

        #meridians = np.linspace(min(lon),max(lon),5)
        #mapb.drawmeridians(meridians,labels=[False, True, False, False],fontsize=10)
        ax.scatter(yu, xu, alt, edgecolor="None", c=colors[xx], label=name)
        #plt.colorbar()
    plt.legend()
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
                print(point)
    return lat, lon, alt

if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description="Read gpx")
    aparser.add_argument('--f', dest='input_files',
                         help="file to read data from", nargs="+")
    aparser.add_argument('--g', dest='glob', default='',
                         help="recursive glob search for files", type=str)
    #parser.add_argument('output_file', help="file to write data to")
    # parse command line
    try:
        args = aparser.parse_args()
    except :
        aparser.print_help()
        sys.exit()
    ifiles = []
    if len(args.glob):
        # do glob search based on paramters
        if str(args.glob) == 'thesis':
            print("searching thesis space")
            g = '/Volumes/johannah_external/thesis-work/201511_sea_state_DRI_Sikululiaq/uas_data/*/n2/log/*.gpx'
        else:
            g = args.glob
        ifiles = sorted(glob(g))
    else:
        ifiles = args.input_files

    print(ifiles)
    if not len(ifiles):
        print("no valid files found")
        aparser.print_help()
        sys.exit()
    lats = []
    lons = []
    alts= []
    cs = []
    names = []
    for xx, ifile in enumerate(ifiles):
        if not os.path.exists(ifile):
            print("Error: input file, %s does not exist" %ifile)
        else:
            if 1:
                lat,lon,alt = parse_gpx(ifile)
                lats.append(lat)
                lons.append(lon)
                alts.append(alt)
                names.append(os.path.split(ifiles[xx])[1])
    if len(lats):
        plot_files(lats, lons, alts, names)

