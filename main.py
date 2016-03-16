"""
main.py - Part of millennium-compact-groups package

Use a clustering algorithm to find compact groups in the Millennium
simulation.

Copyright(C) 2016 by
Trey Wenger; tvwenger@gmail.com
Chris Wiens; cdw9bf@virginia.edu
Kelsey Johnson; kej7a@virginia.edu

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

14 Mar 2016 - TVW Finalized version 1.0
"""
_PACK_NAME = 'millennium-compact-groups'
_PROG_NAME = 'main.py'
_VERSION = 'v1.0'

# System utilities
import os
import argparse
import time
import traceback
# Numerical utilities
import numpy as np
# Other utilities
import multiprocessing as mp
import ipyparallel as ipp
import itertools
# Classes for this project
import millennium_query
import cg_logger
import worker

def run_worker(w):
    """
    Downloads this simulation chunk, performs clustering and
    other analysis
    """
    try:
        start_time = time.time()
        w.logger.log("Working on snapnum: {0}, Box {1}, {2}, {3}".\
            format(w.snapnum,w.xbounds[0],w.ybounds[0],w.zbounds[0]))
        # If the member file and group file for this chunk already
        # exists and we are not overwriting, we're done!
        if ((os.path.exists(w.groupsfile) and os.path.exists(w.membersfile) and
            os.path.exists(w.all_groupsfile) and os.path.exists(w.all_membersfile)
            and not w.overwrite)):
            w.logger.log('Found {0} and {1}. Returning.'.format(w.groupsfile,w.membersfile))
            return
        #
        # If data exists and we're asked to overwrite it, or if data
        # doesn't exists, download it
        #
        if ((os.path.exists(w.datafile) and w.get_data) or
            (not os.path.exists(w.datafile))):
            w.logger.log('Downloading data.')
            w.download_data()
            w.logger.log('Done.')
        #
        # Read in the data
        #
        w.logger.log('Reading data.')
        w.read_data()
        w.logger.log('Done.')
        if len(w.data) == 0:
            # No galaxies found. We're done!
            w.logger.log('No galaxies found. Returning.')
            return
        #
        # Perform clustering
        #
        if (w.cluster or not os.path.exists(w.clusterfile)):
            w.logger.log('Performing clustering.')
            if w.use_dbscan:
                w.logger.log('Using DBSCAN.')
                w.dbscan()
            else:
                w.logger.log('Using MeanShift.')
                w.mean_shift()
            w.logger.log('Done.')
        #
        # Read cluster results
        #
        w.logger.log('Reading cluster results.')
        w.read_cluster()
        w.logger.log('Done.')
        # num-1 here accounts for the "-1" ungrouped cluster
        w.logger.log('Found {0} clusters.'.format(w.num_clusters-1))
        #
        # Measure the discovered groups' properties
        #
        w.logger.log('Analyzing groups.')
        w.analyze_groups()
        w.logger.log('Done.')
        #
        # Filter groups 
        #
        w.logger.log('Filtering groups.')
        w.filter_groups()
        w.logger.log('Done.')
        #
        # Save the group and member statistics
        #
        w.logger.log('Saving group and member statistics.')
        w.save()
        w.logger.log('Done.')
        # calculate run-time
        time_diff = time.time() - start_time
        hours = int(time_diff/3600.)
        mins = int((time_diff - hours*3600.)/60.)
        secs = time_diff - hours*3600. - mins*60.
        w.logger.log("Runtime: {0}h {1}m {2:.2f}s".format(hours,mins,secs))
    except Exception as e:
        w.logger.log("Caught exception in snapnum: {0}, Box {1},{2},{3}".\
            format(w.snapnum,w.xbounds[0],w.ybounds[0],w.zbounds[0]))
        traceback.print_exc()
        raise e

def main(username,password,
         get_data=False,snapnums=range(64),size=100.,
         cluster=False,
         use_dbscan=False,neighborhood=0.05,bandwidth=0.1,
         min_members=3,dwarf_range=3.0,crit_velocity=1000.,
         annular_radius=1.,max_annular_mass_ratio=0.0001,min_secondtwo_mass_ratio=0.1,
         num_cpus=1,profile=None,
         outdir='results',overwrite=False,
         verbose=False,nolog=False,test=False):
    """
    Set up workers to download simulation chunks, perform clustering,
    and calculate group and member statistics  
    """
    start_time = time.time()
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    #
    # Handle test case
    #
    if test:
        snapnums = range(50,61)
        size = 100.
    #
    # Open main log file
    #
    logfile = os.path.join(outdir,'log_{0}.txt'.format(time.strftime('%Y-%m-%d_%H-%M-%S')))
    logger = cg_logger.Logger(logfile,nolog=nolog,verbose=verbose)
    logger.log("Using the following parameters:")
    logger.log("username: {0}".format(username))
    logger.log("get_data: {0}".format(get_data))
    logger.log("snapnums: {0}".format(snapnums))
    logger.log("size: {0}".format(size))
    logger.log("cluster: {0}".format(cluster))
    logger.log("use_dbscan: {0}".format(use_dbscan))
    logger.log("neighborhood: {0}".format(neighborhood))
    logger.log("bandwidth: {0}".format(bandwidth))
    logger.log("min_members: {0}".format(min_members))
    logger.log("dwarf_range: {0}".format(dwarf_range))
    logger.log("crit_velocity: {0}".format(crit_velocity))
    logger.log("annular_radius: {0}".format(annular_radius))
    logger.log("max_annular_mass_ratio: {0}".format(max_annular_mass_ratio))
    logger.log("min_secondtwo_mass_ratio: {0}".format(min_secondtwo_mass_ratio))
    logger.log("num_cpus: {0}".format(num_cpus))
    logger.log("profile: {0}".format(profile))
    logger.log("outdir: {0}".format(outdir))
    logger.log("overwrite: {0}".format(outdir))
    logger.log("verbose: {0}".format(verbose))
    logger.log("test: {0}".format(test))
    #
    # Set up output directories
    #
    for snapnum in snapnums:
        directory = os.path.join(outdir,"snapnum_{0:02}".\
                                 format(snapnum))
        if not os.path.isdir(directory):
            os.mkdir(directory)
            logger.log('Created {0}'.format(directory))
        data_directory = os.path.join(directory,'data')
        if not os.path.isdir(data_directory):
            os.mkdir(data_directory)
            logger.log('Created {0}'.format(data_directory))
        members_directory = os.path.join(directory,'members')
        if not os.path.isdir(members_directory):
            os.mkdir(members_directory)
            logger.log('Created {0}'.format(members_directory))
        groups_directory = os.path.join(directory,'groups')
        if not os.path.isdir(groups_directory):
            os.mkdir(groups_directory)
            logger.log('Created {0}'.format(groups_directory))
    #
    # Get Millennium Simulation cookies
    #
    cookies = millennium_query.get_cookies(username,password)
    logger.log('Acquired login cookies')
    #
    # Set up simulation chunk boundaries
    #
    if test:
        mins = np.array([0])
    else:
        mins = np.arange(0,500,size)
    maxs = mins + size
    #
    # adjust mins and maxs to overlap by annular_radius, but do not
    # go beyond simulation boundaries
    #
    mins = mins - annular_radius
    mins[mins < 0.] = 0.
    maxs = maxs + annular_radius
    maxs[maxs > 500.] = 500.
    boundaries = zip(mins,maxs)
    #
    # Set up worker pool
    #
    args = []
    for snapnum,xbounds,ybounds,zbounds in \
      itertools.product(snapnums,boundaries,boundaries,boundaries):
        # Set-up a new Worker
        arg = worker.Worker(username,password,cookies,
                            snapnum,xbounds,ybounds,zbounds,
                            get_data=get_data,cluster=cluster,
                            use_dbscan=use_dbscan,neighborhood=neighborhood,bandwidth=bandwidth,
                            min_members=min_members,dwarf_range=dwarf_range,
                            crit_velocity=crit_velocity,annular_radius=annular_radius,
                            max_annular_mass_ratio=max_annular_mass_ratio,
                            min_secondtwo_mass_ratio=min_secondtwo_mass_ratio,
                            outdir=outdir,overwrite=overwrite,
                            verbose=verbose,nolog=nolog)
        # Append to list of worker arguments
        args.append(arg)
        logger.log('Created worker for snapnum: {0}, xmin: {1}, ymin: {2}, zmin: {3}'.format(snapnum,xbounds[0],ybounds[0],zbounds[0]))
    logger.log("Found {0} tasks to process".format(len(args)))
    #
    # Set up IPython.parallel
    #
    if profile is not None:
        logger.log("Using IPython.parallel")
        rc = ipp.Client(profile=profile,block=False)
        pool = rc.load_balanced_view()
        pool.block = False
        logger.log("Found {0} tasks available to run simultaneously".\
              format(len(pool)))
        jobs = pool.map(run_worker,args)
        while not jobs.ready()
            time.sleep(1)
    #
    # Set up multiprocessing
    #
    elif num_cpus > 1:
        logger.log("Using multiprocessing with {0} cpus".format(num_cpus))
        pool = mp.Pool(num_cpus)
        jobs = pool.map_async(run_worker,args)
        pool.close()
        pool.join()
    #
    # One job at a time
    #
    else:
        logger.log("Not using parallel processing.")
        for arg in args:
            run_worker(arg)
    logger.log("All jobs done.")
    #
    # Clean up
    #
    # calculate run-time
    time_diff = time.time() - start_time
    hours = int(time_diff/3600.)
    mins = int((time_diff - hours*3600.)/60.)
    secs = time_diff - hours*3600. - mins*60.
    logger.log("Runtime: {0}h {1}m {2:.2f}s".format(hours,mins,secs))

#=====================================================================
# Command Line Arguments
#=====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find Compact Groups in Full Millenium Simulation",
        prog=_PROG_NAME,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    # Required Parameters
    #
    parser.add_argument('--username',type=str,required=True,
                        help="Username for Millennium Simulation access.")
    parser.add_argument('--password',type=str,required=True,
                        help="Password for Millennium Simulation access")
    #
    # Simulation parameters
    #
    parser.add_argument('--get_data',action='store_true',
                        help='Re-download simulation data even if it already exists.')
    parser.add_argument('--snapnums',nargs="+",type=int,
                        default=range(64),
                        help="snapnums to process. Default: All (0 to 63)")
    parser.add_argument('--size',type=int,
                        default=100,
                        help="Simulation chunk cube side length in Mpc/h. Default: 100")
    #
    # Clustering parameters
    #
    parser.add_argument('--cluster',action='store_true',
                        help='Re-do clustering even if clustering output already exists.')
    parser.add_argument('--use_dbscan',action='store_true',
                        help='If set, use DBSCAN for clustering. Default: MeanShift')
    parser.add_argument('--neighborhood',type=float,default=0.05,
                        help='Neighborhood parameter for DBSCAN. Default 0.05')
    parser.add_argument('--bandwidth',type=float,default=0.1,
                        help='Bandwidth parameter for MeanShift. Default 0.1')
    #
    # Filter parameters
    #
    parser.add_argument('--min_members',type=int,default=3,
                        help='Minimum members to be considered a group. Default: 3')
    parser.add_argument('--dwarf_range',type=float,default=3.0,
                        help=('Magnitude difference from brightest '
                              'group galaxy to be considered a dwarf. '
                              'Default: 3.0'))
    parser.add_argument('--crit_velocity',type=float,default=1000.0,
                        help=('Velocity difference (km/s) between a '
                              'galaxy and median group velocity to '
                              'exclude (i.e. high-velocity fly-bys). '
                              'Default: 1000.0'))
    parser.add_argument('--annular_radius',type=float,default=1.0,
                        help=('Size (in Mpc/h) of outer annular radius '
                              'for annular mass ratio calculation. Default: 1.0'))
    parser.add_argument('--max_annular_mass_ratio',type=float,default=0.0001,
                        help=('Maximum allowed value for the ratio of mass '
                              'in annulus to total mass. Default: 0.0001'))
    parser.add_argument('--min_secondtwo_mass_ratio',type=float,default=0.1,
                        help=('Minimum allowed value for the ratio of mass '
                              'of the second two most massive galaxies to '
                              ' the most massive galaxy. Default: 0.1'))
    #
    # Multiprocessing parameters
    #
    parser.add_argument('--num_cpus',type=int,default=1,
                        help=("Number of cores to use with "
                              "multiprocessing (not "
                              "IPython.parallel). Default: 1"))
    parser.add_argument('--profile',type=str,default=None,
                        help=("IPython profile if running on computing "
                              "cluster using IPython.parallel. "
                              "Default: None (use multiprocessing "
                              "on single machine)"))
    #
    # Data output parameters
    #
    parser.add_argument('--outdir',type=str,default='results',
                        help="directory to save results. Default: results/")
    parser.add_argument('--overwrite',action='store_true',
                        help='Re-do analysis if member file and group file exists.')
    #
    # Other
    #
    parser.add_argument('--verbose',action='store_true',
                        help='Output messages along the way.')
    parser.add_argument('--nolog',action='store_true',
                        help="Do not save log files")
    parser.add_argument('--test',action='store_true',
                        help="Run a test on one chunk. (snapnum=50-60, box=0,0,0, size=100)")
    #
    # Parse the arguments and send to main function
    #
    args = parser.parse_args()
    main(args.username,args.password,
         get_data=args.get_data,snapnums=args.snapnums,size=args.size,
         cluster=args.cluster,
         use_dbscan=args.use_dbscan,neighborhood=args.neighborhood,bandwidth=args.bandwidth,
         min_members=args.min_members,dwarf_range=args.dwarf_range,
         crit_velocity=args.crit_velocity,annular_radius=args.annular_radius,
         max_annular_mass_ratio=args.max_annular_mass_ratio,min_secondtwo_mass_ratio=args.min_secondtwo_mass_ratio,
         num_cpus=args.num_cpus,profile=args.profile,
         outdir=args.outdir,overwrite=args.overwrite,
         verbose=args.verbose,nolog=args.nolog,test=args.test)
