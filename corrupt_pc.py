from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng
rng = default_rng(0)

original_vals = False
if original_vals == True:
    num_points = 1000
    x_mu = 0
    x_sigma = 20.0
    y_mu = 0
    y_sigma = 20.0
    z_mu = 0
    z_sigma = 0.5
    intensity_mu = 0.5
    intensity_sigma = 0.11
else:
    num_points = 10000
    x_mu = 0
    x_sigma = 8.0
    y_mu = 0
    y_sigma = 8.0
    z_mu = 0.1
    z_sigma = 0.3
    intensity_mu = 0.5
    intensity_sigma = 0.11
    intensity_mu = 0.0002
    intensity_sigma = 0.00004

# Creating dataset
x_vals = rng.normal(x_mu, x_sigma, num_points)
y_vals = rng.normal(y_mu, y_sigma, num_points)
z_vals = rng.normal(z_mu, z_sigma, num_points)
i_vals = rng.normal(intensity_mu, intensity_sigma, num_points)
# z = np.random.randint(100, size =(50))
# x = np.random.randint(80, size =(50))
# y = np.random.randint(60, size =(50))

corrupt_pc = np.stack([x_vals, y_vals, z_vals, i_vals], axis=1)
print('min intensity corrupt_pc', np.min(corrupt_pc[:, 3]))
print('max intensity corrupt_pc', np.max(corrupt_pc[:, 3]))
print('mean intensity corrupt_pc', np.mean(corrupt_pc[:, 3]))

# Load kitti point cloud
# kitti training 0 - 7480
lidar_file = '/home/matthew/git/cadc_testing/WISEOpenLidarPerceptron/data/kitti/training/velodyne/00' + '0008' + '.bin'
kitti_pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
print('min intensity kitti', np.min(kitti_pc[:, 3]))
print('max intensity kitti', np.max(kitti_pc[:, 3]))
print('mean intensity kitti', np.mean(kitti_pc[:, 3]))

lidar_file = '/home/matthew/git/cadc_testing/WISEOpenLidarPerceptron/data/cadc/2019_02_27/0070/labeled/lidar_points/data/'
# lidar_file = '/home/matthew/git/cadc_testing/WISEOpenLidarPerceptron/data/cadc/2019_02_27/0054/labeled/lidar_points/data/'
# lidar_file = '/home/matthew/git/cadc_testing/WISEOpenLidarPerceptron/data/cadc/2018_03_06/0001/labeled/lidar_points/data/'
cadc_pc = np.fromfile(str(lidar_file + '0000000000.bin'), dtype=np.float32).reshape(-1, 4)
print('min intensity cadc', np.min(cadc_pc[:, 3]))
print('max intensity cadc', np.max(cadc_pc[:, 3]))
print('mean intensity cadc', np.mean(cadc_pc[:, 3]))
# cadc_pc[:, 3] = cadc_pc[:, 3]*255 # mean of kitti
FILTER_LOW_INTENSITY_PTS = True

if FILTER_LOW_INTENSITY_PTS:
    new_cadc_pc = []
    print('original CADC number points', len(cadc_pc))
    for i in range(len(cadc_pc)):
        if cadc_pc[i, 3] > 0.02:
            continue
        if cadc_pc[i, 2] < -0.45: # 0.004
            continue
        if np.sqrt(np.sum(np.square(cadc_pc[i, :3]))) > 10.0:
            continue
        new_cadc_pc.append(cadc_pc[i])
    new_cadc_pc = np.array(new_cadc_pc)
    print('filtered CADC number points', len(new_cadc_pc))

    new_kitti_pc = []
    for i in range(len(kitti_pc)):
        if kitti_pc[i, 3] != 0.0:
            continue
        new_kitti_pc.append(kitti_pc[i])
    new_kitti_pc = np.array(new_kitti_pc)
print('min intensity cadc', np.min(cadc_pc[:, 3]))
print('max intensity cadc', np.max(cadc_pc[:, 3]))
print('mean intensity cadc', np.mean(cadc_pc[:, 3]))

# Replace points in kitti
mask = np.zeros(len(kitti_pc))
mask[0:len(corrupt_pc)] = 1
rng.shuffle(mask)
corrupt_pc_index = 0
for i in range(len(kitti_pc)):
    if mask[i] == 1:
        kitti_pc[i] = corrupt_pc[corrupt_pc_index]
        corrupt_pc_index += 1

# Intensity histogram
DISPLAY_HISTOGRAM = True
count0 = 0
if DISPLAY_HISTOGRAM:
    r_vals = []
    i_vals = []
    for i in range(len(kitti_pc)):
        r_val = np.sqrt(np.sum(np.square(kitti_pc[i, :3])))
        r_vals.append(r_val)
        if kitti_pc[i, 3] == 0.0:
            count0+= 1
        i_vals.append(kitti_pc[i, 3])
    i_vals = np.array(i_vals, dtype=np.float64)
    print('number of 0s', count0)
    print('number of points', len(i_vals))
    print('number of non zero points', len(i_vals) - count0)
    print('mean intensity of histogram', np.mean(i_vals))
    print('min intensity of histogram', np.min(i_vals))
    print('max intensity of histogram', np.max(i_vals))
    plt.hist2d(x=r_vals, y=i_vals, bins=50, cmap='hot', cmin = 1)
    plt.colorbar()
    plt.show()

# From OpenPCDet visualize_utils.py
def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig

import numpy as np
import pcl
import pcl.pcl_visualization
import sys, math

# View it.
from mayavi import mlab
# s = mlab.mesh(x, y, z)
# mlab.show()

print('cadc shape', cadc_pc.shape)
print('kitti shape', kitti_pc.shape)
print('corrupt pc shape', corrupt_pc.shape)
combined_pc = np.concatenate((kitti_pc, corrupt_pc), axis=0, dtype=np.float32)
print('min intensity combined_pc', np.min(combined_pc[:, 3]))
print('max intensity combined_pc', np.max(combined_pc[:, 3]))
print('mean intensity combined_pc', np.mean(combined_pc[:, 3]))
print(combined_pc.shape)
visualize_pts(kitti_pc, show_intensity=True)
mlab.view(azimuth=-180, elevation=10.0, distance=80.0, roll=90.0, focalpoint =[0,0,0])
visualize_pts(kitti_pc, bgcolor=(205/255, 205/255, 205/255), show_intensity=True)
mlab.view(azimuth=-180, elevation=10.0, distance=80.0, roll=90.0, focalpoint =[0,0,0])
visualize_pts(cadc_pc, show_intensity=True)
mlab.view(azimuth=-180, elevation=10.0, distance=80.0, roll=90.0, focalpoint =[0,0,0])
visualize_pts(new_cadc_pc, bgcolor=(205/255, 205/255, 205/255), show_intensity=True)
mlab.view(azimuth=-180, elevation=10.0, distance=80.0, roll=90.0, focalpoint =[0,0,0])
mlab.show()

def dror_filter(input_cloud):
    radius_multiplier_ = 3
    azimuth_angle_ = 0.16 # 0.04
    min_neighbors_ = 3
    k_neighbors_ = min_neighbors_ + 1
    min_search_radius_ = 0.04

    filtered_cloud_list = []

    # init. kd search tree
    kd_tree = input_cloud.make_kdtree_flann()

    # Go over all the points and check which doesn't have enough neighbors
    # perform filtering
    for p_id in range(input_cloud.size):
        x_i = input_cloud[p_id][0];
        y_i = input_cloud[p_id][1];
        range_i = math.sqrt(pow(x_i, 2) + pow(y_i, 2));
        search_radius_dynamic = \
            radius_multiplier_ * azimuth_angle_ * 3.14159265359 / 180 * range_i;

        if (search_radius_dynamic < min_search_radius_):
            search_radius_dynamic = min_search_radius_

        [ind, sqdist] = kd_tree.nearest_k_search_for_point(input_cloud, p_id, k_neighbors_)

        # Count all neighbours
        neighbors = -1 # Start at -1 since it will always be its own neighbour
        for val in sqdist:
            if math.sqrt(val) < search_radius_dynamic:
                neighbors += 1;

        # This point is not snow, add it to the filtered_cloud
        if (neighbors >= min_neighbors_):
            filtered_cloud_list.append(input_cloud[p_id]);
            # print(filtered_cloud_list)
    
    return pcl.PointCloud(np.array(filtered_cloud_list, dtype=np.float32))

def crop_cloud(input_cloud):
    clipper = input_cloud.make_cropbox()
    clipper.set_Translation(0,0,0) # tx,ty,tz
    clipper.set_Rotation(0,0,0) # rx,ry,rz
    min_vals = [-4,-4,-3,0] # x,y,z,s
    max_vals = [4,4,10,0] # x,y,z,s
    clipper.set_MinMax(min_vals[0], min_vals[1], min_vals[2], min_vals[3], \
        max_vals[0], max_vals[1], max_vals[2], max_vals[3])
    return clipper.filter()

# Convert lidar 2d array to pcl cloud
point_cloud = pcl.PointCloud()
# point_cloud.from_array(combined_pc[:,0:3])
point_cloud.from_array(cadc_pc[:,0:3])
# Crop the pointcloud to around autnomoose
# cropped_cloud = crop_cloud(point_cloud)
cropped_cloud = point_cloud
# Run DROR
filtered_cloud = dror_filter(cropped_cloud)
# Print number of snow points
number_snow_points = cropped_cloud.size - filtered_cloud.size
# CADC 0070 expect 1443
# CADC 0054 expect 739
print('cropped cloud pts', cropped_cloud.size)
print('filtered cloud pts', filtered_cloud.size)
print('number snow pts', number_snow_points)

cadc_filtered_pc = np.asarray(filtered_cloud)

visualize_pts(new_cadc_pc, show_intensity=False)
mlab.view(azimuth=-180, elevation=10.0, distance=80.0, roll=90.0, focalpoint =[0,0,0])
visualize_pts(cadc_filtered_pc, show_intensity=False)
mlab.view(azimuth=-180, elevation=10.0, distance=80.0, roll=90.0, focalpoint =[0,0,0])
mlab.show()
exit()
visual = pcl.pcl_visualization.CloudViewing()

# PointXYZ
visual.ShowMonochromeCloud(point_cloud, b'cloud')
# visual.ShowMonochromeCloud(cropped_cloud, b'cloud')
# visual.ShowMonochromeCloud(filtered_cloud, b'cloud')
# visual.ShowGrayCloud(ptcloud_centred, b'cloud')
# visual.ShowColorCloud(ptcloud_centred, b'cloud')
# visual.ShowColorACloud(ptcloud_centred, b'cloud')

v = True
while v:
    v = not(visual.WasStopped())