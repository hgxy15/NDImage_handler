#! /Share/app/anaconda3/bin/python 
# coding: utf-8
import numpy as np
import mrcfile
from optparse import OptionParser

def distance_point_to_line_segment(point, segment_start, segment_end):
    """Calculate the distance from a point to a line segment."""
    # Create vectors
    seg_vec = segment_end - segment_start
    pt_vec = point - segment_start

    # Project point_vec onto seg_vec using dot product
    proj = np.dot(pt_vec, seg_vec) / np.dot(seg_vec, seg_vec)
    
    if proj < 0:
        # Point projection is before segment_start
        closest_point = segment_start
    elif proj > 1:
        # Point projection is after segment_end
        closest_point = segment_end
    else:
        # Point projection is on the segment
        closest_point = segment_start + proj * seg_vec

    # Compute the distance from the point to the closest point
    distance = np.linalg.norm(point - closest_point)
    return distance, (point - closest_point)

def nearest_distance_to_polygon_sides(point, vertices):
    """Calculate the nearest distance from a 3D point to the sides of a polygon."""
    num_vertices = len(vertices)
    distances = []
    distance_path_vecs = []

    # Iterate over the edges of the polygon
    for i in range(num_vertices):
        # Define the current edge
        segment_start = vertices[i]
        segment_end = vertices[(i + 1) % num_vertices]

        # Calculate distance from the point to the current edge
        distance, closest_path_vec =  distance_point_to_line_segment(point, segment_start, segment_end)
        distances.append(distance)
        distance_path_vecs.append(closest_path_vec)        

    # Return the minimum distance
    return np.min(distances), distance_path_vecs[np.argmin(distances)]


def create_density_polygon(num_sides, side_length, side_width, thickness, grid_size, num_layers = 1, layer_spacing = 0):
    #Grid size is the size of the box in pixels, in tuple format (x,y,z)
    if layer_spacing == 0 and num_layers > 1:
        layer_spacing = grid_size[2] / (num_layers + 2)
    # Create an empty 3D grid
    grid = np.zeros(grid_size, dtype=np.float32)
    for i_layer in range(num_layers):
        curr_num_sides = num_sides[i_layer]
        curr_side_length = side_length[i_layer]
        curr_side_width = side_width[i_layer]
        curr_thickness = thickness[i_layer]
        #Calculate the radius of the polygon.
        radius = ((curr_side_length) / 2) / np.sin(np.pi / curr_num_sides)
        # Define the center of the current polygon.
        z_offset = (i_layer - (num_layers - 1) / 2.0) * layer_spacing
        center = np.asarray([(grid_size[0] - 1) / 2, (grid_size[0] - 1) / 2, z_offset + (grid_size[0] - 1) / 2], dtype=np.float32)

        # Calculate the vertices of the polygon in 2D
        angle = 2 * np.pi / curr_num_sides
        vertices = np.asarray([[center[0] + radius * np.cos(i * angle), center[1] + radius * np.sin(i * angle), center[2]] for i in range(curr_num_sides)])

        # Fill in the polygon density
        for z in range(grid_size[2]):
            for y in range(grid_size[1]):
                for x in range(grid_size[0]):
                    #The increment of the density in each voxel, contributed from the current layer of the polygon
                    #should be exponentially decayed as the distance from the polygon increases.
                    #The decay rate is determined by the thickness of the polygon.
                    nearest_distance, distance_path_vec = nearest_distance_to_polygon_sides(np.asarray([x, y, z], dtype=np.float32), vertices)
                    
                    plane_distance = np.linalg.norm(distance_path_vec[0:2])
                    z_distance = distance_path_vec[2]
                    if plane_distance == 0:
                        curr_decay_angle = np.pi / 2
                    else:
                        curr_decay_angle = np.arctan2(z_distance, plane_distance)
                    curr_decay_diameter = np.sqrt((curr_side_width * np.cos(curr_decay_angle)) ** 2 + (curr_thickness * np.sin(curr_decay_angle)) ** 2)
                    nearest_distance = np.max([0, nearest_distance - curr_decay_diameter / 2])
                    grid[x, y, z] += np.exp(-nearest_distance / (curr_decay_diameter / 16))

        return grid


def __main__():
    usage = """Usage:use -h for help message"""
    optParser = OptionParser(usage)
    optParser.add_option("--num_sides", action="store", dest="num_sides", type="int", default="3",
                         help="number of sides to generate of the polygon. default:[%default]")
    optParser.add_option("--num_layers", action="store", dest="num_layers", type="int", default="1",
                         help="number of layers of polygons to generate, stacked to each other. default:[%default]")
    optParser.add_option("--side_length", action="store", dest="side_length", type="float", default="20.0",
                         help="The length of the polygon to generate, in angstroms. default:[%default]")
    optParser.add_option("--side_width", action="store", dest="side_width", type="float", default="5.0",
                         help="The width of the side of the polygon to generate, in angstroms. default:[%default]")
    optParser.add_option("--side_thickness", action="store", dest="side_thickness", type="float", default="5.0",
                         help="The thickness of each side in the z direction, in angstroms. default:[%default]")
    optParser.add_option("--layer_spacing", action="store", dest="layer_spacing", type="float", default="10.0",
                         help="The distance between different layers, in angstroms. default:[%default]")
    optParser.add_option("--angpix", action="store", dest="angpix", type="float", default="1.0",
                         help="The pixel size of the generated volume. default:[%default]")
    optParser.add_option("--box_size", action="store", dest="box_size", type="int", default="128",
                         help="The box size of the generated volume, in number of pixels. default:[%default]")
    optParser.add_option("--output", action="store", dest="output", type="string", default="polygon.mrc",
                         help="The output file name. default:[%default]")
    optParser.add_option("--mp_threads", action="store", type=int, dest="num_mp_per_process",
                         default=0, help="Number of threads to use per process. \
                        Always set this unless you are only using one node.default:[%default]")
    options, args = optParser.parse_args()
    #Now convert all scales into units of pixels
    side_length_in_pixels = options.side_length / options.angpix
    side_width_in_pixels = options.side_width / options.angpix
    side_thickness_in_pixels = options.side_thickness / options.angpix
    layer_spacing_in_pixels = options.layer_spacing / options.angpix
    num_sides = []
    side_length = []
    side_width = []
    side_thickness = []
    
    for i in range(options.num_layers):
        num_sides.append(options.num_sides)
        side_length.append(side_length_in_pixels)
        side_width.append(side_width_in_pixels)
        side_thickness.append(side_thickness_in_pixels)
#def create_density_polygon(num_sides, side_length, side_width, thickness, grid_size, num_layers = 1, layer_spacing = 0):   
    generated_volume = create_density_polygon(num_sides=num_sides,\
                                            side_length=side_length,\
                                            side_width=side_width,\
                                            thickness=side_thickness,\
                                            grid_size=(options.box_size, options.box_size, options.box_size),\
                                            num_layers=options.num_layers,\
                                            layer_spacing=layer_spacing_in_pixels)
    with mrcfile.new(options.output, overwrite=True) as mrc:
        mrc.set_data(generated_volume)
        
if __name__ == "__main__":
    __main__()