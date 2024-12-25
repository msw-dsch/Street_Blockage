import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import *
from StreetBlockage import *

simu_name = input('Put the Simulation Name')
trial_number = int(input('Put the Number of Trials'))

'''
DATA INPUT
'''

def input_gdf(dir, name):
    file = os.path.join(dir, f"{name}.geojson")
    return gpd.read_file(file)

input_directory = "sample_data"
input_name = ['bldg', 'block', 'pole', 'road', 'link', 'soil', 'node']

bldg_gdf, block_gdf, pole_gdf, road_gdf, link_gdf, soil_gdf, node_gdf = map(lambda n: input_gdf(input_directory, n), input_name)
start_point_df = pd.read_json(os.path.join(input_directory, "start_points.json"))

'''
SETTING
'''

## SETTING OBJECTS

print('\n--PREPARING--\n')
print('Setting Objects')

soil = Soil(soil_gdf)
bldgs = [Bldg(bldg.id, bldg.geometry, bldg.totalArea, bldg.height, bldg.storeys, bldg.structure, 
              bldg.year, bldg.usage, bldg.district, bldg.access) 
              for bldg in bldg_gdf.itertuples()]
blocks = [Block(block.id, block.geometry, block.height) for block in block_gdf.itertuples()]
poles = [Pole(pole.id, pole.geometry, pole.type, pole.height) for pole in pole_gdf.itertuples()]
roads = [Road(road.id, road.geometry, road.rank, road.section) for road in road_gdf.itertuples()]
links = [RoadLink(link.id, link.section, link.length, link.width, *[part for part in get_parts(link.geometry)]) 
         for link in link_gdf.itertuples()]
nodenames = {}
for row in node_gdf.itertuples():
    nodenames[row.name_from] = {'id': row.name_to, 'ground': row.geometry, 'bldg_access': row.accessibility}
fire_station = {}
for row in start_point_df.itertuples():
    fire_station[row.name] = {'distance': row.distance, 'ids': row.ids}

## SETTING PARAMS

PGV = 69.3
PGA = 729
I = 6

## SETTING PL
print('Calculating PL')

soil.get_PL(PGA)
matrix = soil.variogram_matrix_inv()
for bldg in bldgs:
    bldg.pl = bldg.get_pl(soil, matrix)
for road in roads:
    road.pl = road.get_pl(soil, matrix)

## SETTING DEBRIS RANGE
print('Calculating Debris Range')

road_geos = unary_union([road.ground for road in roads])
debris_precal_list = {}

for bldg in bldgs:
    debris_ground = bldg.blocked_area(road_geos)
    debris_precal_list[bldg.obj_id] = debris_ground

for block in blocks:
    debris_ground = block.blocked_area(road_geos)
    debris_precal_list[block.obj_id] = debris_ground

for pole in poles:
    debris_ground = pole.blocked_area()
    debris_precal_list[pole.obj_id] = debris_ground

print('Preparation Finished\n')

'''
FLOW
'''

start_time = time.time()
os.makedirs(simu_name) 
print('\nSTART SIMULATION\n')

for trial in range(trial_number):
    save_dir = os.path.join(simu_name, f"trial_{trial + 1}")
    os.makedirs(save_dir) 

    ## COLLAPESE

    debris_name_list = [] # debris for blockage
    debriss = [] # whole debris

    for bldg in bldgs:
        bldg.reset()
        z1, z2 = np.random.rand(), np.random.rand()
        a = bldg.shaking(PGV, z1)
        if not a:
            bldg.shaking(PGV, z1, 'Half')
        b = bldg.liquefaction(z2)
        if not b:
            bldg.liquefaction(z2, 'Half')
        if bldg.status != 'exist':
            mass = bldg.debris_mass()
            if bldg.status == 'collapesed':
                debris_name_list.append(bldg.obj_id)
                debris_ground = debris_precal_list[bldg.obj_id]
            else:
                debris_ground = bldg.ground
            debriss.append(Debris(bldg.obj_id, debris_ground, mass))

    for block in blocks:
        block.reset()
        z1 = np.random.rand()
        a = block.shaking(PGA, z1)
        z2 = 0
        for debris in debris_name_list:
            if debris_precal_list[debris].intersects(block.ground):
                z2 = 1
                break
        b = block.bldg(z2)
        if block.status != 'exist':
            mass = block.debris_mass()
            debriss.append(Debris(block.obj_id, debris_precal_list[block.obj_id], mass))
            debris_name_list.append(block.obj_id)

    for pole in poles:
        pole.reset()
        z1, z3 = np.random.rand(), np.random.rand()
        a = pole.shaking(I, z1)
        z2 = 0
        for debris in debris_name_list:
            if debris_precal_list[debris].intersects(pole.ground):
                z2 = 1
                break
        b = pole.bldg(z2, z3)
        if pole.status != 'exist':
            mass = pole.debris_mass()
            debriss.append(Debris(pole.obj_id, debris_precal_list[pole.obj_id], mass))
            debris_name_list.append(pole.obj_id)

    for road in roads:
        road.reset()
        z1, z2 = np.random.rand(), np.random.rand()
        a = road.subsidence(z1, z2)
        if road.status == 'collapesed':
            debris_name_list.append(road.obj_id)

    ## BLOCKAGE

    for link in links:
        link.reset()
        link.blockage(debris_name_list, debris_precal_list)

    ## FIRE

    base_nodes = set()
    base_edges = []

    for n in nodenames.values():
        a = RoadNode(n['id'], n['ground'])
        base_nodes.add(RoadNode(n['id'], n['ground']))

    for link in links:
        if link.width != 0:
            base_edges.append(RoadEdge(link, nodenames, 'emergency'))

    graph = TranGraph(base_nodes, base_edges)

    for bldg in bldgs:
        graph.reset()

        if not any(bldg.access):
            bldg.time_fire = float('inf')
            continue

        end_point = BldgNode(bldg)
        graph.add_node(end_point)

        for n, d in bldg.access.items():
            N = nodenames[n]
            end_edge = BldgAccessEdge(end_point, N['id'], N['ground'])
            graph.add_edge(end_edge)

        for st_name, st_value in fire_station.items():
            st_weight = 3600 * st_value['distance'] / 40000 + 60
            for start_point in st_value['ids']:
                graph.add_imaginal_path('start', start_point, st_name, st_weight)

        graph.remove_isolated_nodes()
        graph.simplify_paths()
        _, __, tm = graph.dijkstra('start', end_point.id)
        
        bldg.time_fire = tm

    bldg_result, block_result, pole_result, road_result, link_result, debris_result = [], [], [], [], [], []
    for bldg in bldgs:
        bldg_result.append({
            'id': bldg.id,
            'status': bldg.status,
            'by_who': f'{bldg.by_who}',
            'time_access': bldg.time_fire
        })
    for block in blocks:
        block_result.append({
            'id': block.id,
            'status': block.status,
            'by_who': f'{block.by_who}'
        })
    for pole in poles:
        pole_result.append({
            'id': pole.id,
            'status': pole.status,
            'by_who': f'{pole.by_who}'
        })
    for road in roads:
        road_result.append({
            'id': road.id,
            'status': road.status,
            'by_who': f'{road.by_who}'
        })
    for link in links:
        link_result.append({
            'id': link.id,
            'status': link.status,
            'by_who': f'{link.by_who}',
            'width': link.width,
            'blockage_rank': link.blockage_rank
        })
    for debris in debriss:
        debris_result.append({
            'id': debris.id,
            'mass': debris.mass
        })

    for obj, obj_result in zip(['bldg', 'block', 'pole', 'road', 'link', 'debris'],
                            [bldg_result, block_result, pole_result, road_result, link_result, debris_result]):
        df = pd.DataFrame(obj_result)
        file_path = os.path.join(save_dir, f'{obj}_result.csv')
        df.to_csv(file_path)

    end_time = time.time()

    print(f'Trial: {trial + 1}, Time: {end_time - start_time}')

print('\nSIMULATION FINISHED\n')