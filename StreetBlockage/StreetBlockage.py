import copy
import heapq

import numpy as np
import pandas as pd
import geopandas as gpd
import optuna
from shapely import *
from scipy.stats import norm

'''土質情報'''
class Soil:
    def __init__(self, soil_gdf: gpd.GeoDataFrame):
        self._soil_gdf = soil_gdf
        self.grounds = soil_gdf['geometry']
        self.soil_data = []
        for soil in soil_gdf.itertuples():
            soil_list = {}
            for i in range(1, 201):
                soil_list[0.1*i] = soil[i]
            self.soil_data.append(soil_list)
        self.shape = len(self.soil_data)

    '''FL値'''
    def get_FL(self, PGA: float):

        FL_matrix = np.zeros([len(self.soil_data), 200])
        for s, soil in enumerate(self.soil_data):
            for z, v in soil.items():
            # N値，上載土圧 sigma (kN/m^2)，有効上載土圧 sigma_dash (kN/m^2)，細粒分含有率 FC (%)，細粒分含有率 D50 から求めたN値低減係数 Cab
                N, sigma, sigma_dash, FC, Cab = v['N'], v['sigma'], v['sigma_dash'], v['FC'], v['Cab']
                # 細粒分含有率 (%)
                if N < 22: # FC値を, 土質から推測する方法とN値から推測する方法で危険側にとる
                    FC = min(916 / (N + 9.21) - 29.5, FC)
                else:
                    FC = 0
                if FC > 35: # FC値が35を超えるものは液状化判定しない
                    FL_matrix[s][int(z*10-1)] = 1
                    continue
                # N値低減
                N = Cab * N
                # 補正N値増分 delta_Nf
                if FC < 5:
                    delta_Nf = 0
                elif FC < 10:
                    delta_Nf = FC * 1.2 - 6
                elif FC < 20:
                    delta_Nf = FC * 0.2 + 4
                else:
                    delta_Nf = FC * 0.1 + 6
                # 換算N値 N1
                N1 = N * np.sqrt(100/sigma_dash)
                # 補正N値 Na
                Na = N1 + delta_Nf
                # 液状化抵抗比 R
                Na_ = 16 * np.sqrt(Na)
                R = 0.45 * 0.57 * (Na_ / 100 + (Na_ / 80) ** 14)
    
                # せん断応力比 L
                M = 7.5 # マグニチュード
                alpha_max = PGA/100 # 地表面における設計用水平加速度 (m/s^2)
                g = 9.8 # 重力加速度 (m/s^2)
                L = 0.1 * (M - 1) * (alpha_max / g) * (sigma / sigma_dash) * (1 - 0.015 * z)

                # 安全率 FL
                FL = R / L 
                
                FL_matrix[s][int(z*10-1)] = min(FL, 1)

        self.fl = FL_matrix
        return FL_matrix

    '''PL値'''    
    def get_PL(self, PGA: float):
        FL_matrix = self.get_FL(PGA)
        PL_array = np.zeros(len(self.soil_data))
        for s, FLs in enumerate(FL_matrix):
            PL = 0
            for Z, FL in enumerate(FLs): # 深さzでFL値に重みをつけて積分
                z = (Z+1) * 0.1
                PL += (1 - FL) * (10 - 0.5 * z) * 0.1
            PL_array[s] = PL

        self.pl = PL_array
        return PL_array

    def variogram_model(self, h, c0, c, a): # 球形モデル
        if h == 0:
            return 0
        elif h > a:
            return c0 + c
        return c0 + c * (1.5 * h / a - 0.5 * (h ** 3 / a ** 3))
    
    '''ヴァリオグラム行列の逆行列'''    
    def variogram_matrix_inv(self):
        # dissimilarity
        all_data = []
        distance_matrix = np.zeros([self.shape, self.shape])
        for i, (PL_i, geo_i) in enumerate(zip(self.pl, self.grounds)):    
            for j, (PL_j, geo_j) in enumerate(zip(self.pl, self.grounds)):
                g = 0.5 * np.square(PL_i - PL_j)
                h = geo_i.distance(geo_j)
                all_data.append({'h': h, 'g': g})
                distance_matrix[i][j] = h
        df_all_data = pd.DataFrame(all_data)

        # optimize
        lag = 50
        df_all_data['h_l'] = list(map(np.floor, df_all_data.h / lag))
        df_experiencial_semivariogram = df_all_data.groupby('h_l').mean()

        def SSE(df_experiencial_semivariogram, c0, c, a):
            sse = 0
            for _, row in df_experiencial_semivariogram.iterrows():
                h = row['h']
                g_e = row['g']
                g_m = self.variogram_model(h, c0, c, a)
                sse += np.square(g_e - g_m)
            return sse

        def objective(trial):
            c0 = trial.suggest_float("c0", 0, 50)
            c = trial.suggest_float("c", 0, 100)
            a = trial.suggest_float("a", 100, 2200)

            sse = SSE(df_experiencial_semivariogram, c0, c, a)
            trial.set_user_attr("constraints", [c0 < c])
            return sse

        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=2500)

        self._c0, self._c, self._a = study.best_params['c0'], study.best_params['c'], study.best_params['a']

        # variogram matrix
        variogram_matrix = np.ones([self.shape+1, self.shape+1])
        for i in range(self.shape):
            for j in range(self.shape):
                variogram_matrix[i][j] = self.variogram_model(distance_matrix[i][j], self._c0, self._c, self._a)
        variogram_matrix[self.shape][self.shape] = 0

        # variogram invers matrix
        variogram_matrix_inv = np.linalg.inv(variogram_matrix)

        return variogram_matrix_inv

'''構造物'''
class CityObject:
    def __init__(self, type: str, cityobject_id, ground: Geometry, status='exist'):
        self.id, self.type, self.ground, self.status = cityobject_id, type, ground, status
        self.obj_id = type + str(cityobject_id)
        self.pl = -1

    '''初期化'''
    def reset(self):
        self.status = 'exist'
    
    '''地点のPL値を推定'''
    def get_pl(self, soil: Soil, variogram_matrix_inv: np.ndarray):
        bs = []
        for geo in soil.grounds:
            h = self.ground.centroid.distance(geo)
            bs.append(soil.variogram_model(h, soil._c0, soil._c, soil._a))
        bs.append(1)
        ws = np.dot(variogram_matrix_inv, bs)
        PL = sum(ws[:soil.shape] * soil.pl[:soil.shape])
        self.pl = PL
        return PL

'''建物'''
class Bldg(CityObject):
    def __init__(self, bldg_id, ground: Geometry, total_area=200,
                 height=8, storeys=2, structure='w', year=0, usage=0, 
                 district=-1, access={}):
        super().__init__('bldg', bldg_id, ground)
        self.ground_area, self.total_area = self.ground.area, total_area
        self.height, self.storeys = height, storeys
        self.structure, self.year, self.usage = structure, str(year), usage
        self.district, self.access = district, access
        self.by_who, self.time_fire = [], float('inf')

    '''初期化'''
    def reset(self):
        self.status, self.by_who, self.time_fire = 'exist', [], float('inf')

    '''揺れによる倒壊モデル'''
    def shaking(self, PGV, Z, level='Completely'):

        if level == 'Completely':
            lamb_list = {
                'w0': 4.96, 'w1': 5.16, 'w2': 5.41, 'w3': 5.83, 'w4': 6.09,
                'rc0': 5.12, 'rc1': 5.33, 'rc2': 6.00, 'rc3': 6.00, 'rc4': 6.00,
                's0': 4.64, 's1': 4.97, 's2': 5.64, 's3': 5.64, 's4': 5.64
                    }
            xi_list = {
                'w0': 0.44, 'w1': 0.50, 'w2': 0.55, 'w3': 0.72, 'w4': 0.72,
                'rc0': 0.646, 'rc1': 0.575, 'rc2': 0.789, 'rc3': 0.789, 'rc4': 0.789,
                's0': 0.619, 's1': 0.490, 's2': 0.731, 's3': 0.731, 's4': 0.731
                }
        elif level == 'Half':
            lamb_list = {
                'w0': 4.47, 'w1': 4.56, 'w2': 4.71, 'w3': 5.10, 'w4': 5.70,
                'rc0': 4.72, 'rc1': 4.85, 'rc2': 5.33, 'rc3': 5.33, 'rc4': 5.33,
                's0': 4.25, 's1': 4.49, 's2': 5.01, 's3': 5.01, 's4': 5.01
                    }
            xi_list = {
                'w0': 0.54, 'w1': 0.61, 'w2': 0.67, 'w3': 0.79, 'w4': 0.83,
                'rc0': 0.691, 'rc1': 0.612, 'rc2': 0.789, 'rc3': 0.789, 'rc4': 0.789,
                's0': 0.712, 's1': 0.549, 's2': 0.733, 's3': 0.733, 's4': 0.733
                }

        
        rank = self.structure + self.year
        lamb, xi = lamb_list[rank], xi_list[rank]
        P = norm.cdf((np.log(PGV)-lamb)/xi)

        result = Z < P
        if result:
            if self.status == 'exist':
                if level in ['Completely', 'Half']:
                    self.status = 'halfly collapesed'
            if self.status == 'halfly collapesed':
                if level == 'Completely':
                    self.status = 'collapesed'
            self.by_who.append('by shaking')

        return result
    
    '''液状化による倒壊モデル'''
    def liquefaction(self, Z, level='Completely'):

        a = 0.096
        b_list = {15: 0.18, 5: 0.05, 0: 0.02}
        b = 0
        for k, v in b_list.items():
            if self.pl > k:
                b = v
                break

        P = a * b
        if level == 'Half':
            P = 2.2 * P

        result = Z < P
        if result:
            if self.status == 'exist':
                if level in ['Completely', 'Half']:
                    self.status = 'halfly collapesed'
            if self.status == 'halfly collapesed':
                if level == 'Completely':
                    self.status = 'collapesed'
            self.by_who.append('by liquefaction')

        return result
    
    '''建物平面の弱軸'''
    def weak_axis_vector(self):

        if get_num_geometries(self.ground) > 1:
            
            centroids = []
            centroid_lines = []
            
            for geom in self.ground.geoms:
                geom_centroid = geom.centroid
                for point in centroids:
                    centroid_lines.append(LineString([point, geom_centroid]).buffer(0.01, 3))
                centroids.append(geom_centroid)

            centroid_lines.append(self.ground)
            bldg_polygon = unary_union(centroid_lines)

        else:
            bldg_polygon = self.ground

        I_max = 0
        a_I_max = 0
        
        for angle in range(0, 180):
        
            # 図心
            bldg_center = bldg_polygon.centroid
        
            # 図心からの相対位置に変換
            bldg_parallel_x, bldg_parallel_y = map(lambda v: list(v[0]-v[1]), zip(map(list, bldg_polygon.exterior.coords.xy), [bldg_center.x, bldg_center.y]))
            bldg_parallel = list(zip(bldg_parallel_x, bldg_parallel_y))
        
            # 図心まわりに angle 度だけ回転
            theta = np.radians(angle)
            cos, sin = np.cos(theta), np.sin(theta)
            bldg_rotate = list(map(lambda v: (v[0] * cos - v[1] * sin, v[0] * sin + v[1] * cos), bldg_parallel))
            bldg_rotate_polygon = Polygon(bldg_rotate)
        
            # 断面二次モーメントの計算
            I_t = 0
            for i in range(len(bldg_rotate)-1):
                xi, yi = bldg_rotate[i]
                xj, yj = bldg_rotate[i+1]
                
                if xi == xj:
                    continue
                
                a = (yj - yi)/(xj - xi)
                b = yi - a * xi
        
                if yj * yi >= 0:
                    
                    if a == 0:
                        I_ti = abs(b**3 * (xj - xi) / 3)
                    else:
                        I_ti = abs((yj ** 4 - yi ** 4)/(12 * a))
                    
                    mid_coord = [(xi + xj)/2, (yi + yj)/2]
                    test_coord = [mid_coord[0] + 0.01 * (yi - yj), mid_coord[1] + 0.01 * (xj - xi)]
                    polygon_side = [Point(test_coord).within(bldg_rotate_polygon), abs(mid_coord[1]) > abs(test_coord[1])]
                
                    if sum(polygon_side) == 1:
                        I_ti = -I_ti
                        
                else:
                    
                    xm = -b/a
                    I_ti = 0.0
                    
                    for x, y in zip([xi, xj], [yi, yj]):
                        
                        I_tm = abs((y ** 4)/(12 * a))
                        
                        mid_coord = [(x + xm)/2, y/2]
                        test_coord = [mid_coord[0] + 0.01 * y, mid_coord[1] - 0.01 * x]
                        polygon_side = [Point(test_coord).within(bldg_rotate_polygon), abs(mid_coord[1]) > abs(test_coord[1])]
                        
                        if sum(polygon_side) == 1:
                            I_tm = -I_tm
        
                        I_ti += I_tm
                
                I_t += I_ti

            if I_max < I_t:
                I_max = I_t
                a_I_max = angle

        return a_I_max
    
    '''瓦礫流出範囲'''
    def blocked_area(self, road_gemetry: Geometry):
            
        a_I_max = self.weak_axis_vector()
        distance = 3  # 瓦礫流出幅を固定する方法

        bldg_buffer = make_valid(self.ground.buffer(distance * 7/12))

        max_blocked_area = 0
        max_blocked_geo = None
        
        # 倒壊方向の決定
        for angle in [a_I_max-90, a_I_max, a_I_max+90, a_I_max+180]:
            
            block_direction = np.radians(angle)
            cos, sin = np.cos(block_direction), np.sin(block_direction)
            
            blocked_geo = None
            
            for bldg_buffer_part in get_parts(bldg_buffer):
                
                new_coords = []
                
                for x, y in bldg_buffer_part.exterior.coords:
                    x_new, y_new = x + distance * 5/12 * cos, y + distance * 5/12 * sin
                    new_coords.append((x_new, y_new))
        
                blocked_geo = unary_union([blocked_geo, Polygon(new_coords)])
                
            blocked_area = blocked_geo.intersection(road_gemetry).area
            
            if max_blocked_area <= blocked_area:
                
                max_blocked_area = blocked_area
                max_blocked_geo = blocked_geo
    
        return max_blocked_geo
    
    '''瓦礫発生量'''
    def debris_mass(self):
        M = { # kg
            'wood': 0, 'concrete': 0, 'metal': 0, 'other_burnable': 0, 'other_unburnable': 0
            }
        unit_list = { # kg/m^2
            'w_wood': 76.3, 'w_concrete': 84.4, 'w_metal': 7.79, 'w_other_burnable': 17.8, 'w_other_unburnable': 126,
            'rc_wood': 19.0, 'rc_concrete': 1026, 'rc_metal': 39.0, 'rc_other_burnable': 0.343, 'rc_other_unburnable': 2.43,
            's_wood': 204, 's_concrete': 566, 's_metal': 27.0, 's_other_burnable': 0.403, 's_other_unburnable': 2.87
        }
        if self.status != 'exist':
            a = 1
            if self.status == 'halfly_collapesed':
                a = 0.5
            for key in M:
                debris_type = self.structure + '_' + key
                M[key] += self.total_area * a * unit_list[debris_type]
                
        return M

'''ブロック塀'''
class Block(CityObject):
    def __init__(self, block_id, ground: LineString, height=1.2):
        super().__init__('block', block_id, ground)
        self.height = height
        self.length = self.ground.length
        self.by_who = []

    '''初期化'''
    def reset(self):
        self.status, self.by_who = 'exist', []

    '''揺れによる倒壊モデル'''
    def shaking(self, PGA, Z):
        P = 0.0007 * PGA - 0.126

        result = Z < P
        if result:
            if self.status == 'exist':
                self.status = 'collapesed'
            self.by_who.append('by shaking')

        return result
    
    '''建物倒壊に起因する倒壊モデル'''
    def bldg(self, debris_exist, Z=0.5):
        if debris_exist:
            P = 1
        else:
            P = 0

        result = Z < P
        if result:
            if self.status == 'exist':
                self.status = 'collapesed'
            self.by_who.append('by debris')

        return result
    
    '''ブロック塀倒壊方向の設定'''
    def fall_direction(self, road_geometry: Geometry):

        block_areas = tuple(map(lambda i: self.ground.buffer(2*i, single_sided=True).intersection(road_geometry).area, (1, -1)))
        d = 'l' if block_areas[0] > block_areas[1] else 'r'
        self.direction = d
        return d
    
    '''瓦礫流出範囲'''
    def blocked_area(self, road_geometry: Geometry):
            
        dist = self.height/1000
        drct = 1 if self.fall_direction(road_geometry) == 'l' else -1
        blocked_geo = self.ground.buffer(drct * dist, single_sided=True)
        
        return blocked_geo 
    
    '''瓦礫発生量'''
    def debris_mass(self):
        M = {'concrete': 0}
        unit_list = {'block_concrete': 13} #kg
        if self.status != 'exist':
            a = 1 / (190 * 0.390)
            if self.status == 'halfly_collapesed':
                a *= 0.5
            for key in M:
                debris_type = self.type + '_' + key
                M[key] += self.length * self.height * a * unit_list[debris_type]
                
        return M
        
'''電柱'''
class Pole(CityObject):
    def __init__(self, pole_id, ground: Point, usage='main', height=8.0):
        super().__init__('pole', pole_id, ground)
        self.usage, self.height = usage, height
        self.by_who = []

    '''初期化'''
    def reset(self):
        self.status, self.by_who = 'exist', []

    '''揺れによる倒壊モデル'''
    def shaking(self, I, Z):
        a = {5: 0.0000005, 6: 0.00056, 7: 0.008}
        P = a[I]

        result = Z < P
        if result:
            if self.status == 'exist':
                self.status = 'collapesed'
            self.by_who.append('by shaking')

        return result
    
    '''建物倒壊に起因する倒壊モデル'''
    def bldg(self, debris_exist, Z):
        if debris_exist:
            P = 0.17155
        else:
            P = 0

        result = Z < P
        if result:
            if self.status == 'exist':
                self.status = 'collapesed'
            self.by_who.append('by debris')

        return result
    
    '''影響範囲モデル'''
    def blocked_area(self):
            
        height = self.height
        blocked_geo = self.ground.buffer(height)
        
        return blocked_geo 
    
    '''瓦礫発生量'''
    def debris_mass(self):
        M = {'concrete': 0}
        unit_list = { #kg
            'main_concrete': 1000,
            'small_concrete': 500,
            'sub_concrete': 500
            }
        if self.status != 'exist':
            a = 1
            for key in M:
                debris_type = self.usage + '_' + key
                M[key] += a * unit_list[debris_type]
                
        return M
        
'''道路'''
class Road(CityObject):
    def __init__(self, road_id, ground: Geometry, rank: int, section: bool):
        super().__init__('road', road_id, ground)
        self.rank, self.section = rank, section
        self.by_who = []

    '''初期化'''
    def reset(self):
        self.status, self.by_who = 'exist', []
    
    '''最狭道路幅'''
    def width(self, roadlinks: list):
        w = float('inf')
        for roadlink in roadlinks:
            if roadlink.road_id == self.id:
                w = min([w, roadlink.width])
        return w
    
    '''道路長'''
    def length(self, roadlinks: list):
        l = 0
        for roadlink in roadlinks:
            if roadlink.road_id == self.id:
                l += roadlink.length
        return l

    '''道路沈下モデル'''
    def subsidence(self, Z1, Z2):
        road_f_list = {0: 0.9307, 1: 0.3122}
        road_sigma_list = {0: 1.026, 1: 1.206}
        f, sigma = road_f_list[self.rank], road_sigma_list[self.rank]

        r = np.sqrt(-2 * np.log(Z1)) * np.cos(2 * np.pi * Z2) * sigma

        subs = f * self.pl * np.exp(r)

        result = subs > 30
        if result:
            if self.status == 'exist':
                self.status = 'collapesed'
            self.by_who.append('by subsidence')

        return result
    
    '''閉塞状態'''
    def get_initial_roadlinks(self, links: list):
        link_list = []
        for link in links:
            if link.road_id == self.id:
                link_list.append(link)
        return link_list
    
    def blockage_rank(self, roadlinks: list):
        b = 0
        for roadlink in roadlinks:
            if roadlink.road_id == self.id:
                b = max([b, roadlink.blockage_rank])
        return b
    
    def is_blocked(self, roadlinks: list, rank: int):
        if rank <= self.blockage_rank(roadlinks):
            return True
        return False
    
    def get_afterblocked_roadlinks(self, links: list, rank: int, backward=False):
        if backward:
            links.reverse()
        link_list = []
        for link in links:
            if link.road_id == self.id:
                if rank > link.blockage_rank:
                    link_list.append(link)
                else:
                    return link_list
        return link_list
    
'''瓦礫'''
class Debris(CityObject):
    def __init__(self, obj_id: str, ground: Polygon, mass_list: dict):
        super().__init__('debris', obj_id, ground)
        self.mass = mass_list
        self.total_mass = sum([v for v in self.mass.values()])

    '''瓦礫体積'''
    def volumn(self):
        volumns = {}
        gravity_list = { # kg/m^3
            'wood': 400,
            'concrete': 1480,
            'metal': 1130,
            'other_burnable': 400,
            'other_unburnable': 1000
        }
        for k, m in self.mass.items():
            volumns[k] = m / gravity_list[k]

        return volumns

'''道路リンク'''
class RoadLink:
    def __init__(self, link_id, section: bool, length: float, width: float, 
                 ground: Polygon, node1: Point, node2: Point, edge1: LineString, edge2: LineString):
        self.status_list = {0: 3, 0.75: 3, 2: 2, 3: 1}
        self.id, self.section = link_id, section
        self.road_id = int(self.id.split('|')[1])
        self.length, self.width, self.initial_width = length, width, width
        self.ground = ground
        self.nodes, self.edges = [node1, node2], [edge1, edge2]
        b = 0
        for k, v in self.status_list.items():
            if self.initial_width < k:
                b = v
                break
        self.status, self.by_who, self.blockage_rank = 'initial', [], b

    '''初期化'''
    def reset(self):
        self.status, self.by_who = 'exist', []
        self.set_blockage_rank(self.initial_width)
        self.width = self.initial_width

    '''閉塞レベル判定'''
    def set_blockage_rank(self, width):
        b = 0
        for k, v in self.status_list.items():
            if width < k:
                b = v
                break
        self.blockage_rank = b

    '''幅員の計算'''
    def get_width(self, link_polygon, edges):

        def furthest_point(geom, polygon): 
            max_distance = 0    
            if get_num_geometries(polygon) > 1:
                for poly in polygon.geoms:
                    for p in poly.exterior.coords:
                        p_distance = Point(p).distance(geom)
                        if p_distance > max_distance:
                            max_distance = p_distance
            else:                
                for p in polygon.exterior.coords:
                    p_distance = Point(p).distance(geom)
                    if p_distance > max_distance:
                        max_distance = p_distance
            return max_distance

        interval = 0.1

        link = LineString(self.nodes)
        l = link.length
        s, g = map(lambda n: [n.x, n.y], self.nodes)
        v = [(g[0] - s[0]) / l, (g[1] - s[1]) / l]
        per = [-v[1], v[0]]

        number = max(int(np.ceil(l/interval))-1, 1)
        max_distance = furthest_point(link, link_polygon)
        
        w_lst = []

        #by interval measure the width
        for n in range(number):
            if number != 1:
                p = (s[0]+v[0]*(n+1)*interval, s[1]+v[1]*(n+1)*interval)
            else:
                p = ((s[0]+g[0])/2, (s[1]+g[1])/2)
            plus_cutter = LineString([(p[0]+per[0]*max_distance, p[1]+per[1]*max_distance), p])
            minus_cutter = LineString([p, (p[0]-per[0]*max_distance, p[1]-per[1]*max_distance)])
            cutter = unary_union([plus_cutter, minus_cutter])
            
            cutter_intersection = cutter.intersection(link_polygon)
            
            if get_num_geometries(cutter_intersection) == 1:
                w_lst.append(cutter_intersection.length)
                
            # if the link is split, measure the widthest one
            else:
                width_part_lst = []
                for line_part in cutter_intersection.geoms:
                    width_part_lst.append(line_part.length)
                    w_lst.append(max(width_part_lst))

        # if cannot measure the right width, return the minimum length of the edges
        if not w_lst:
            for edge in edges:
                if edge:
                    w_lst.append(2 * minimum_bounding_radius(edge))
        if not w_lst:
            return 0
            
        return min(w_lst)
        
    '''道路閉塞モデル'''
    def blockage(self, debris_list, debris_precal_list):
        
        obj_id = 'road' + str(self.road_id)
        if obj_id in debris_list:
            self.status = 'blocked'
            self.by_who.append(f'by {obj_id}')
            self.width = 0
            self.blockage_rank = self.status_list[0]
            return
        
        total_debris = []
        for debris_id in debris_list:
            if 'road' in debris_id:
                continue
            if debris_precal_list[debris_id].intersects(self.ground):
                self.status = 'blocked'
                self.by_who.append(f'by {debris_id}')
                total_debris.append(debris_precal_list[debris_id])
        if not total_debris:
            return

        blocked_area = unary_union(total_debris)
        unblocked_edges = list(map(lambda edge: edge.difference(blocked_area), self.edges))
            
        if unblocked_edges[0].is_empty or unblocked_edges[1].is_empty: # リンク端部が完全に閉塞している場合
            self.width = 0
            self.blockage_rank = self.status_list[0]
        elif self.section:
            self.width = min([unblocked_edges[0].length, unblocked_edges[1].length])
            self.set_blockage_rank(self.width)
        else:
            unblocked_polygon = self.ground.difference(blocked_area)
            self.width = self.get_width(unblocked_polygon, unblocked_edges)
            self.set_blockage_rank(self.width)

'''経路：ノード'''
class TranNode:
    def __init__(self, node_id):
        self.id = node_id

class RoadNode(TranNode):
    def __init__(self, node_id, ground: Point):
        super().__init__(node_id)
        self.ground = ground

class BldgNode(TranNode):
    def __init__(self, bldg: Bldg):
        super().__init__(bldg.obj_id)
        self.ground = bldg.ground.centroid
        self.access = bldg.access

'''経路：エッジ'''
class TranEdge:
    def __init__(self, edge_id, node1, node2, weight):
        self.id = edge_id
        self.node1, self.node2 = node1, node2
        self.weight = weight

class RoadEdge(TranEdge):
    def __init__(self, link: RoadLink, nodenames: dict, value='distance'):
        name_pre, name_post = link.id.split('*')
        node1_name, node2_name = map(lambda n: name_pre + '*' + name_post.split('|')[n], (0, 1))
        node1_id, node2_id = map(lambda i: nodenames[i]['id'], (node1_name, node2_name))
        self.ground = link.ground
        if value == 'distance':
            weight = link.length
        elif value == 'emergency':
            if link.width < 1:
                weight = float('inf')
            elif link.width < 3:
                weight = 3600 * link.length / 3000 #[s]
            elif link.width < 12:
                weight = max([3600 * link.length / 25000,  (3600 * link.length) / (2761 * link.width + 6566)]) #[s]
            else:
                weight = 3600 * link.length / 40000 #[s]
        super().__init__(link.id, node1_id, node2_id, weight)

class BldgAccessEdge(TranEdge):
    def __init__(self, bldgnode: BldgNode, node_id, node_geo):
        node1_id, node2_id = bldgnode.id, node_id
        weight = bldgnode.ground.distance(node_geo)
        super().__init__(f'{node1_id}-{node2_id}', node1_id, node2_id, weight)

'''経路：グラフ'''
class TranGraph:
    def __init__(self, nodes: set, edges: list):
        self.nodes, self._nodes = set([node.id for node in nodes]), set([node.id for node in nodes])
        self.edges, self._edges = {node: [] for node in self.nodes}, {node: [] for node in self.nodes}
        for edge in edges:
            self.edges[edge.node1].append((edge.node2, edge.id, edge.weight))
            self.edges[edge.node2].append((edge.node1, edge.id, edge.weight))
            self._edges[edge.node1].append((edge.node2, edge.id, edge.weight))
            self._edges[edge.node2].append((edge.node1, edge.id, edge.weight))

    '''初期化'''
    def reset(self):
        self.nodes = copy.deepcopy(self._nodes)
        self.edges = copy.deepcopy(self._edges)
    
    '''ノードの追加'''
    def add_node(self, node: TranNode):
        self.nodes.add(node.id)
        if node.id not in self.edges:
            self.edges[node.id] = []

    '''エッジの追加'''
    def add_edge(self, edge: TranEdge):
        if edge.node1 in self.edges:
            self.edges[edge.node1].append((edge.node2, edge.id, edge.weight))
        else:
            self.edges[edge.node1] = [(edge.node2, edge.id, edge.weight)]
        if edge.node2 in self.edges:
            self.edges[edge.node2].append((edge.node1, edge.id, edge.weight))
        else:
            self.edges[edge.node2] = [(edge.node1, edge.id, edge.weight)]

    '''エッジの削除'''
    def delete_edge(self, edge: TranEdge):
        if edge.node1 in self.edges and (edge.node2, edge.id, edge.weight) in self.edges[edge.node1]:
            self.edges[edge.node1].remove((edge.node2, edge.id, edge.weight))
        if edge.node2 in self.edges and (edge.node1, edge.id, edge.weight) in self.edges[edge.node2]:
            self.edges[edge.node2].remove((edge.node1, edge.id, edge.weight))

    '''孤立ノードの削除'''
    def remove_isolated_nodes(self):
        for n in list(self.nodes):
            if len(self.edges[n]) == 0:
                self.nodes.remove(n)
                del self.edges[n]

    '''無意味なエッジ分割の解消'''
    def simplify_paths(self):
        simplified_edges = {k: list(v) for k, v in self.edges.items()}

        for nd in list(self.nodes):
            if len(simplified_edges[nd]) != 2:
                continue

            (neighbor1, edge1, weight1), (neighbor2, edge2, weight2) = simplified_edges[nd]

            if neighbor1 != neighbor2:
                total_weight = weight1 + weight2
                new_edge_name = f"{edge1}-{edge2}"

                simplified_edges[neighbor1] = [
                    (n, e, w) for n, e, w in simplified_edges[neighbor1] if n != nd
                ]
                simplified_edges[neighbor1].append((neighbor2, new_edge_name, total_weight))

                simplified_edges[neighbor2] = [
                    (n, e, w) for n, e, w in simplified_edges[neighbor2] if n != nd
                ]
                simplified_edges[neighbor2].append((neighbor1, new_edge_name, total_weight))

                del simplified_edges[nd]
                self.nodes.remove(nd)

        self.edges = simplified_edges

    '''仮想経路の設定（複数スタート地点の設定）'''
    def add_imaginal_path(self, node_id, connection_id, edge_id, weight=0):
        self.add_node(TranNode(node_id))
        self.add_edge(TranEdge(edge_id, node_id, connection_id, weight))

    '''ダイクストラ法による経路探索'''
    def dijkstra(self, start_node, end_node):
        priority_queue = []
        shortest_distances = {nd: float('inf') for nd in self.nodes}
        shortest_distances[start_node] = 0
        previous_nodes = {node: None for node in self.nodes}
        previous_edges = {node: None for node in self.nodes}

        heapq.heappush(priority_queue, (0, start_node))

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node == end_node:
                path = []
                path_edge = []
                while current_node:
                    path.append(current_node)
                    if previous_edges[current_node]:
                        path_edge.append(previous_edges[current_node])
                    current_node = previous_nodes[current_node]
                return path[::-1], path_edge[::-1], shortest_distances[end_node]

            for neighbor, edge_id, weight in self.edges[current_node]:
                if neighbor not in self.nodes:
                    continue
                distance = current_distance + weight
                if distance < shortest_distances[neighbor]:
                    shortest_distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    previous_edges[neighbor] = edge_id
                    heapq.heappush(priority_queue, (distance, neighbor))

        return None, None, float('inf')