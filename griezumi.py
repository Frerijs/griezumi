# app.py

import os
import glob
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString, Polygon
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import Delaunay
import rasterio
from rasterio.vrt import WarpedVRT
import rasterio.plot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
from io import BytesIO
import streamlit as st
import tempfile
import warnings

# Ignorēt rasterio brīdinājumus par 'FutureWarning'
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================
# 1) Definējam palīgf-jas
# ==========================

# Definējam EPSG:3059 kā mērķa CRS
TARGET_EPSG = 3059

# ==========================
# 2) Palīgfunkcijas
# ==========================

def generate_map_image(line_geom, points_gdf, ortho_dataset):
    """
    Izveido karti ar ortofoto, griezuma līniju un punktiem.
    Atgriež attēlu kā base64 enkodētu string.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Rāda ortofoto
    if ortho_dataset:
        rasterio.plot.show(ortho_dataset, ax=ax, alpha=0.5)
    else:
        st.warning("Ortofoto dati nav pieejami, karte tiks veidota bez ortofota.")

    # Rāda griezuma līniju
    x_line, y_line = line_geom.xy
    ax.plot(x_line, y_line, color='red', linewidth=2, label='Griezuma līnija')

    # Rāda punktus
    points_gdf.plot(ax=ax, marker='o', color='yellow', markersize=5, label='Punkti')

    ax.axis('off')
    ax.legend()

    # Iestatīt robežas ar nelielu marginu
    all_geoms = [line_geom] + list(points_gdf.geometry)
    total_bounds = gpd.GeoSeries(all_geoms).total_bounds
    x_min, y_min, x_max, y_max = total_bounds
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Saglabāt attēlu buferī
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

def visualize_triangulation(tri, ortho_dataset):
    """
    Vizualizē Delaunay triangulāciju uz ortofoto kartes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    if ortho_dataset:
        rasterio.plot.show(ortho_dataset, ax=ax, alpha=0.5)
    else:
        st.warning("Ortofoto dati nav pieejami, triangulācija tiks veidota bez ortofota.")

    plt.triplot(tri.points[:,0], tri.points[:,1], tri.simplices.copy(), color='blue', alpha=0.3)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Delaunay Triangulation')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

def load_landxml(surface_directory):
    """
    Ielādē jau triangulētus LandXML failus, ekstraktē punktus un triangulāciju.
    Atgriež sarakstu ar interpolatoriem un to nosaukumiem.
    """
    surface_interpolators = {}
    landxml_files = glob.glob(os.path.join(surface_directory, '*.xml'))
    
    for landxml_file in landxml_files:
        surface_name = os.path.splitext(os.path.basename(landxml_file))[0] + '_xml'
        try:
            tree = ET.parse(landxml_file)
            root = tree.getroot()

            # Extrahē PntList3D punktus
            all_coords = []
            pnt_list3d = root.findall('.//{*}PntList3D')
            for pnt in pnt_list3d:
                text = pnt.text.strip()
                points = text.split()
                for i in range(0, len(points), 3):
                    y = float(points[i])  # Apmainītas vietas, lai pārbaudītu
                    x = float(points[i+1])
                    z = float(points[i+2])
                    all_coords.append([x, y, z])

            # Extrahē P elementus (ja nepieciešams)
            p_elements = root.findall('.//{*}P')
            for p in p_elements:
                text = p.text.strip()
                if text:
                    points = text.split()
                    if len(points) >= 3:
                        y = float(points[0])
                        x = float(points[1])
                        z = float(points[2])
                        all_coords.append([x, y, z])

            if not all_coords:
                st.warning(f"LandXML fails '{landxml_file}' nesatur derīgus punktus.")
                continue

            coords = np.array(all_coords)
            coords_xy = coords[:, :2]
            z_coords = coords[:, 2]

            if coords.shape[0] < 3:
                st.warning(f"LandXML fails '{landxml_file}' satur mazāk nekā 3 punktus.")
                continue

            # Izveido Delaunay triangulāciju (ja jau nav triangulēts)
            # Pārbaudiet, vai triangulācija jau ir iekļauta LandXML
            # Ja ir 'Faces' elementi, izmanto tos
            faces = []
            faces_elem = root.findall('.//{*}Faces/{*}F')
            if faces_elem:
                for face in faces_elem:
                    indices = list(map(int, face.text.strip().split()))
                    faces.append([idx - 1 for idx in indices])  # Atgriež uz nulles indeksiem
                tri = Delaunay(coords_xy)
                tri.simplices = np.array(faces)
                interpolator = LinearNDInterpolator(tri, z_coords)
            else:
                # Ja Faces nav, izveido triangulāciju no punktiem
                tri = Delaunay(coords_xy)
                interpolator = LinearNDInterpolator(tri, z_coords)

            # Saglabā interpolatoru
            surface_interpolators[surface_name] = {
                'interpolator': interpolator,
                'tri': tri,
                'points_xy': coords_xy,
                'z': z_coords
            }

            st.write(f"✅ Ielādēts LandXML virsmas: {surface_name}")
            st.write(f"   Punktu skaits: {len(coords)}")
            st.write(f"   Triangulācija ar {len(tri.simplices)} trijstūriem.")

            # Vizualizē triangulāciju
            tri_image = visualize_triangulation(tri, ortho_dataset=None)  # Ortofoto nav pieejams šajā punktā
            st.image(f'data:image/png;base64,{tri_image}', use_container_width=True, caption=f'Delaunay Triangulation for {surface_name}')

        except Exception as e:
            st.error(f"Kļūda LandXML faila '{landxml_file}' apstrādē: {e}")

    return surface_interpolators

def load_dem(surface_directory):
    """
    Ielādē DEM (Digital Elevation Model) failus.
    Atgriež sarakstu ar DEM datasetiem un to nosaukumiem.
    """
    surface_rasters = {}
    dem_files = glob.glob(os.path.join(surface_directory, '*.tif'))
    
    for dem_file in dem_files:
        surface_name = os.path.splitext(os.path.basename(dem_file))[0] + '_dem'
        try:
            dem_dataset = rasterio.open(dem_file)

            # Ja CRS nav EPSG:3059, pārveidojam to
            if dem_dataset.crs.to_epsg() != TARGET_EPSG:
                vrt_params = {'crs': f'EPSG:{TARGET_EPSG}'}
                dem_dataset = WarpedVRT(dem_dataset, **vrt_params)

            surface_rasters[surface_name] = dem_dataset
            st.write(f"✅ Ielādēts DEM virsmas: {surface_name}")
        except Exception as e:
            st.error(f"Kļūda DEM faila '{dem_file}' apstrādē: {e}")
    
    return surface_rasters

def load_ortho(ortho_directory):
    """
    Ielādē Ortofoto (GeoTIFF) failus.
    Atgriež vienu Ortofoto datasetu (pirmo atrasto).
    """
    ortho_files = glob.glob(os.path.join(ortho_directory, '*.tif'))
    if not ortho_files:
        st.warning("Ortofoto faili netika atrasti.")
        return None
    
    ortho_file = ortho_files[0]
    try:
        ortho_dataset = rasterio.open(ortho_file)

        if ortho_dataset.crs.to_epsg() != TARGET_EPSG:
            vrt_params = {'crs': f'EPSG:{TARGET_EPSG}'}
            ortho_dataset = WarpedVRT(ortho_dataset, **vrt_params)

        st.write(f"✅ Ielādēts Ortofoto: {os.path.basename(ortho_file)}")
        return ortho_dataset
    except Exception as e:
        st.error(f"Kļūda Ortofoto faila '{ortho_file}' apstrādē: {e}")
        return None

def points_within_bounds(x, y, tri):
    """
    Pārbauda, vai punkti atrodas triangulācijas ietvarā.
    """
    x_min = tri.points[:,0].min()
    y_min = tri.points[:,1].min()
    x_max = tri.points[:,0].max()
    y_max = tri.points[:,1].max()
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

# ==========================
# 3) Streamlit lietotnes daļa
# ==========================

st.title("Šķērsgriezumu Profili ar Triangulētu LandXML Interpolāciju")

st.sidebar.header("Datu Ielāde")

with tempfile.TemporaryDirectory() as tmpdirname:
    shp_dir = os.path.join(tmpdirname, 'linijas')
    surface_dir = os.path.join(tmpdirname, 'virsmas')
    ortho_dir = os.path.join(tmpdirname, 'ortofoto')
    
    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(ortho_dir, exist_ok=True)
    
    st.sidebar.subheader("SHP Līnijas")
    uploaded_shp_zip = st.sidebar.file_uploader("Augšupielādēt SHP ZIP (satur .shp, .shx, .dbf, utt.)", type=["zip"], key="shp_zip")
    
    if uploaded_shp_zip is not None:
        with zipfile.ZipFile(uploaded_shp_zip) as z:
            z.extractall(shp_dir)
        st.sidebar.success("SHP faili augšupielādēti un izvilkti veiksmīgi!")
    else:
        st.sidebar.info("Lūdzu, augšupielādējiet SHP ZIP arhīvu.")
    
    st.sidebar.subheader("LandXML Virsmas")
    uploaded_landxml = st.sidebar.file_uploader("Augšupielādēt Triangulētus LandXML faili (.xml)", type=["xml"], accept_multiple_files=True, key="landxml")
    
    if uploaded_landxml:
        for file in uploaded_landxml:
            with open(os.path.join(surface_dir, file.name), 'wb') as f:
                f.write(file.getbuffer())
        st.sidebar.success("Triangulētie LandXML faili augšupielādēti veiksmīgi!")
    
    st.sidebar.subheader("DEM Virsmas")
    uploaded_dem = st.sidebar.file_uploader("Augšupielādēt DEM faili (.tif)", type=["tif"], accept_multiple_files=True, key="dem")
    
    if uploaded_dem:
        for file in uploaded_dem:
            with open(os.path.join(surface_dir, file.name), 'wb') as f:
                f.write(file.getbuffer())
        st.sidebar.success("DEM faili augšupielādēti veiksmīgi!")
    
    st.sidebar.subheader("Ortofoto")
    uploaded_ortho_zip = st.sidebar.file_uploader("Augšupielādēt Ortofoto ZIP (satur .tif)", type=["zip"], key="ortho_zip")
    uploaded_ortho = st.sidebar.file_uploader("Augšupielādēt Ortofoto faili (.tif)", type=["tif"], accept_multiple_files=True, key="ortho_files")
    
    if uploaded_ortho_zip is not None:
        with zipfile.ZipFile(uploaded_ortho_zip) as z:
            z.extractall(ortho_dir)
        st.sidebar.success("Ortofoto faili augšupielādēti un izvilkti veiksmīgi!")
    elif uploaded_ortho:
        for file in uploaded_ortho:
            with open(os.path.join(ortho_dir, file.name), 'wb') as f:
                f.write(file.getbuffer())
        st.sidebar.success("Ortofoto faili augšupielādēti veiksmīgi!")
    else:
        st.sidebar.info("Lūdzu, augšupielādējiet Ortofoto failus.")
    
    # Nolasa SHP failus
    shp_files = glob.glob(os.path.join(shp_dir, '*.shp'))
    
    if shp_files:
        st.write("### Atrasti SHP faili:")
        for shp in shp_files:
            st.write(f"- {os.path.basename(shp)}")
    else:
        st.warning("Nav atrasti SHP faili. Lūdzu, augšupielādējiet SHP ZIP arhīvu.")
    
    # Nolasa LandXML virsmas
    surface_interpolators = load_landxml(surface_dir)
    
    # Nolasa DEM virsmas
    surface_rasters = load_dem(surface_dir)
    
    # Nolasa Ortofoto
    ortho_dataset = load_ortho(ortho_dir)
    
    if not surface_interpolators and not surface_rasters:
        st.error("Nav atrastas derīgas virsmas apstrādei.")
        st.stop()
    else:
        st.success("Virsmas ielādētas veiksmīgi!")
    
    st.header("Šķērsgriezumu Profili")
    
    if shp_files and (surface_interpolators or surface_rasters):
        for shp_file in shp_files:
            lines_gdf = gpd.read_file(shp_file)
    
            if lines_gdf.crs is None:
                st.warning(f"SHP failam {os.path.basename(shp_file)} nav norādīta koordinātu sistēma.")
                continue
            else:
                st.write(f"SHP faila koordinātu sistēma: {lines_gdf.crs}")
    
            if lines_gdf.crs.to_epsg() != TARGET_EPSG:
                lines_gdf = lines_gdf.to_crs(epsg=TARGET_EPSG)
                st.write("Koordinātu sistēma pārvērsta uz EPSG:3059.")
    
            for idx, row in lines_gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    st.warning(f"Līnijas {idx} ģeometrija ir tukša.")
                    continue
    
                # Identifikators
                attribute_fields = {field.lower(): field for field in lines_gdf.columns}
                if 'id' in attribute_fields:
                    line_attribute_id = row[attribute_fields['id']]
                    if pd.isna(line_attribute_id) or str(line_attribute_id).strip() == '':
                        line_attribute_id = f"Line_{idx}"
                    else:
                        line_attribute_id = str(line_attribute_id).replace(',', '').strip()
                else:
                    line_attribute_id = f"Line_{idx}"
    
                st.write(f"### Līnijas {line_attribute_id}")
                st.write(f"X koordinātu diapazons: {geom.bounds[0]} - {geom.bounds[2]}")
                st.write(f"Y koordinātu diapazons: {geom.bounds[1]} - {geom.bounds[3]}")
    
                # Samazina līniju ar noteiktu punktu skaitu
                num_points = st.sidebar.number_input(f"Punktu skaits līnijā {line_attribute_id}", min_value=100, max_value=10000, value=500, key=line_attribute_id)
                distances = np.linspace(0, geom.length, num_points)
                points = [geom.interpolate(distance) for distance in distances]
    
                x_coords = np.array([point.x for point in points])
                y_coords = np.array([point.y for point in points])
    
                points_gdf = gpd.GeoDataFrame({'geometry': points}, crs=TARGET_EPSG)
    
                df_profile = pd.DataFrame({'Distance (m)': distances})
    
                # Interpolācija no LandXML virsmas
                for surface_name, surface_data in surface_interpolators.items():
                    interpolator = surface_data['interpolator']
                    tri = surface_data['tri']
                    z_values = interpolator(x_coords, y_coords)
                    z_values = np.array(z_values)
    
                    # Diagnostika
                    st.write(f"🔍 Interpolācija priekš {surface_name}:")
                    st.write(f"   Izmantojot {len(surface_data['points_xy'])} punktus triangulācijā.")
    
                    # Pārbauda, vai kādi z vērtības ir iegūtas
                    if np.all(np.isnan(z_values)):
                        st.warning(f"Interpolācija nevarēja atrast z vērtības virsmas {surface_name} līnijai {line_attribute_id}.")
                        # Mēģina 'nearest' metodi
                        st.write("   Mēģinājums ar 'nearest' metodi.")
                        z_values_nearest = griddata(
                            surface_data['points_xy'],
                            surface_data['z'],
                            (x_coords, y_coords),
                            method='nearest'
                        )
                        df_profile[f'Elevation_{surface_name}'] = z_values_nearest
                        st.write(f"   Pēc 'nearest' interpolācijas: {z_values_nearest[:5]} ...")
                        if np.all(np.isnan(z_values_nearest)):
                            st.error(f"   'Nearest' interpolācija arī neveica z vērtību iegūšanu virsmas {surface_name}.")
                        else:
                            st.success(f"   'Nearest' interpolācija veiksmīgi atrada z vērtības virsmas {surface_name}.")
                    else:
                        df_profile[f'Elevation_{surface_name}'] = z_values
                        st.success(f"   Interpolācija veiksmīgi atrada z vērtības virsmas {surface_name}.")
    
                # Interpolācija no DEM virsmas
                for dem_name, dem_dataset in surface_rasters.items():
                    try:
                        row_indices, col_indices = dem_dataset.index(x_coords, y_coords)
                        dem_data = dem_dataset.read(1)
                        nodata = dem_dataset.nodata
                        z_dem = []
                        for row_idx, col_idx in zip(row_indices, col_indices):
                            try:
                                value = dem_data[row_idx, col_idx]
                                if value == nodata:
                                    z_dem.append(np.nan)
                                else:
                                    z_dem.append(value)
                            except IndexError:
                                z_dem.append(np.nan)
                        z_dem = np.array(z_dem)
                        df_profile[f'Elevation_{dem_name}'] = z_dem
    
                        # Diagnostika
                        st.write(f"🔍 Interpolācija priekš DEM virsmas {dem_name}:")
                        if np.all(np.isnan(z_dem)):
                            st.warning(f"Interpolācija nevarēja atrast z vērtības DEM virsmai {dem_name} līnijai {line_attribute_id}.")
                        else:
                            st.success(f"Interpolācija veiksmīgi atrada z vērtības DEM virsmai {dem_name}.")
                    except Exception as e:
                        st.error(f"Kļūda interpolācijas no DEM virsmas {dem_name}: {e}")
    
                # Pārbauda, vai ir pievienoti kādi elevācijas kolonnas
                elevation_columns = [col for col in df_profile.columns if col.startswith('Elevation_')]
                if not elevation_columns:
                    st.warning(f"Līnijai {line_attribute_id} nav pievienotas elevācijas kolonnas.")
                    continue
    
                # Pievieno vizualizāciju
                fig = go.Figure()
                for elev_col in elevation_columns:
                    fig.add_trace(go.Scatter(
                        x=df_profile['Distance (m)'],
                        y=df_profile[elev_col],
                        mode='lines',
                        name=elev_col.replace('Elevation_', '')
                    ))
    
                fig.update_layout(
                    title=f"Šķērsgriezuma Profils: {line_attribute_id}",
                    xaxis_title='Attālums gar līniju (m)',
                    yaxis_title='Augstums (m)',
                    template='plotly_white',
                    height=500,
                    autosize=True
                )
    
                st.plotly_chart(fig, use_container_width=True)
    
                # Vizualizē karti ar griezumu un punktiem
                if ortho_dataset:
                    map_image = generate_map_image(geom, points_gdf, ortho_dataset)
                    st.image(f'data:image/png;base64,{map_image}', use_container_width=True, caption=f'Karte griezumam {line_attribute_id}')
                else:
                    st.write("Ortofoto nav ielādēts. Karte netiks parādīta.")
    
    st.success("Apstrāde pabeigta!")
