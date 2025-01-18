# app.py

import os
import glob
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import rasterio
from rasterio.vrt import WarpedVRT
import rasterio.plot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
from io import BytesIO
import streamlit as st
import tempfile

# === 1. Definēt ceļus ===

st.title("Šķērsgriezumu Profili")

st.sidebar.header("Iestatījumi")

with tempfile.TemporaryDirectory() as tmpdirname:
    shp_dir = os.path.join(tmpdirname, 'linijas')
    surface_dir = os.path.join(tmpdirname, 'virsmas')
    ortho_dir = os.path.join(tmpdirname, 'ortofoto')
    output_dir = os.path.join(tmpdirname, 'out')

    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(ortho_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    st.write("## Repozitorija Informācija")
    st.write("Šī lietotne analizē SHP failus, virsmas (LandXML un DEM) un ortofoto datus, lai ģenerētu šķērsgriezumu profilus.")

    # === 2. Datu Augšupielāde ===

    st.sidebar.header("Datu Ielāde")

    # 2.1. Augšupielādēt Linijas (SHP) ZIP arhīvu
    st.sidebar.subheader("Linijas (SHP)")
    uploaded_shp_zip = st.sidebar.file_uploader("Augšupielādēt Linijas ZIP (satur .shp, .shx, .dbf, utt.)", type=["zip"], key="shp_zip")

    if uploaded_shp_zip is not None:
        with zipfile.ZipFile(uploaded_shp_zip) as z:
            z.extractall(shp_dir)
        st.sidebar.success("Linijas faili augšupielādēti un izvilkti veiksmīgi!")
    else:
        st.sidebar.info("Lūdzu, augšupielādējiet Linijas ZIP arhīvu.")

    # 2.2. Augšupielādēt Virsmas (LandXML un DEM)
    st.sidebar.subheader("Virsmas (LandXML un DEM)")
    uploaded_landxml = st.sidebar.file_uploader("Augšupielādēt LandXML faili (.xml)", type=["xml"], accept_multiple_files=True, key="landxml")
    uploaded_dem = st.sidebar.file_uploader("Augšupielādēt DEM faili (.tif)", type=["tif"], accept_multiple_files=True, key="dem")

    if uploaded_landxml:
        for file in uploaded_landxml:
            with open(os.path.join(surface_dir, file.name), 'wb') as f:
                f.write(file.getbuffer())
        st.sidebar.success("LandXML faili augšupielādēti veiksmīgi!")

    if uploaded_dem:
        for file in uploaded_dem:
            with open(os.path.join(surface_dir, file.name), 'wb') as f:
                f.write(file.getbuffer())
        st.sidebar.success("DEM faili augšupielādēti veiksmīgi!")

    if not (uploaded_landxml or uploaded_dem):
        st.sidebar.info("Lūdzu, augšupielādējiet Virsmas failus.")

    # 2.3. Augšupielādēt Ortofoto (GeoTIFF vai ZIP)
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

    # === 3. Nolasa SHP failus no mapes ===

    st.header("Ielādētie Dati")

    @st.cache_data
    def load_shp_files(shp_directory):
        shp_files = glob.glob(os.path.join(shp_directory, '*.shp'))
        return shp_files

    shp_files = load_shp_files(shp_dir)

    if shp_files:
        st.write("### Atrasti SHP faili:")
        for shp in shp_files:
            st.write(f"- {os.path.basename(shp)}")
    else:
        st.stop()

    # === 4. Nolasa virsmas (LandXML un DEM) un izveido interpolatorus ===

    @st.cache_data
    def load_surfaces(surface_directory):
        surface_interpolators = {}
        surface_rasters = {}
        surface_types = {}

        landxml_files = glob.glob(os.path.join(surface_directory, '*.xml'))
        dem_files = glob.glob(os.path.join(surface_directory, '*.tif'))

        for landxml_file in landxml_files:
            surface_name = os.path.splitext(os.path.basename(landxml_file))[0] + '_xml'
            tree = ET.parse(landxml_file)
            root = tree.getroot()

            all_coords = []
            pnt_list3d = root.findall('.//{*}PntList3D')
            for pnt_list in pnt_list3d:
                text = pnt_list.text.strip()
                points = text.split()
                for i in range(0, len(points), 3):
                    x = float(points[i])
                    y = float(points[i+1])
                    z = float(points[i+2])
                    all_coords.append([x, y, z])

            p_elements = root.findall('.//{*}P')
            for p in p_elements:
                text = p.text.strip()
                if text:
                    points = text.split()
                    x = float(points[0])
                    y = float(points[1])
                    z = float(points[2])
                    all_coords.append([x, y, z])

            if not all_coords:
                continue

            coords = np.array(all_coords)
            coords_xy = coords[:, :2]
            z_coords = coords[:, 2]

            if coords.shape[0] < 3:
                continue

            tri = Delaunay(coords_xy)
            interpolator = LinearNDInterpolator(tri, z_coords)

            surface_interpolators[surface_name] = interpolator
            surface_types[surface_name] = 'landxml'

        for dem_file in dem_files:
            surface_name = os.path.splitext(os.path.basename(dem_file))[0] + '_dem'
            dem_dataset = rasterio.open(dem_file)

            if dem_dataset.crs.to_epsg() != 3059:
                vrt_params = {'crs': 'EPSG:3059'}
                dem_dataset = WarpedVRT(dem_dataset, **vrt_params)

            surface_rasters[surface_name] = dem_dataset
            surface_types[surface_name] = 'dem'

        return surface_interpolators, surface_rasters, surface_types

    surface_interpolators, surface_rasters, surface_types = load_surfaces(surface_dir)

    if not surface_interpolators and not surface_rasters:
        st.error("Nav atrastas derīgas virsmas apstrādei.")
        st.stop()
    else:
        st.success("Virsmas ielādētas veiksmīgi!")

    # === 5. Nolasa ortofoto ===

    @st.cache_resource
    def load_ortho(ortho_directory):
        ortho_files = glob.glob(os.path.join(ortho_directory, '*.tif'))
        if not ortho_files:
            return None

        ortho_file = ortho_files[0]
        ortho_dataset = rasterio.open(ortho_file)

        if ortho_dataset.crs.to_epsg() != 3059:
            vrt_params = {'crs': 'EPSG:3059'}
            ortho_dataset = WarpedVRT(ortho_dataset, **vrt_params)

        return ortho_dataset

    ortho_dataset = load_ortho(ortho_dir)

    if ortho_dataset:
        st.success(f"Ortofoto ielādēts: {os.path.basename(ortho_dataset.name)}")
    else:
        st.stop()

    # === 6. Apstrādā katru SHP failu ===

    def generate_map_image(line_geom, points_gdf, ortho_dataset):
        fig, ax = plt.subplots(figsize=(8, 8))

        rasterio.plot.show(ortho_dataset, ax=ax)
        x_line, y_line = line_geom.xy
        ax.plot(x_line, y_line, color='red', linewidth=3, label='Griezuma līnija')
        points_gdf.plot(ax=ax, marker='o', color='yellow', markersize=10, label='Punkti')

        ax.axis('off')

        all_geoms = [line_geom] + list(points_gdf.geometry)
        total_bounds = gpd.GeoSeries(all_geoms).total_bounds
        x_min, y_min, x_max, y_max = total_bounds
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64

    st.header("Šķērsgriezumu Profili")

    for shp_file in shp_files:
        lines_gdf = gpd.read_file(shp_file)

        if lines_gdf.crs is None:
            st.warning(f"SHP failam {os.path.basename(shp_file)} nav norādīta koordinātu sistēma.")
        else:
            # Šī rinda ir noņemta, lai neatspoguļotu koordinātu sistēmu
            # st.write(f"SHP faila koordinātu sistēma: {lines_gdf.crs}")
            pass

        if lines_gdf.crs and lines_gdf.crs.to_epsg() != 3059:
            lines_gdf = lines_gdf.to_crs(epsg=3059)
            st.write("Koordinātu sistēma pārvērsta uz EPSG:3059.")

        for idx, row in lines_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue

            attribute_fields = {field.lower(): field for field in lines_gdf.columns}
            if 'id' in attribute_fields:
                line_attribute_id = row[attribute_fields['id']]
                if pd.isna(line_attribute_id) or str(line_attribute_id).strip() == '':
                    line_attribute_id = "nav ID"
                else:
                    line_attribute_id = str(line_attribute_id).replace(',', '').strip()
            else:
                line_attribute_id = "nav ID"

            if geom.geom_type == 'LineString':
                line = geom
                line_id = f"{os.path.splitext(os.path.basename(shp_file))[0]}_{line_attribute_id}"
                parts = [line]
            elif geom.geom_type == 'MultiLineString':
                parts = geom
            else:
                continue

            for part_idx, part in enumerate(parts):
                if geom.geom_type == 'LineString':
                    current_line_id = line_id
                else:
                    current_line_id = f"{os.path.splitext(os.path.basename(shp_file))[0]}_{line_attribute_id}_part{part_idx}"

                # Šīs rindas ir noņemtas, lai neatspoguļotu Līnijas ID un koordinātu diapazonus
                # st.write(f"### Līnijas {current_line_id}")
                # st.write(f"X koordinātu diapazons: {part.bounds[0]} - {part.bounds[2]}")
                # st.write(f"Y koordinātu diapazons: {part.bounds[1]} - {part.bounds[3]}")

                num_points = st.sidebar.number_input(f"Punktu skaits līnijā {current_line_id}", min_value=100, max_value=1000, value=500, key=current_line_id)
                distances = np.linspace(0, part.length, num_points)
                points = [part.interpolate(distance) for distance in distances]

                x_coords = np.array([point.x for point in points])
                y_coords = np.array([point.y for point in points])

                points_gdf = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:3059')

                df = pd.DataFrame({'Distance': distances})

                for surface_name in surface_types.keys():
                    surface_type = surface_types[surface_name]

                    if surface_type == 'landxml':
                        interpolator = surface_interpolators[surface_name]
                        x_coords_swapped, y_coords_swapped = y_coords, x_coords
                        z_values = interpolator(x_coords_swapped, y_coords_swapped)
                        nan_indices = np.isnan(z_values)
                        if np.all(nan_indices):
                            continue
                        df[f'Elevation_{surface_name}'] = z_values

                    elif surface_type == 'dem':
                        dem_dataset = surface_rasters[surface_name]
                        row_indices, col_indices = dem_dataset.index(x_coords, y_coords)
                        z_values = []
                        dem_data = dem_dataset.read(1)
                        nodata = dem_dataset.nodata
                        for row_idx, col_idx in zip(row_indices, col_indices):
                            try:
                                value = dem_data[row_idx, col_idx]
                                if value == nodata:
                                    z_values.append(np.nan)
                                else:
                                    z_values.append(value)
                            except IndexError:
                                z_values.append(np.nan)
                        z_values = np.array(z_values)
                        df[f'Elevation_{surface_name}'] = z_values

                elevation_columns = [col for col in df.columns if col.startswith('Elevation_')]
                if not elevation_columns:
                    continue

                # Šīs rindas ir noņemtas, lai neatspoguļotu Datu Pārskatu tabulu
                # st.write("#### Datu Pārskats")
                # st.dataframe(df.head())

                fig = go.Figure()

                for elevation_col in elevation_columns:
                    surface_name = elevation_col.replace('Elevation_', '')
                    fig.add_trace(go.Scatter(
                        x=df['Distance'],
                        y=df[elevation_col],
                        mode='lines',
                        name=surface_name,
                        connectgaps=False
                    ))

                fig.update_layout(
                    xaxis_title='Attālums gar līniju (m)',
                    yaxis_title='Augstums (m)',
                    template='plotly_white',
                    height=500,
                    autosize=True,
                    margin=dict(l=50, r=50, t=50, b=50),
                    xaxis=dict(range=[0, part.length])
                )

                fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'plot_{current_line_id}', config={'responsive': True})

                map_image_base64 = generate_map_image(part, points_gdf, ortho_dataset)

                if isinstance(line_attribute_id, (int, float)):
                    line_attribute_id_str = f"{line_attribute_id}"
                else:
                    line_attribute_id_str = str(line_attribute_id).replace(',', '').strip()

                st.markdown(f"### Griezums: {line_attribute_id_str}")
                st.image(f'data:image/png;base64,{map_image_base64}', use_container_width=True, caption=f'Karte griezumam {line_attribute_id_str}')
                st.plotly_chart(fig, use_container_width=True)

    st.success("Apstrāde pabeigta!")
