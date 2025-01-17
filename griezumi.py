import os
import glob
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, Point, Polygon, MultiLineString
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


# ===========================================================
#    Palīgfunkcijas
# ===========================================================

def read_landxml_surfaces(landxml_files):
    """
    Nolasa LandXML virsmas un atgriež interpolatoru vārdnīcu, kā arī virsmas tipu vārdnīcu.
    """
    surface_interpolators = {}
    surface_types = {}

    for landxml_file in landxml_files:
        surface_name = os.path.splitext(os.path.basename(landxml_file))[0]
        surface_name = surface_name + '_xml'
        st.write(f"**Nolasa LandXML virsmu**: {surface_name}")

        # LandXML parse
        tree = ET.parse(landxml_file)
        root = tree.getroot()
        # ns = {'landxml': 'http://www.landxml.org/schema/LandXML-1.0'}
        # bet ja nav vajadzīgs, var nelikt

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

        # Ja nav atrasti punkti
        if not all_coords:
            st.warning(f"Virsmā {surface_name} nav atrasti punkti.")
            continue

        # Sagatavo interpolatoru
        coords = np.array(all_coords)
        coords_xy = coords[:, :2]
        z_coords = coords[:, 2]

        if coords.shape[0] < 3:
            st.warning(f"Nepietiek punktu virsmas {surface_name} interpolācijai.")
            continue

        st.write(f"Izveido interpolāciju virsmai *{surface_name}*...")
        tri = Delaunay(coords_xy)
        interpolator = LinearNDInterpolator(tri, z_coords)

        # Saglabājam rezultātus
        surface_interpolators[surface_name] = interpolator
        surface_types[surface_name] = 'landxml'

        st.write(f"Virsmā {surface_name} ir **{len(z_coords)}** punkti.")

    return surface_interpolators, surface_types


def read_dem_surfaces(dem_files):
    """
    Nolasa DEM virsmas, pārveido tās, ja nepieciešams, uz EPSG:3059 un saglabā atgriežot rastru vārdnīcu, kā arī virsmas tipu vārdnīcu.
    """
    surface_rasters = {}
    surface_types = {}

    for dem_file in dem_files:
        surface_name = os.path.splitext(os.path.basename(dem_file))[0]
        surface_name = surface_name + '_dem'
        st.write(f"**Nolasa DEM virsmu**: {surface_name}")

        dem_dataset = rasterio.open(dem_file)

        if dem_dataset.crs is not None and dem_dataset.crs.to_epsg() != 3059:
            st.info(f"Pārvērš DEM virsmas {surface_name} koordinātu sistēmu uz EPSG:3059.")
            vrt_params = {'crs': 'EPSG:3059'}
            dem_dataset = WarpedVRT(dem_dataset, **vrt_params)

        surface_rasters[surface_name] = dem_dataset
        surface_types[surface_name] = 'dem'

        st.write(f"DEM virsmas {surface_name} izmēri: {dem_dataset.width} x {dem_dataset.height}")

    return surface_rasters, surface_types


def generate_map_image(line_geom, points_gdf, ortho_dataset):
    """
    Ģenerē kartes attēlu (matplotlib) ar šķērsgriezuma līniju un punktiem, pārklājot tos uz ortofoto.
    Atgriež Base64 kodētu attēlu.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # Attēlo ortofoto ar rasterio
    rasterio.plot.show(ortho_dataset, ax=ax)

    # Attēlo līniju
    x_line, y_line = line_geom.xy
    ax.plot(x_line, y_line, color='red', linewidth=3, label='Griezuma līnija')

    # Attēlo punktus
    points_gdf.plot(ax=ax, marker='o', color='yellow', markersize=10, label='Punkti')

    # Noņem asis
    ax.axis('off')

    # Iestatām mērogu
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


# ===========================================================
#    Streamlit aplikācijas loģika
# ===========================================================
def main():
    st.title("Šķērsgriezumu Profili – DEM un LandXML analīze")

    # -------------------------------
    # 1) Failu augšupielāde
    # -------------------------------
    st.header("1. Augšupielādējiet datus")

    # LandXML un DEM
    landxml_files = st.file_uploader(
        "LandXML faili (var atlasīt vairākus):",
        accept_multiple_files=True,
        type=['xml', 'landxml']
    )
    dem_files = st.file_uploader(
        "DEM faili (GeoTIFF, var atlasīt vairākus):",
        accept_multiple_files=True,
        type=['tif']
    )

    # Ortofoto fails
    ortho_file = st.file_uploader(
        "Ortofoto fails (GeoTIFF):",
        type=['tif']
    )

    # SHP fails
    shp_files = st.file_uploader(
        "SHP faili ar šķērsgriezuma līnijām (var atlasīt vairākus):",
        accept_multiple_files=True,
        type=['shp']
    )
    # Piezīme: lai veiksmīgi augšupielādētu shapefile, jāaugšupielādē visa .shp, .dbf, .shx, .prj komplektācija
    # Streamlit šo var pārvaldīt, bet bieži ērtāk lietot .zip ar visu saturu.

    if st.button("Sākt apstrādi"):

        # -------------------------------
        # 2) Apstrādā LandXML un DEM
        # -------------------------------
        surface_interpolators = {}
        surface_rasters = {}
        surface_types = {}

        if landxml_files:
            # Pagaidu saglabāšana atmiņā
            saved_landxml = []
            for uploaded_file in landxml_files:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_landxml.append(uploaded_file.name)

            s_intrp, s_types = read_landxml_surfaces(saved_landxml)
            surface_interpolators.update(s_intrp)
            surface_types.update(s_types)

        if dem_files:
            saved_dem = []
            for uploaded_file in dem_files:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_dem.append(uploaded_file.name)

            s_rasts, s_types = read_dem_surfaces(saved_dem)
            surface_rasters.update(s_rasts)
            surface_types.update(s_types)

        # -------------------------------
        # 3) Ortofoto ielāde
        # -------------------------------
        ortho_dataset = None
        if ortho_file is not None:
            with open(ortho_file.name, "wb") as f:
                f.write(ortho_file.getbuffer())
            ds = rasterio.open(ortho_file.name)
            # Pārliecināmies par koord. sistēmu (EPSG:3059)
            if ds.crs and ds.crs.to_epsg() != 3059:
                st.info("Pārvērš ortofoto koordinātu sistēmu uz EPSG:3059.")
                vrt_params = {'crs': 'EPSG:3059'}
                ortho_dataset = WarpedVRT(ds, **vrt_params)
            else:
                ortho_dataset = ds

        # -------------------------------
        # 4) Nolasa SHP un veido profilu
        # -------------------------------
        if shp_files:
            for shp_upload in shp_files:
                # Saglabājam
                with open(shp_upload.name, "wb") as f:
                    f.write(shp_upload.getbuffer())

            # Noskaidrojam visus .shp failus, kas ir saglabāti
            all_shp = glob.glob("*.shp")
            for shp_file in all_shp:
                st.subheader(f"Apstrādā SHP failu: {shp_file}")
                lines_gdf = gpd.read_file(shp_file)

                if lines_gdf.crs is None:
                    st.warning(f"SHP failam {shp_file} nav norādīta koordinātu sistēma.")
                    # Ja zināt, kāda koordinātu sistēma nepieciešama, var norādīt to šeit:
                    # lines_gdf.crs = 'EPSG:3059'
                else:
                    st.write(f"SHP faila {shp_file} koordinātu sistēma: {lines_gdf.crs}")

                if lines_gdf.crs and lines_gdf.crs.to_epsg() != 3059:
                    st.info(f"Pārvērš SHP faila koordinātu sistēmu uz EPSG:3059.")
                    lines_gdf = lines_gdf.to_crs(epsg=3059)

                # Apstrādā katru līniju
                for idx, row in lines_gdf.iterrows():
                    geom = row.geometry
                    if geom is None:
                        continue

                    # Iegūstam ID
                    attribute_fields = {field.lower(): field for field in lines_gdf.columns}
                    if 'id' in attribute_fields:
                        line_attribute_id = row[attribute_fields['id']]
                        if pd.isna(line_attribute_id) or str(line_attribute_id).strip() == '':
                            st.warning(f"Brīdinājums: Līnijai indeksā {idx} nav ID.")
                            line_attribute_id = "nav_ID"
                        else:
                            line_attribute_id = str(line_attribute_id).replace(',', '').strip()
                    else:
                        st.warning("Brīdinājums: nav atrasts lauks 'ID'.")
                        line_attribute_id = "nav_ID"

                    # Apstrādā LineString vai MultiLineString
                    if geom.geom_type == 'LineString':
                        line_list = [(geom, 0)]
                    elif geom.geom_type == 'MultiLineString':
                        line_list = [(part, i) for i, part in enumerate(geom)]
                    else:
                        continue

                    for line_part, part_idx in line_list:
                        line_id = f"{os.path.splitext(os.path.basename(shp_file))[0]}_{line_attribute_id}_part{part_idx}"

                        st.write(f"**Līnija**: {line_id}")
                        st.write(
                            f"X diapazons: {line_part.bounds[0]} - {line_part.bounds[2]} | "
                            f"Y diapazons: {line_part.bounds[1]} - {line_part.bounds[3]}"
                        )

                        # Ģenerējam punktus gar līniju
                        num_points = 500
                        distances = np.linspace(0, line_part.length, num_points)
                        points = [line_part.interpolate(distance) for distance in distances]

                        x_coords = np.array([p.x for p in points])
                        y_coords = np.array([p.y for p in points])

                        points_gdf = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:3059')

                        # DataFrame sagatave
                        df = pd.DataFrame({'Distance': distances})

                        # Interpolē virsmas
                        for s_name in surface_types.keys():
                            s_type = surface_types[s_name]

                            if s_type == 'landxml':
                                interpolator = surface_interpolators[s_name]
                                # LandXML virsmām mainām (x <-> y)
                                z_values = interpolator(y_coords, x_coords)
                                if z_values is not None:
                                    nan_idx = np.isnan(z_values)
                                    if np.all(nan_idx):
                                        st.warning(f"Visi punkti atrodas ārpus LandXML virsmas {s_name}.")
                                    elif np.any(nan_idx):
                                        st.warning(f"Daļa punktu atrodas ārpus LandXML virsmas {s_name}.")
                                    df[f'Elev_{s_name}'] = z_values

                            elif s_type == 'dem':
                                dem_dataset = surface_rasters[s_name]
                                row_indices, col_indices = dem_dataset.index(x_coords, y_coords)
                                z_vals = []
                                dem_data = dem_dataset.read(1)
                                nodata = dem_dataset.nodata

                                for r_i, c_i in zip(row_indices, col_indices):
                                    try:
                                        val = dem_data[r_i, c_i]
                                        if val == nodata:
                                            z_vals.append(np.nan)
                                        else:
                                            z_vals.append(val)
                                    except IndexError:
                                        z_vals.append(np.nan)

                                z_values = np.array(z_vals)
                                if np.all(np.isnan(z_values)):
                                    st.warning(f"Visi punkti atrodas ārpus DEM virsmas {s_name}.")
                                elif np.any(np.isnan(z_values)):
                                    st.warning(f"Daļa punktu atrodas ārpus DEM virsmas {s_name}.")

                                df[f'Elev_{s_name}'] = z_values

                        # Vai ir kāda Elev_ kolonna
                        elevation_cols = [col for col in df.columns if col.startswith('Elev_')]
                        if not elevation_cols:
                            st.info(f"Līnijai {line_id} netika iegūtas Z vērtības.")
                            continue

                        # Parādām galviņu
                        st.dataframe(df.head())

                        # 5.1. Plotly grafiks
                        fig = go.Figure()
                        for e_col in elevation_cols:
                            s_name = e_col.replace('Elev_', '')
                            fig.add_trace(
                                go.Scatter(
                                    x=df['Distance'],
                                    y=df[e_col],
                                    mode='lines',
                                    name=s_name
                                )
                            )

                        fig.update_layout(
                            xaxis_title='Attālums gar līniju (m)',
                            yaxis_title='Augstums (m)',
                            template='plotly_white',
                            height=400,
                            margin=dict(l=50, r=50, t=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 5.2. Ortofoto attēls (ja pieejams ortho_dataset)
                        if ortho_dataset:
                            map_image_base64 = generate_map_image(line_part, points_gdf, ortho_dataset)
                            st.markdown(
                                f"<div style='text-align:center;'>"
                                f"<img src='data:image/png;base64,{map_image_base64}' "
                                f"alt='Karte griezumam: {line_id}' style='max-width:100%;'/>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

        else:
            st.warning("Nav izvēlēti SHP faili vai netika aktivizēta apstrāde.")


# ===========================================================
#    Palaižam aplikāciju
# ===========================================================
if __name__ == "__main__":
    main()
