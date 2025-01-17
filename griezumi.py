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

# ===================================================================
# 1. Papildu funkcija, lai pārbaudītu Shapefile kopumu
# ===================================================================
def check_shapefile_completeness(shp_file):
    """
    Pārbauda, vai Shapefile (shp, shx, dbf, prj) komplekts ir pilns.
    Ja trūkst kāds no failiem, izmet FileNotFoundError.
    """
    required_extensions = [".shp", ".shx", ".dbf", ".prj"]  # Varat pielāgot atbilstoši vajadzībai

    # No ceļa “fails.shp” atdalīt pamatnosaukumu “fails”
    shp_base, shp_ext = os.path.splitext(shp_file)
    missing_files = []
    for ext in required_extensions:
        candidate = shp_base + ext
        if not os.path.exists(candidate):
            missing_files.append(candidate)

    if missing_files:
        raise FileNotFoundError(
            "Shapefile komplekts nav pilnīgs. Trūkst šādi faili:\n  " + "\n  ".join(missing_files)
        )

# ===================================================================
# 2. Definēt ceļus
# ===================================================================

# SHP līniju ceļš
shp_dir = r'C:\EF\skripti\GRIEZUMI_landXML_DEM\DATI\linijas'

# Virsmu (LandXML un DEM) ceļš
surface_dir = r'C:\EF\skripti\GRIEZUMI_landXML_DEM\DATI\virsmas'

# Ortofoto ceļš
ortho_dir = r'C:\EF\skripti\GRIEZUMI_landXML_DEM\DATI\ortofoto'

# Izvades ceļš
output_dir = r'C:\EF\skripti\GRIEZUMI_landXML_DEM\DATI\out'

# Pārliecināties, ka izvades mape eksistē
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ===================================================================
# 3. Nolasa SHP failus no mapes
# ===================================================================

print("Nolasa SHP failus...")

# Atrod visus SHP failus mapē 'linijas'
shp_files = glob.glob(os.path.join(shp_dir, '*.shp'))

if not shp_files:
    print("Mapē nav atrasti SHP faili. Lūdzu, pārbaudiet mapes ceļu un failus.")
    exit()

# ===================================================================
# 4. Nolasa virsmas (LandXML un DEM) un izveido interpolatorus
# ===================================================================

print("Nolasa virsmas...")

# Izveido vārdnīcu, kurā glabāsies interpolatori vai DEM objekti katrai virsmai
surface_interpolators = {}
surface_rasters = {}
surface_types = {}  # 'landxml' vai 'dem'

# Atrod visus LandXML un DEM failus norādītajā mapē
landxml_files = glob.glob(os.path.join(surface_dir, '*.xml'))
dem_files = glob.glob(os.path.join(surface_dir, '*.tif'))  # Pieņemot, ka DEM faili ir GeoTIFF formātā

if not landxml_files and not dem_files:
    print("Mapē nav atrasti LandXML vai DEM faili. Lūdzu, pārbaudiet mapes ceļu un failus.")
    exit()

# -----------------------------------------------------------
# 4.1. Apstrādā LandXML virsmas
# -----------------------------------------------------------
for landxml_file in landxml_files:
    surface_name = os.path.splitext(os.path.basename(landxml_file))[0]
    surface_name = surface_name + '_xml'  # Pievienot sufiksu, lai atšķirtu no DEM virsmām
    print(f"Nolasa LandXML virsmu: {surface_name}")
    tree = ET.parse(landxml_file)
    root = tree.getroot()
    ns = {'landxml': 'http://www.landxml.org/schema/LandXML-1.0'}

    # Izvilkt punktus no <PntList3D> elementiem
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

    # Izvilkt punktus no <P> elementiem
    p_elements = root.findall('.//{*}P')
    for p in p_elements:
        text = p.text.strip()
        if text:
            points = text.split()
            x = float(points[0])
            y = float(points[1])
            z = float(points[2])
            all_coords.append([x, y, z])

    # Pārbaudīt, vai ir atrasti punkti
    if not all_coords:
        print(f"Virsmā {surface_name} nav atrasti punkti.")
        continue

    # Pārveidot uz NumPy masīvu
    coords = np.array(all_coords)
    coords_xy = coords[:, :2]
    z_coords = coords[:, 2]

    # Pārbaudīt, vai ir pietiekami daudz punktu interpolācijai
    if coords.shape[0] < 3:
        print(f"Nepietiek punktu virsmas {surface_name} interpolācijai.")
        continue

    # Izveidot interpolācijas funkciju virsmai
    print(f"Izveido interpolācijas funkciju virsmai {surface_name}...")
    tri = Delaunay(coords_xy)
    interpolator = LinearNDInterpolator(tri, z_coords)

    # Saglabāt interpolatoru un virsmas tipu
    surface_interpolators[surface_name] = interpolator
    surface_types[surface_name] = 'landxml'

    # Izdrukāt virsmas informāciju
    print(f"Virsmā {surface_name} ir {len(z_coords)} punkti.")

# -----------------------------------------------------------
# 4.2. Apstrādā DEM virsmas
# -----------------------------------------------------------
for dem_file in dem_files:
    surface_name = os.path.splitext(os.path.basename(dem_file))[0]
    surface_name = surface_name + '_dem'  # Pievienot sufiksu, lai atšķirtu no LandXML virsmām
    print(f"Nolasa DEM virsmu: {surface_name}")
    dem_dataset = rasterio.open(dem_file)

    # Pārliecināties, ka DEM ir EPSG:3059 koordinātu sistēmā
    if dem_dataset.crs.to_epsg() != 3059:
        print(f"Pārvērš DEM virsmas {surface_name} koordinātu sistēmā uz EPSG:3059.")
        vrt_params = {'crs': 'EPSG:3059'}
        dem_dataset = WarpedVRT(dem_dataset, **vrt_params)

    # Saglabāt DEM datasetu un virsmas tipu
    surface_rasters[surface_name] = dem_dataset
    surface_types[surface_name] = 'dem'

    # Izdrukāt DEM informāciju
    print(f"DEM virsmas {surface_name} izmēri: {dem_dataset.width} x {dem_dataset.height}")

# Pārbaudīt, vai ir izveidotas virsmas
if not surface_interpolators and not surface_rasters:
    print("Nav atrastas derīgas virsmas apstrādei.")
    exit()

# ===================================================================
# 5. Nolasa ortofoto
# ===================================================================
ortho_files = glob.glob(os.path.join(ortho_dir, '*.tif'))

if not ortho_files:
    print("Mapē nav atrasti ortofoto faili. Lūdzu, pārbaudiet mapes ceļu un failus.")
    exit()

# Pieņemam, ka ir viens ortofoto fails
ortho_file = ortho_files[0]
print(f"Nolasa ortofoto: {ortho_file}")
ortho_dataset = rasterio.open(ortho_file)

# Pārliecināties, ka ortofoto ir EPSG:3059 koordinātu sistēmā
if ortho_dataset.crs.to_epsg() != 3059:
    print(f"Pārvērš ortofoto koordinātu sistēmu uz EPSG:3059.")
    vrt_params = {'crs': 'EPSG:3059'}
    ortho_dataset = WarpedVRT(ortho_dataset, **vrt_params)

# ===================================================================
# 6. Sagatavojam HTML satura daļas
# ===================================================================
html_header = '''
<html>
<head>
    <title>Šķērsgriezumu Profili</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
    body {
        font-family: Arial, sans-serif;
        margin: 20px;
    }
    h2 {
        margin-top: 50px;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 30px;
    }
    th, td {
        border: 1px solid black;
        text-align: left;
        padding: 5px;
    }
    img {
        max-width: 100%;
        height: auto;
        margin-bottom: 20px;
    }
    /* Nodrošina, ka Plotly grafiki aizņem visu pieejamo platumu */
    .plotly-graph-div {
        width: 100% !important;
    }
    </style>
</head>
<body>
'''
html_content = html_header

# JavaScript kods interaktivitātei
javascript_code = '''
<script>
    // JavaScript kods interaktivitātei
</script>
'''

# ===================================================================
# 7. Palīgfunkcija ortofoto attēlam
# ===================================================================
def generate_map_image(line_geom, points_gdf, ortho_dataset):
    """
    Ģenerē kartes attēlu ar šķērsgriezuma līniju un punktiem, pārklājot tos uz ortofoto.
    Attēlu saglabā kā Base64 kodētu virkni.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Attēlojam ortofoto
    rasterio.plot.show(ortho_dataset, ax=ax)

    # Attēlojam griezuma līniju
    x_line, y_line = line_geom.xy
    ax.plot(x_line, y_line, color='red', linewidth=3, label='Griezuma līnija')  # Palielinām linewidth

    # Attēlojam punktus
    points_gdf.plot(ax=ax, marker='o', color='yellow', markersize=10, label='Punkti')  # Samazinām markersize un mainām krāsu

    # Noņemam asu nosaukumus
    ax.axis('off')

    # Iestatām kartes robežas pēc griezuma un punktu ģeometrijām
    all_geoms = [line_geom] + list(points_gdf.geometry)
    total_bounds = gpd.GeoSeries(all_geoms).total_bounds
    x_min, y_min, x_max, y_max = total_bounds
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Saglabājam attēlu BytesIO objektā
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

# ===================================================================
# 8. Apstrādā katru SHP failu
# ===================================================================
for shp_file in shp_files:
    print(f"Apstrādā SHP failu: {shp_file}")

    # ---------------------------------------------------------------
    # Jaunums: pārbaudām, vai ir klāt .shp, .shx, .dbf, .prj utt.
    # ---------------------------------------------------------------
    check_shapefile_completeness(shp_file)

    lines_gdf = gpd.read_file(shp_file)

    # Pārbaudīt SHP faila koordinātu sistēmu
    if lines_gdf.crs is None:
        print(f"SHP failam {shp_file} nav norādīta koordinātu sistēma.")
        # Ja zināt, kāda koordinātu sistēma tiek izmantota, piešķiriet to šeit
        # lines_gdf.crs = 'EPSG:XXXX'
    else:
        print(f"SHP faila {shp_file} koordinātu sistēma: {lines_gdf.crs}")

    # Ja nepieciešams, pārvērst uz EPSG:3059
    if lines_gdf.crs and lines_gdf.crs.to_epsg() != 3059:
        print(f"Pārvērš SHP faila koordinātu sistēmu uz EPSG:3059.")
        lines_gdf = lines_gdf.to_crs(epsg=3059)

    # Apstrādā katru līniju SHP failā
    for idx, row in lines_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # Izgūstam ID no atribūtu tabulas (case-insensitive)
        attribute_fields = {field.lower(): field for field in lines_gdf.columns}
        if 'id' in attribute_fields:
            line_attribute_id = row[attribute_fields['id']]
            # Pārbaudām, vai ID nav tukšs
            if pd.isna(line_attribute_id) or str(line_attribute_id).strip() == '':
                print(f"Brīdinājums: SHP faila {shp_file} līnijai indeksā {idx} ID lauks ir tukšs.")
                line_attribute_id = "nav ID"
            else:
                # Nodrošinām, ka ID ir teksta formātā un nav komatiem
                line_attribute_id = str(line_attribute_id).replace(',', '').strip()
        else:
            print(f"Brīdinājums: SHP faila {shp_file} lauks 'id' vai 'ID' nav atrasts. Lieto 'nav ID'.")
            line_attribute_id = "nav ID"  # Ja nav ID lauka, lieto "nav ID"

        if geom.geom_type == 'LineString':
            line = geom
            # Izveidot unikālu identifikatoru līnijai
            line_id = f"{os.path.splitext(os.path.basename(shp_file))[0]}_{line_attribute_id}"
            # Zemāk turpinās apstrāde ar line

            # Izdrukāt līnijas koordinātu diapazonus
            print(f"Līnijas {line_id} X koordinātu diapazons: {line.bounds[0]} - {line.bounds[2]}")
            print(f"Līnijas {line_id} Y koordinātu diapazons: {line.bounds[1]} - {line.bounds[3]}")

            # Izveidot punktus gar līniju (sampling)
            num_points = 500  # Punktu skaits gar līniju, pielāgojiet pēc vajadzības
            distances = np.linspace(0, line.length, num_points)
            points = [line.interpolate(distance) for distance in distances]

            # Izvilkt X un Y koordinātas priekš attēlošanas (bez koordinātu apmaiņas)
            x_coords = np.array([point.x for point in points])
            y_coords = np.array([point.y for point in points])

            # Izveidot GeoDataFrame punktiem priekš attēlošanas
            points_gdf = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:3059')

            # Izveidot DataFrame šķērsgriezuma datiem
            df = pd.DataFrame({'Distance': distances})

            # Interpolēt Z vērtības no katras virsmas
            for surface_name in surface_types.keys():
                surface_type = surface_types[surface_name]

                if surface_type == 'landxml':
                    interpolator = surface_interpolators[surface_name]
                    # Apmainām koordinātas LandXML virsmām
                    x_coords_swapped, y_coords_swapped = y_coords, x_coords
                    # Interpolēt Z vērtības, izmantojot apmainītās koordinātas
                    z_values = interpolator(x_coords_swapped, y_coords_swapped)
                    # Apstrādāt NaN vērtības
                    nan_indices = np.isnan(z_values)
                    if np.all(nan_indices):
                        print(f"Visi punkti līnijā {line_id} atrodas ārpus virsmas {surface_name}.")
                        continue
                    elif np.any(nan_indices):
                        print(f"Brīdinājums: Daži punkti līnijā {line_id} atrodas ārpus virsmas {surface_name}.")
                    # Pievienot Z vērtības DataFrame
                    df[f'Elevation_{surface_name}'] = z_values

                elif surface_type == 'dem':
                    dem_dataset = surface_rasters[surface_name]
                    # DEM virsmām koordinātas atstājam nemainītas
                    # Pārveidot koordinātas uz rindu un kolonnu indeksiem
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
                    # Apstrādāt NaN vērtības
                    nan_indices = np.isnan(z_values)
                    if np.all(nan_indices):
                        print(f"Visi punkti līnijā {line_id} atrodas ārpus DEM virsmas {surface_name}.")
                        continue
                    elif np.any(nan_indices):
                        print(f"Brīdinājums: Daži punkti līnijā {line_id} atrodas ārpus DEM virsmas {surface_name}.")
                    # Pievienot Z vērtības DataFrame
                    df[f'Elevation_{surface_name}'] = z_values

            # Pārbaudīt, vai DataFrame satur Z vērtības
            elevation_columns = [col for col in df.columns if col.startswith('Elevation_')]
            if not elevation_columns:
                print(f"Līnija {line_id} nav derīgu Z vērtību. Izlaižam.")
                continue

            # Izdrukāt DataFrame kolonnas
            print(f"DataFrame kolonnas: {df.columns.tolist()}")
            # Izdrukāt DataFrame galveni
            print(df.head())

            # === 5.1. Izveidot grafiku ar Plotly ===
            fig = go.Figure()
            for elevation_col in elevation_columns:
                surface_name = elevation_col.replace('Elevation_', '')
                fig.add_trace(go.Scatter(
                    x=df['Distance'],
                    y=df[elevation_col],
                    mode='lines',
                    name=surface_name,
                    connectgaps=False  # Nenovērš pārrāvumus
                ))
            fig.update_layout(
                xaxis_title='Attālums gar līniju (m)',
                yaxis_title='Augstums (m)',
                template='plotly_white',
                height=500,  # Fiksēts vertikālais izmērs
                autosize=True,  # Ļauj Plotly automātiski pielāgot platumu
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(range=[0, line.length])
            )

            # Ģenerēt grafika HTML kodu ar responsīvu konfigurāciju
            fig_html = fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=f'plot_{line_id}',
                config={'responsive': True}
            )

            # === 5.2. Ģenerēt ortofoto attēlu kā Base64 ===
            map_image_base64 = generate_map_image(line, points_gdf, ortho_dataset)

            # === 5.3. Pievienot saturu HTML failam ===
            if isinstance(line_attribute_id, (int, float)):
                line_attribute_id_str = f"{line_attribute_id}"
            else:
                line_attribute_id_str = str(line_attribute_id).replace(',', '').strip()

            html_content += f"<h2 id='griezums_{line_id}'>Griezums: {line_attribute_id_str}</h2>\n"
            html_content += f"<div style='text-align: center;'><img src='data:image/png;base64,{map_image_base64}' alt='Karte griezumam {line_attribute_id_str}'></div>\n"
            html_content += fig_html + "\n"

        elif geom.geom_type == 'MultiLineString':
            # Ja ir MultiLineString, apstrādā katru daļu atsevišķi
            for part_idx, part in enumerate(geom):
                line = part
                # Izveidot unikālu identifikatoru
                line_id = f"{os.path.splitext(os.path.basename(shp_file))[0]}_{line_attribute_id}_part{part_idx}"

                print(f"Līnijas {line_id} X koordinātu diapazons: {line.bounds[0]} - {line.bounds[2]}")
                print(f"Līnijas {line_id} Y koordinātu diapazons: {line.bounds[1]} - {line.bounds[3]}")

                num_points = 500
                distances = np.linspace(0, line.length, num_points)
                points = [line.interpolate(distance) for distance in distances]
                x_coords = np.array([point.x for point in points])
                y_coords = np.array([point.y for point in points])
                points_gdf = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:3059')
                df = pd.DataFrame({'Distance': distances})

                # Interpolēt Z vērtības no katras virsmas
                for surface_name in surface_types.keys():
                    surface_type = surface_types[surface_name]
                    if surface_type == 'landxml':
                        interpolator = surface_interpolators[surface_name]
                        x_coords_swapped, y_coords_swapped = y_coords, x_coords
                        z_values = interpolator(x_coords_swapped, y_coords_swapped)
                        nan_indices = np.isnan(z_values)
                        if np.all(nan_indices):
                            print(f"Visi punkti līnijā {line_id} atrodas ārpus virsmas {surface_name}.")
                            continue
                        elif np.any(nan_indices):
                            print(f"Brīdinājums: Daži punkti līnijā {line_id} atrodas ārpus virsmas {surface_name}.")
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
                        nan_indices = np.isnan(z_values)
                        if np.all(nan_indices):
                            print(f"Visi punkti līnijā {line_id} atrodas ārpus DEM virsmas {surface_name}.")
                            continue
                        elif np.any(nan_indices):
                            print(f"Brīdinājums: Daži punkti līnijā {line_id} atrodas ārpus DEM virsmas {surface_name}.")
                        df[f'Elevation_{surface_name}'] = z_values

                elevation_columns = [col for col in df.columns if col.startswith('Elevation_')]
                if not elevation_columns:
                    print(f"Līnija {line_id} nav derīgu Z vērtību. Izlaižam.")
                    continue

                print(f"DataFrame kolonnas: {df.columns.tolist()}")
                print(df.head())

                fig = go.Figure()
                for elevation_col in elevation_columns:
                    s_name = elevation_col.replace('Elevation_', '')
                    fig.add_trace(go.Scatter(
                        x=df['Distance'],
                        y=df[elevation_col],
                        mode='lines',
                        name=s_name,
                        connectgaps=False
                    ))
                fig.update_layout(
                    xaxis_title='Attālums gar līniju (m)',
                    yaxis_title='Augstums (m)',
                    template='plotly_white',
                    height=500,
                    autosize=True,
                    margin=dict(l=50, r=50, t=50, b=50),
                    xaxis=dict(range=[0, line.length])
                )
                fig_html = fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    div_id=f'plot_{line_id}',
                    config={'responsive': True}
                )

                map_image_base64 = generate_map_image(line, points_gdf, ortho_dataset)
                if isinstance(line_attribute_id, (int, float)):
                    line_attribute_id_str = f"{line_attribute_id}"
                else:
                    line_attribute_id_str = str(line_attribute_id).replace(',', '').strip()

                html_content += f"<h2 id='griezums_{line_id}'>Griezums: {line_attribute_id_str}</h2>\n"
                html_content += f"<div style='text-align: center;'><img src='data:image/png;base64,{map_image_base64}' alt='Karte griezumam {line_attribute_id_str}'></div>\n"
                html_content += fig_html + "\n"
        else:
            # Ja ģeometrija nav Linestring, izlaiž
            continue

# ===================================================================
# 9. Pabeidzam HTML failu un saglabājam
# ===================================================================
html_content += javascript_code
html_content += "\n</body></html>"

# Saglabājam HTML failu
output_html_file = os.path.join(output_dir, 'skersgriezumi.html')
with open(output_html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML fails veiksmīgi saglabāts: {output_html_file}")
print("Process pabeigts.")
