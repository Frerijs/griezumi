import streamlit as st
import geopandas as gpd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import tempfile
import os
from scipy.spatial import Delaunay

# Funkcija, lai ielādētu LandXML un izveidotu triangulāciju

def load_landxml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {'ns': 'http://www.landxml.org/schema/LandXML-1.0'}
    
    points = []
    
    # Meklējam punktus dažādās LandXML sadaļās
    point_elements = root.findall(".//ns:Pnts/ns:P", ns)
    if not point_elements:
        point_list_element = root.find(".//ns:DataPoints/ns:PntList3D", ns)
        if point_list_element is not None:
            point_list = point_list_element.text.split()
            for i in range(0, len(point_list), 3):
                x, y, z = map(float, point_list[i:i+3])
                points.append((x, y, z))
    else:
        for p in point_elements:
            x, y, z = map(float, p.text.split())
            points.append((x, y, z))
    
    if not points:
        st.write("⚠️ Brīdinājums: Netika atrasti punkti LandXML failā.")
        return None, []
    
    # Izveidojam triangulāciju no punktiem
    xy_points = np.array([(p[0], p[1]) for p in points])
    delaunay = Delaunay(xy_points)
    triangles = [tuple(points[i] for i in tri) for tri in delaunay.simplices]
    
    st.write(f"✅ Atrasti {len(points)} punkti no LandXML, izveidoti {len(triangles)} trijstūri!")
    return triangles, points

# Funkcija, lai interpolētu Z vērtības
def interpolate_z(triangles, x, y):
    for tri in triangles:
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = tri
        
        A = np.array([
            [x1, y1, 1],
            [x2, y2, 1],
            [x3, y3, 1]
        ])
        b = np.array([z1, z2, z3])
        
        try:
            coeffs = np.linalg.solve(A, b)
            z = coeffs[0] * x + coeffs[1] * y + coeffs[2]
            return z
        except np.linalg.LinAlgError:
            continue
    return None

# Funkcija, lai aprēķinātu griezumu līnijas punktiem
def calculate_profile(triangles, line):
    profile = []
    for point in line.coords:
        x, y = point[:2]
        z = interpolate_z(triangles, x, y)
        if z is not None:
            profile.append((x, y, z))
    return profile

# Streamlit UI
st.title("LandXML & SHP griezumu vizualizācija")

landxml_files = st.file_uploader("Augšupielādē LandXML failus", type=["xml"], accept_multiple_files=True)
shp_files = st.file_uploader("Augšupielādē SHP un papildfailus", type=["shp", "shx", "dbf", "prj", "cpg"], accept_multiple_files=True)

if landxml_files and shp_files:
    temp_dir = tempfile.mkdtemp()
    shp_main_file = None
    
    for file in shp_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        if file.name.endswith(".shp"):
            shp_main_file = file_path
    
    if shp_main_file:
        gdf = gpd.read_file(shp_main_file)
        landxml_surfaces = {}
        
        for landxml_file in landxml_files:
            file_path = os.path.join(temp_dir, landxml_file.name)
            with open(file_path, "wb") as f:
                f.write(landxml_file.getvalue())
            triangles, points = load_landxml(file_path)
            if triangles:
                landxml_surfaces[landxml_file.name] = triangles
        
        st.write("Ielādētas SHP līnijas un LandXML virsmas.")
        st.write(f"Atrastās {len(gdf)} līnijas no SHP faila.")
        st.write(f"LandXML virsmas: {[key for key in landxml_surfaces.keys()]}")
        
        for _, row in gdf.iterrows():
            line = row.geometry
            fig, ax = plt.subplots()
            
            for surface_name, triangles in landxml_surfaces.items():
                profile = calculate_profile(triangles, line)
                if profile:
                    x_vals, y_vals, z_vals = zip(*profile)
                    ax.plot(x_vals, z_vals, label=f"{surface_name} - Līnija {row.name}")
                else:
                    st.write(f"⚠️ Brīdinājums: Nav aprēķināts griezums {surface_name} - Līnija {row.name}")
            
            ax.set_xlabel("Attālums")
            ax.set_ylabel("Augstums")
            ax.set_title(f"Griezuma profils - Līnija {row.name}")
            ax.legend()
            
            st.pyplot(fig)
