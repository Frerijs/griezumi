import streamlit as st
import geopandas as gpd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import tempfile
import os

# Funkcija, lai ielādētu LandXML un iegūtu triangulāciju

def load_landxml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {'ns': 'http://www.landxml.org/schema/LandXML-1.0'}  # Atbalstām LandXML-1.0
    
    points = {}
    triangles = []
    
    # Meklējam punktus dažādās LandXML sadaļās
    point_elements = root.findall(".//ns:Pnts/ns:P", ns)  # Meklē <Pnts> struktūru
    if not point_elements:
        point_list_element = root.find(".//ns:DataPoints/ns:PntList3D", ns)  # Alternatīvs meklējums
        if point_list_element is not None:
            point_list = point_list_element.text.split()
            for i in range(0, len(point_list), 3):
                x, y, z = map(float, point_list[i:i+3])
                points[len(points) + 1] = (x, y, z)
    else:
        for p in point_elements:
            pid = p.attrib['id']
            x, y, z = map(float, p.text.split())
            points[pid] = (x, y, z)
    
    if not points:
        st.write("⚠️ Brīdinājums: Netika atrasti punkti LandXML failā.")
        return triangles, []
    
    st.write(f"✅ Atrasti {len(points)} punkti no LandXML!")
    return triangles, list(points.values())

# Funkcija, lai apmainītu X un Y koordinātes LandXML punktiem
def swap_xy(triangles):
    swapped_triangles = []
    for tri in triangles:
        swapped_triangles.append(((tri[0][1], tri[0][0], tri[0][2]),
                                  (tri[1][1], tri[1][0], tri[1][2]),
                                  (tri[2][1], tri[2][0], tri[2][2])))
    return swapped_triangles

# Funkcija, lai iegūtu Z vērtību jebkurā X, Y punktā
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
        x, y = point[:2]  # Ignorēt sākotnējo Z
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
        landxml_points = {}
        
        for landxml_file in landxml_files:
            file_path = os.path.join(temp_dir, landxml_file.name)
            with open(file_path, "wb") as f:
                f.write(landxml_file.getvalue())
            triangles, points = load_landxml(file_path)
            triangles = swap_xy(triangles)
            landxml_surfaces[landxml_file.name] = triangles
            landxml_points[landxml_file.name] = points
        
        st.write("Ielādētas SHP līnijas un LandXML virsmas.")
        st.write(f"Atrastās {len(gdf)} līnijas no SHP faila.")
        st.write(f"LandXML virsmas: {[key for key in landxml_surfaces.keys()]}")
        
        # Pārbaude: Vai SHP līnijas satur Z vērtības?
        if any(len(coord) == 3 for line in gdf.geometry for coord in line.coords):
            st.write("✅ SHP fails satur Z vērtības.")
        else:
            st.write("⚠️ Brīdinājums: SHP failam nav Z vērtību, interpolācija var nebūt precīza.")
        
        fig, ax = plt.subplots()
        
        for _, row in gdf.iterrows():
            line = row.geometry
            for surface_name, triangles in landxml_surfaces.items():
                profile = calculate_profile(triangles, line)
                if profile:
                    x_vals, y_vals, z_vals = zip(*profile)
                    ax.plot(x_vals, z_vals, label=f"{surface_name} - Līnija {row.name}")
                else:
                    st.write(f"Brīdinājums: Nav aprēķināts griezums {surface_name} - Līnija {row.name}")
        
        ax.set_xlabel("Attālums")
        ax.set_ylabel("Augstums")
        ax.set_title("Griezumu profili")
        ax.legend()
        
        st.pyplot(fig)
