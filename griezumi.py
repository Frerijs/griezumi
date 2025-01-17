import streamlit as st
import geopandas as gpd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

# Funkcija, lai ielādētu LandXML un iegūtu triangulāciju
def load_landxml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = {'ns': 'http://www.landxml.org/schema/LandXML-1.2'}
    
    points = {}
    triangles = []
    
    for p in root.findall(".//ns:Point", ns):
        pid = p.attrib['id']
        x, y, z = map(float, p.text.split())
        points[pid] = (x, y, z)
    
    for tri in root.findall(".//ns:Triangle", ns):
        p1, p2, p3 = tri.text.split()
        triangles.append((points[p1], points[p2], points[p3]))
    
    return triangles

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

landxml_file = st.file_uploader("Augšupielādē LandXML", type=["xml"])
shp_file = st.file_uploader("Augšupielādē SHP", type=["shp"])

if landxml_file and shp_file:
    triangles = load_landxml(landxml_file)
    triangles = swap_xy(triangles)  # Apmainīt X un Y koordinātes
    gdf = gpd.read_file(shp_file)
    
    st.write("Ielādētas SHP līnijas un LandXML virsma.")
    
    fig, ax = plt.subplots()
    
    for _, row in gdf.iterrows():
        line = row.geometry
        profile = calculate_profile(triangles, line)
        
        if profile:
            x_vals, y_vals, z_vals = zip(*profile)
            ax.plot(x_vals, z_vals, label=f"Līnija {row.name}")
    
    ax.set_xlabel("Attālums")
    ax.set_ylabel("Augstums")
    ax.set_title("Griezumu profili")
    ax.legend()
    
    st.pyplot(fig)
