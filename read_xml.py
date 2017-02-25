import xml.etree.ElementTree as ET

tree = ET.parse('brain.svg')
root = tree.getroot()

svg = list(root.findall('*'))[-1]

print(list(svg.findall('*')))
