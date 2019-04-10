import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text, # filename
                     int(root.find('size')[0].text), # width
                     int(root.find('size')[1].text), # height
                     member[0].text, # class
                     int(member[5][0].text), # xmin
                     int(member[5][1].text), # ymin
                     int(member[5][2].text), # xmax
                     int(member[5][3].text)  # ymax
                     )
            print(value[0])
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for directory in ['train', 'val']:
        image_path = os.path.join(os.getcwd(), f'annotations_xml_{directory}')
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(f'annotations_xml_{directory}/{directory}.csv', index=None)
        print('Successfully converted xml to csv.')

main()