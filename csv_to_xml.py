# Trans csv to xml
from PIL import Image
import csv

def csv_to_xml(switch):
    if switch is True:
        for i in range(3):
            # mode : train / validation / test
            if i == 0:
                mode = 'train'
            elif i == 1:
                mode = 'validation'
            elif i == 2:
                mode = 'test'

            img_path = mode + '/images/'
            xml_path = mode + '/annots/'

            reader = list(csv.reader(open('detection_dataset/sub-' + mode + '-annotations-bbox.csv', 'r')))
            before_xml = str(reader[1][0])
            init_row = True
            num_row = 0

            try:
                for row in reader:
                    if num_row == 0:
                        before_xml = row[0]
                        num_row += 1
                        continue
                    else:
                        if row[0] != before_xml or init_row == True:
                            f = open('refri_dataset/' + xml_path + row[0], 'w')
                            f.write('<annotation>\n')
                            f.write('\t<folder>' + row[5] + '</folder>\n')
                            f.write('\t<filename>' + row[0][:-4] + '.jpg</filename>\n')
                            f.write('\t<path>' + 'refri_dataset/' + img_path + row[0][:-4] + '.jpg</path>\n')
                            f.write('\t<source>\n')
                            f.write('\t\t<database>Unknown</database>\n')
                            f.write('\t</source>\n')
                            width, height = Image.open('detection_dataset/' + mode + '/' + row[0][:-4] + '.jpg').size
                            f.write('\t<size>\n')
                            f.write('\t\t<width>' + str(width) + '</width>\n')
                            f.write('\t\t<height>'+ str(height) + '</height>\n')
                            f.write('\t\t<depth>3</depth>\n')
                            f.write('\t</size>\n')
                            f.write('\t<segmented>0</segmented>\n')
                            init_row = False

                        width, height = Image.open('detection_dataset/' + mode + '/' + row[0][:-4] + '.jpg').size
                        f.write('\t<object>\n')
                        f.write('\t\t<name>' + row[5] + '</name>\n')
                        f.write('\t\t<pose>Unspecified</pose>\n')
                        f.write('\t\t<truncated>0</truncated>\n')
                        f.write('\t\t<difficult>0</difficult>\n')
                        f.write('\t\t<bndbox>\n')
                        f.write('\t\t\t<xmin>' + str(int(width*float(row[1]))) + '</xmin>\n')
                        f.write('\t\t\t<ymin>' + str(int(height*float(row[3]))) + '</ymin>\n')
                        f.write('\t\t\t<xmax>' + str(int(width*float(row[2]))) + '</xmax>\n')
                        f.write('\t\t\t<ymax>' + str(int(height*float(row[4]))) + '</ymax>\n')
                        f.write('\t\t</bndbox>\n')
                        f.write('\t</object>\n')

                        # is same name next xml
                        if reader[num_row+1][0] != row[0] and num_row > 0:
                            f.write('</annotation>\n')

                    before_xml = row[0]
                    num_row += 1

            except IndexError as e:
                # close last end tag
                f.write('</annotation>')
