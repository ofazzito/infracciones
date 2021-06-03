from alpr.alpr import ALPR
import cv2
import yaml

im = cv2.imread('data/imags/prueba.jpg')
with open('config.yaml', 'r') as stream:
    cfg = yaml.safe_load(stream)
alpr = ALPR(cfg['modelo'], cfg['db'])
predicciones = alpr.predict(im)
print("Prueba: ", predicciones)