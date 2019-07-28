import json
import logging
import sys
import os
import numpy as np
from imageio import imread, imsave
from skimage.draw import polygon_perimeter as draw_perimeter, line as draw_line
import glob

from features import farthest_coords, axial_ratio, compactness
from identify import make_poly, monolayers


def get_analysis(original, logger=logging.getLogger("draw")):
    analysis = monolayers(original, logger)
    base, monolayer_group = analysis["original"], analysis["monolayers"]
    data = {}
    for group in set(monolayer_group[monolayer_group > 0]):
        p0, p1, p2, p3 = farthest_coords(monolayer_group, group)
        rr, cc = draw_line(*p0.astype('int'), *p1.astype('int'))
        rr2, cc2 = draw_line(*p2.astype('int'), *p3.astype('int'))
        polygon = make_poly(monolayer_group, group)
        rb, cb = draw_perimeter(polygon[:, 0], polygon[:, 1], shape=original.shape, clip=True)
        base[rr, cc] = [0, 0, 0]
        base[rr2, cc2] = [0, 0, 0]
        base[rb, cb] = [0, 0, 0]
        data[str(group)] = {"axial_ratio": axial_ratio(monolayer_group, group),
                            "compactness": compactness(monolayer_group, group)}
    return (base * 255).astype(np.uint8), data


logger = logging.getLogger('draw')
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

if not os.path.exists("./annotated"):
    os.mkdir("./annotated")
for pattern in sys.argv[1:]:
    for f in glob.glob(pattern):
        image = imread(f)[:, :, :3] / 255
        f = f[f.rfind("/")+1:]
        separator = f.rfind('.')
        handler.setFormatter(logging.Formatter('[' + f + '] %(message)s'))
        annotated, features = get_analysis(image, logger)
        imsave("./annotated/{}".format(f), annotated)
        with open('./annotated/{}-feature.json'.format(f[:separator]), 'w') as out:
            json.dump(features, out)
