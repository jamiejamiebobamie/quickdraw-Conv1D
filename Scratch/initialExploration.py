import glob
import os
import json
from PIL import Image, ImageDraw

# cwd = os.getcwd()
# relative_path = os.path.join(cwd, "Datasets/*")
# files = glob.glob(relative_path)
# data = []

# for i,file in enumerate(files):
#     with open(file, "r") as f:
#         for line in f:
#             data.append(line)
#     print("reading file " + str(i) + " of " + str(len(files)))


# [                                                                drawing
#     [                                                            stroke
#     [0,33,132,164,165,155,85,37,23,28,190,157,89,61,65,125,201], x
#      [17,9,0,0,5,17,69,118,137,140,139,173,219,253,255,241,240]  y
#      ]
# ]

# min_x = float("inf")
# min_y = float("inf")
# max_x = float("-inf")
# max_y = float("-inf")

# for d in data:
#     d = json.loads(d)
#     drawing = d["drawing"]
#     for stroke in drawing:
#         [x_ps, y_ps] = stroke
#         # print({ "x_ps": "".join([str(x) for x in x_ps]), "y_ps": "".join([str(y) for y in y_ps])})
#         for x in x_ps:
#             min_x = min(min_x, x)
#             max_x = max(max_x, x)
#         for y in y_ps:
#             # print(y)
#             min_y = min(min_y, y)
#             max_y = max(max_y, y)
#     #     break
#     # break

# print(min_x, max_x, min_y, max_y) # 0 255 0 255



# n_strokes = []

# acc = 0

# for d in data:
#     d = json.loads(d)
#     drawing = d["drawing"]
#     # for stroke in drawing:
#     n_strokes.append(len(drawing))

# # mean_num_strokes = [acc + n for n in n_strokes][len(n_strokes)-1] // len(n_strokes)

# sorted_arr = sorted(n_strokes)
# print("max and min number of strokes:")
# print(sorted_arr[0],sorted_arr[len(sorted_arr) - 1])
# middle_i = len(n_strokes) // 2
# median_num_strokes = sorted_arr[middle_i]
# print("median_num_strokes:")
# print(median_num_strokes)

# sum = functools.reduce(lambda a,b: a+b, n_strokes)
# mean_num_strokes = sum // len(n_strokes)
# print("mean_num_strokes:")
# print(mean_num_strokes)

# n_1_stroke = {}

# for d in data:
#     d = json.loads(d)
#     drawing = d["drawing"]
#     category = d["word"]
#     isR = d["recognized"]
#     if len(drawing) == 1 and isR:
#         if category not in n_1_stroke:
#             n_1_stroke[category] = 1
#         else:
#             n_1_stroke[category] += 1

# acc = 0

# for k in n_1_stroke:
#     print(k)
#     print(n_1_stroke[k])
#     acc += n_1_stroke[k]

# print("number of drawings of one stroke:")
# print(acc)
# print("out of")
# print(len(data))
# print("drawings")


# create a new dataset of one line drawings
# with open(os.path.join(cwd, "one_line_drawings.ndjson"), "w") as f:
#     for d in data:
#         d = json.loads(d)
#         drawing = d["drawing"]
#         category = d["word"]
#         isR = d["recognized"]
#         if len(drawing) == 1 and isR:
#             json.dump(d, f)
#             f.write("\n")
#             print(category)


# n_1_stroke = {}

# for d in data:
#     d = json.loads(d)
#     drawing = d["drawing"]
#     category = d["word"]
#     isR = d["recognized"]
#     if len(drawing) == 1 and isR:
#         if category not in n_1_stroke:
#             n_1_stroke[category] = drawing

# for k in n_1_stroke:
#     stroke = n_1_stroke[k][0]
#     [strokex, strokey] = stroke
#     im = Image.new (mode="RGB", size=(255, 255), color=(255,255,255))
#     draw = ImageDraw.Draw(im)
#     line = [ (strokex[i], strokey[i]) for i in range(len(strokex))]
#     for i in range(len(strokex)):
#         draw.line(line, fill=0, width= 1)
#     im.show()



im = Image.new (mode="RGB", size=(255, 255), color=(255,255,255))
draw = ImageDraw.Draw(im)
strokex = [0,33,132,164,165,155,85,37,23,28,190,157,89,61,65,125,201]
strokey = [17,9,0,0,5,17,69,118,137,140,139,173,219,253,255,241,240]

line = [ (strokex[i], strokey[i]) for i in range(len(strokex))]

# for i in range(len(strokex)):
draw.line(line, fill=0, width= 1)
im.show()


# anvil
# 53396

# apple
# 17156

# axe
# 13435

# banana
# 115951

# bandage
# 1315

# bat
# 16958

# bear
# 5286

# bird
# 7439

# boomerang
# 104111

# brain
# 3749

# bread
# 48168

# broccoli
# 20232

# broom
# 19984

# bush
# 47420

# butterfly
# 3278

# camouflage
# 13474

# castle
# 21585

# diamond
# 35567

# feather
# 10904

# fence
# 6251

# finger
# 58177

# fish
# 27134

# flower
# 8117

# flying saucer
# 8553

# foot
# 113400

# frog
# 4311

# grass
# 35391

# hammer
# 18037

# hand
# 193229

# harp
# 1733

# hedgehog
# 1138

# hexagon
# 93621

# horse
# 20053

# hot air balloon
# 1970

# hourglass
# 41694

# house
# 14468

# hurricane
# 82460

# key
# 28834

# knife
# 26442

# leaf
# 4196

# leg
# 60350

# light bulb
# 20432

# lighter
# 1931

# lightning
# 84839

# line
# 126595

# marker
# 52408

# mermaid
# 1156

# moon
# 72923

# mosquito
# 1057

# mountain
# 54483

# mouse
# 4434

# mouth
# 16866

# mushroom
# 30951

# nose
# 97881

# ocean
# 51738

# octagon
# 89732

# octopus
# 48795

# pillow
# 58518

# pond
# 30479

# pool
# 10558

# rabbit
# 1483

# rainbow
# 33101

# rifle
# 55899

# river
# 8381

# saw
# 33493

# scissors
# 4882

# scorpion
# 4614

# shark
# 22422

# sheep
# 2274

# skyscraper
# 45144

# snail
# 11664

# snake
# 41830

# snowflake
# 5995

# snowman
# 4204

# square
# 95632

# squiggle
# 85194

# stairs
# 102823

# star
# 105428

# streetlight
# 27140

# sun
# 1451

# sword
# 8483

# tiger
# 1303

# tornado
# 96629

# tree
# 13919

# triangle
# 90045

# whale
# 5788

# yoga
# 2625

# zigzag
# 105307