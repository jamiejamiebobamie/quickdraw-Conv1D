import os
import json
import psutil
from PIL import Image, ImageDraw

# {"word": "bat", "countrycode": "US", "timestamp": "2017-03-29 16:40:07.89666 UTC", "recognized": true, "key_id": "6589479475216384", "drawing": [[[32, 10, 0, 6, 31, 36, 52, 68, 99, 104, 106, 102, 101, 124, 144, 170, 193, 211, 168, 167, 161, 142, 139, 130, 125, 117, 118, 98, 83, 51, 48, 53, 49, 44, 30, 27], [80, 52, 19, 22, 45, 46, 38, 41, 3, 1, 12, 28, 72, 78, 89, 112, 139, 178, 152, 195, 195, 163, 175, 184, 183, 174, 242, 227, 207, 210, 216, 250, 255, 255, 230, 213]]]}

def getRemainingCount(data):
    print(len([item for item in data if item != False]))
    return len([item for item in data if item != False])


cwd = os.getcwd()
file = os.path.join(cwd, "Dataset\one_line_drawings_pruned.ndjson")
data = []

# create an array of drawing entries from full dataset
with open(file, "r") as f:
    for line in f:
        data.append(line)

# # --------------
# update = []
# # iteration ends when only 100(?) entries are in the array
# while getRemainingCount(data) > 100:
#     print(getRemainingCount(data))
#     i = 0
#     while i < len(data):
#         d = json.loads(data[i])
#         stroke = d["drawing"][0]
#         label = d["word"]
#         [strokex, strokey] = stroke
#         im = Image.new (mode="RGB", size=(255, 255), color=(255,255,255))
#         draw = ImageDraw.Draw(im)
#         line = [ (strokex[k], strokey[k]) for k in range(len(strokex))]
#         for _ in range(len(strokex)):
#             draw.line(line, fill=0, width= 1)
#         im.show()
#         # create user interface that accepts input to recursively iterate through array
#         #   input commands: "toss", "keep", and "skip to next word" (tosses all entries of that word that have not been reviewed)
#         resp = ""
#         print(label)
#         while not (resp == 't' or resp == 'k' or resp == 's'):
#             resp = input("'T' to toss \n'K' to keep \n'S' to skip to next word and toss out all unseen entries of this word ")
#             if resp.lower() == 't':
#                 continue
#             elif resp.lower() == 'k':
#                 update.append(d)
#             elif resp.lower() == 's':
#                 j = i
#                 while j < len(data):
#                     skip = json.loads(data[j])
#                     l = skip["word"]
#                     if l != label:
#                         i = j
#                         break
#                     else:
#                         j+=1
#             else:
#                 resp = input("'T' to toss \n'K' to keep \n'S' to skip to next word and toss out all unseen entries of this word ")
#         i+=1
#         print(i, len(data), update)
#     data = update
#     update = []
#     print(getRemainingCount(data))
#     break

# print(getRemainingCount(data))
# # ----




for i in range(len(data)):
    d = json.loads(data[i])
    stroke = d["drawing"][0]
    label = d["word"]
    key_id = d["key_id"]
    [strokex, strokey] = stroke
    im = Image.new (mode="RGB", size=(255, 255), color=(255,255,255))
    draw = ImageDraw.Draw(im)
    line = [ (strokex[k], strokey[k]) for k in range(len(strokex))]
    for _ in range(len(strokex)):
        draw.line(line, fill=0, width= 1)
    im.save(os.path.join(cwd, "Images/" + label + "_" + key_id + ".png"))