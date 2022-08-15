import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    if is_image_file(filepath):
        img = Image.open(filepath)
        return img


def calculate_distance(angle):
    if angle == 0:
        return 1
    elif angle <= 3.6:
        return 2
    elif angle <= 7.2:
        return 4
    elif angle <= 10.79:
        return 6
    elif angle <= 14.4:
        return 8
    elif angle <= 18:
        return 10
    elif angle <= 21.6:
        return 12
    elif angle <= 25.2:
        return 14
    elif angle <= 28.8:
        return 16
    elif angle <= 32.4:
        return 19
    elif angle <= 36:
        return 21
    elif angle <= 39.6:
        return 23
    elif angle <= 43.2:
        return 25
    elif angle <= 46.8:
        return 27
    elif angle <= 50.4:
        return 29
    elif angle <= 54:
        return 31
    elif angle <= 57.6:
        return 33
    elif angle <= 61.2:
        return 35
    elif angle <= 64.8:
        return 38
    elif angle <= 68.4:
        return 40
    elif angle <= 72:
        return 42
    elif angle <= 75.6:
        return 44
    elif angle <= 79.2:
        return 46
    elif angle <= 82.8:
        return 48
    elif angle <= 86.4:
        return 50
    elif angle <= 90:
        return 52
    elif angle <= 93.6:
        return 55
    elif angle <= 97.2:
        return 57
    elif angle <= 100.8:
        return 59
    elif angle <= 104.4:
        return 61
    elif angle <= 108:
        return 63
    elif angle <= 111.6:
        return 65
    elif angle <= 115.2:
        return 67
    elif angle <= 118.8:
        return 69
    elif angle <= 122.4:
        return 72
    elif angle <= 126:
        return 74
    elif angle <= 129.6:
        return 76
    elif angle <= 133.2:
        return 78
    elif angle <= 136.8:
        return 80
    elif angle <= 140.4:
        return 82
    elif angle <= 144:
        return 84
    elif angle <= 147.6:
        return 86
    elif angle <= 151.2:
        return 88
    elif angle <= 154.8:
        return 91
    elif angle <= 158.4:
        return 93
    elif angle <= 162:
        return 95
    elif angle <= 165.6:
        return 97
    elif angle <= 169.2:
        return 99
    elif angle <= 172.8:
        return 101
    elif angle <= 176.4:
        return 103
    elif angle <= 180:
        return 105
    elif angle <= 183.6:
        return 108
    elif angle <= 187.2:
        return 110
    elif angle <= 190.8:
        return 112
    elif angle <= 194.4:
        return 114
    elif angle <= 198:
        return 116
    elif angle <= 201.6:
        return 118
    elif angle <= 205.2:
        return 120
    elif angle <= 208.8:
        return 122
    elif angle <= 212.4:
        return 125
    elif angle <= 216:
        return 127
    elif angle <= 219.6:
        return 129
    elif angle <= 223.2:
        return 131
    elif angle <= 226.8:
        return 133
    elif angle <= 230.4:
        return 135
    elif angle <= 234:
        return 137
    elif angle <= 237.6:
        return 139
    elif angle <= 241.2:
        return 141
    elif angle <= 244.8:
        return 144
    elif angle <= 248.4:
        return 146
    elif angle <= 252:
        return 148
    elif angle <= 255.6:
        return 150
    elif angle <= 259.2:
        return 152
    elif angle <= 262.8:
        return 154
    elif angle <= 266.4:
        return 156
    elif angle <= 270:
        return 158
    elif angle <= 273.6:
        return 161
    elif angle <= 277.2:
        return 163
    elif angle <= 280.8:
        return 165
    elif angle <= 284.4:
        return 167
    elif angle <= 288:
        return 169
    elif angle <= 291.6:
        return 171
    elif angle <= 295.2:
        return 173
    elif angle <= 298.8:
        return 175
    elif angle <= 302.4:
        return 178
    elif angle <= 306.0:
        return 180
    elif angle <= 309.6:
        return 182
    elif angle <= 313.2:
        return 184
    elif angle <= 316.8:
        return 186
    elif angle <= 320.4:
        return 188
    elif angle <= 324:
        return 190
    elif angle <= 327.6:
        return 192
    elif angle <= 331.2:
        return 194
    elif angle <= 334.8:
        return 197
    elif angle <= 338.4:
        return 199
    elif angle <= 342:
        return 201
    elif angle <= 345.6:
        return 203
    elif angle <= 349.2:
        return 205
    elif angle <= 352.8:
        return 207
    elif angle <= 356.4:
        return 209
    else:
        return 211


def cast_to_class(angle):
    if angle == 0:
        return 0
    elif angle <= 3.6:
        return 1
    elif angle <= 7.2:
        return 2
    elif angle <= 10.79:
        return 3
    elif angle <= 14.4:
        return 4
    elif angle <= 18:
        return 5
    elif angle <= 21.6:
        return 6
    elif angle <= 25.2:
        return 7
    elif angle <= 28.8:
        return 8
    elif angle <= 32.4:
        return 9
    elif angle <= 36:
        return 10
    elif angle <= 39.6:
        return 11
    elif angle <= 43.2:
        return 12
    elif angle <= 46.8:
        return 13
    elif angle <= 50.4:
        return 14
    elif angle <= 54:
        return 15
    elif angle <= 57.6:
        return 16
    elif angle <= 61.2:
        return 17
    elif angle <= 64.8:
        return 18
    elif angle <= 68.4:
        return 19
    elif angle <= 72:
        return 20
    elif angle <= 75.6:
        return 21
    elif angle <= 79.2:
        return 22
    elif angle <= 82.8:
        return 23
    elif angle <= 86.4:
        return 24
    elif angle <= 90:
        return 25
    elif angle <= 93.6:
        return 26
    elif angle <= 97.2:
        return 27
    elif angle <= 100.8:
        return 28
    elif angle <= 104.4:
        return 29
    elif angle <= 108:
        return 30
    elif angle <= 111.6:
        return 31
    elif angle <= 115.2:
        return 32
    elif angle <= 118.8:
        return 33
    elif angle <= 122.4:
        return 34
    elif angle <= 126:
        return 35
    elif angle <= 129.6:
        return 36
    elif angle <= 133.2:
        return 37
    elif angle <= 136.8:
        return 38
    elif angle <= 140.4:
        return 39
    elif angle <= 144:
        return 40
    elif angle <= 147.6:
        return 41
    elif angle <= 151.2:
        return 42
    elif angle <= 154.8:
        return 43
    elif angle <= 158.4:
        return 44
    elif angle <= 162:
        return 45
    elif angle <= 165.6:
        return 46
    elif angle <= 169.2:
        return 47
    elif angle <= 172.8:
        return 48
    elif angle <= 176.4:
        return 49
    elif angle <= 180:
        return 50
    elif angle <= 183.6:
        return 51
    elif angle <= 187.2:
        return 52
    elif angle <= 190.8:
        return 53
    elif angle <= 194.4:
        return 54
    elif angle <= 198:
        return 55
    elif angle <= 201.6:
        return 56
    elif angle <= 205.2:
        return 57
    elif angle <= 208.8:
        return 58
    elif angle <= 212.4:
        return 59
    elif angle <= 216:
        return 60
    elif angle <= 219.6:
        return 61
    elif angle <= 223.2:
        return 62
    elif angle <= 226.8:
        return 63
    elif angle <= 230.4:
        return 64
    elif angle <= 234:
        return 65
    elif angle <= 237.6:
        return 66
    elif angle <= 241.2:
        return 67
    elif angle <= 244.8:
        return 68
    elif angle <= 248.4:
        return 69
    elif angle <= 252:
        return 70
    elif angle <= 255.6:
        return 71
    elif angle <= 259.2:
        return 72
    elif angle <= 262.8:
        return 73
    elif angle <= 266.4:
        return 74
    elif angle <= 270:
        return 75
    elif angle <= 273.6:
        return 76
    elif angle <= 277.2:
        return 77
    elif angle <= 280.8:
        return 78
    elif angle <= 284.4:
        return 79
    elif angle <= 288:
        return 80
    elif angle <= 291.6:
        return 81
    elif angle <= 295.2:
        return 82
    elif angle <= 298.8:
        return 83
    elif angle <= 302.4:
        return 84
    elif angle <= 306.0:
        return 85
    elif angle <= 309.6:
        return 86
    elif angle <= 313.2:
        return 87
    elif angle <= 316.8:
        return 88
    elif angle <= 320.4:
        return 89
    elif angle <= 324:
        return 90
    elif angle <= 327.6:
        return 91
    elif angle <= 331.2:
        return 92
    elif angle <= 334.8:
        return 93
    elif angle <= 338.4:
        return 94
    elif angle <= 342:
        return 95
    elif angle <= 345.6:
        return 96
    elif angle <= 349.2:
        return 97
    elif angle <= 352.8:
        return 98
    elif angle <= 356.4:
        return 99
    else:
        return 100


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = image_dir

        self.image_filenames = []
        for directory, subdirectory, filenames in os.walk(image_dir):
            for folder in subdirectory:
                filename = 0
                while round(filename) < 360:
                    # for filename in range(0, 360):
                    filename = round(filename)
                    self.image_filenames.append("./data/" + folder + "/" + str(filename) + '.png')
                    filename += 3.6
        self.transform = transform

    def __getitem__(self, index):
        filename = self.image_filenames[index]

        input = load_img(filename)

        input = self.transform(input)

        target = np.array(cast_to_class(int(filename.split("/")[3].split(".")[0])))
        target = torch.from_numpy(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromTestFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(DatasetFromTestFolder, self).__init__()
        self.image_dir = image_dir

        self.image_filenames = []
        for directory, subdirectory, filenames in os.walk(image_dir):
            for folder in subdirectory:
                filename = 0
                while round(filename) < 3.6:
                    # for filename in range(0, 360):
                    filename = round(filename)
                    self.image_filenames.append("./test/" + folder + "/" + str(filename) + '.png')
                    filename += 3.6
        self.transform = transform

    def __getitem__(self, index):
        filename = self.image_filenames[index]

        input = load_img(filename)

        input = self.transform(input)

        target = np.array(calculate_distance(int(filename.split("/")[3].split(".")[0])))
        target = torch.from_numpy(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
