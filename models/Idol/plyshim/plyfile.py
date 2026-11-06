# Minimal write-only stand-in for the GPL-3.0 `plyfile` package, covering idol.gaussian.save_ply only:
# one vertex element of float32 fields, written as binary little-endian PLY. Own code, Apache-2.0.
import numpy as np

_TYPES = {'f4': 'float', 'f8': 'double', 'i1': 'char', 'u1': 'uchar', 'i2': 'short', 'u2': 'ushort',
          'i4': 'int', 'u4': 'uint'}


class PlyElement:
    def __init__(self, name, data):
        self.name, self.data = name, data

    @staticmethod
    def describe(data, name):
        return PlyElement(name, data)


class PlyData:
    def __init__(self, elements):
        self.elements = list(elements)

    def write(self, path):
        lines = ['ply', 'format binary_little_endian 1.0']
        for e in self.elements:
            lines.append(f'element {e.name} {len(e.data)}')
            for field in e.data.dtype.names:
                lines.append(f'property {_TYPES[e.data.dtype[field].str[1:]]} {field}')
        lines.append('end_header')
        with open(path, 'wb') as f:
            f.write(('\n'.join(lines) + '\n').encode('ascii'))
            for e in self.elements:
                f.write(e.data.astype(e.data.dtype.newbyteorder('<')).tobytes())
