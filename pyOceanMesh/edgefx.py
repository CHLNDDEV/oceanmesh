from .geodata import Geodata

__all__ = ["Edgefx"]


class Edgefx:
    def __init__(self, gdat, hmin, **kwargs):
        """Construct a mesh sizing function using geospatial
        datata from an instance gdat of :class:`Geodata`"""

        self.gdat = gdat
        self.fields = {"hmin": None, "dis": None}
        self.hh_m = None

        # fields defines "reasonable" default values
        for name, field in kwargs.items():
            if name in self.fields.keys():
                self.fields[name].assign(field)

    @property
    def gdat(self):
        return self.__gdat

    @gdat.setter
    def gdat(self, obj):
        if not isinstance(obj, Geodata):
            raise ValueError("Passed object is not a Geodata object")
        self.__gdat = obj

    @property
    def hmin(self):
        return self.__hmin

    @hmin.setter
    def hmin(self, value):
        if value < 0:
            raise ValueError("Hmin must be > 0.0")
        self.__hmin = value

    def build(self):
        """Build each sizing function"""
        for item in self.fields():
            if item[1] is not None:
                print("Building " + item[0] + " sizing function")
