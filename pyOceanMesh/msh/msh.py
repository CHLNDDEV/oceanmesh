class Msh:
    """
    MSH: Mesh class
    Contains, handles, and builds properties of a mesh such as vertices,
    an element table, bathymetry and different boundary types
    """

    def __init__(
        self, title="pyOceanMesh", points=None, triangles=None, bathymetry=None
    ):

        self.title = title
        self.p = points
        self.t = triangles
        self.b = bathymetry

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, value):
        self.__p = value
