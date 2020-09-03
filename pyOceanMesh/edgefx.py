import numpy

__all__ = ["Grid", "DistanceSizingFunction"]


class Grid:
    def __init__(self, bbox, grid_spacing):
        """Class to create abstract a structured grid"""

        self.x0y0 = (
            min(bbox[0:2]),
            min(bbox[3:]),
        )  # bottom left corner coordinates
        self.grid_spacing = grid_spacing
        ceil, abs = numpy.ceil, numpy.abs
        self.nx = ceil(abs(self.x0y0(0) - bbox(1)) / self.grid_spacing)
        self.ny = ceil(abs(self.x0y0(1) - bbox(3)) / self.grid_spacing)

    @property
    def grid_spacing(self):
        return self.__grid_spacing

    @grid_spacing.setter
    def grid_spacing(self, value):
        if value < 0:
            raise ValueError("Grid spacing must be > 0.0")
        self.__grid_spacing = value

    @property
    def bbox(self):
        return self.__bbox

    @bbox.setter
    def bbox(self, value):
        if value is None:
            self.__bbox = value
        else:
            if len(value) < 4:
                raise ValueError("bbox has wrong number of values.")
            if value[1] < value[0]:
                raise ValueError("bbox has wrong values.")
            if value[3] < value[2]:
                raise ValueError("bbox has wrong values.")
            self.__bbox = value

    def create_grid(self):
        x = self.x0y0[0] + numpy.arange(0, self.nx - 1) * self.grid_spacing
        y = self.x0y0[1] + numpy.arange(0, self.ny - 1) * self.grid_spacing
        return numpy.meshgrid(x, y, sparse=False, indexing="ij")

    # TODO overload plus for grid objects


class DistanceSizingFunction(Grid):
    def __init__(self, Shoreline, dis=0.15):
        """Create a sizing functiont that varies linearly at a rate `dis`
        from the union of the shoreline features"""
        super().__init__(bbox=Shoreline.bbox, grid_spacing=Shoreline.h0)
        # TODO calculate distance from
