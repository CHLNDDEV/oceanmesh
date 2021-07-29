 --import oceanmesh as om

   fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"

   bbox, min_edge_length = (-75.000, -70.001, 40.0001, 41.9000), 1e3

   shore = om.Shoreline(fname, bbox, min_edge_length)

   edge_length = om.distance_sizing_function(shore, rate=0.35)

   test_edge_length = om.enforce_mesh_gradation(edge_length, gradation=0.20)
   test_edge_length.plot(show=False, file_name="test_edge_length.png")
