import matplotlib

# IFRace saves a plot; never open a GUI backend under pytest.
matplotlib.use("Agg")
