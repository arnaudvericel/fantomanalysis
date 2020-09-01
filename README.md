# fantomanalysis

fantomanalysis is a python module that provides tools to analyze phantom simulations when dustgrowth is ON.

fantomanalysis offers the following functionalities:

- Read a dump file and returns a pandas DataFrame object filled with the particles properties
- Bin thus properties along the radial or vertical directions of space as well as along the grain size / Stokes number dimensions
- Flag dust particles that fulfil certain conditions (radii, altitude, size)
- Follow selected particles through a set of dumps and returns the evolution of their properties as a DataFrame

The idea behind fantomanalysis is to provide a user friendly analyzing tool that takes advantage of the other python packages and flexibility

---

## Installation:

```
git clone https://github.com/arnaudvericel/fantomanalysis.git
cd fantomanalysis
python3 setup.py install
```

If you don't have the `sudo` rights, use `python3 setup.py install --user`.

To install in developer mode: (i.e. using symlinks to point directly
at this directory, so that code changes here are immediately available
without needing to repeat the above step):

```
 python3 setup.py develop
```

Have fun!
