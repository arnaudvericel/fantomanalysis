# fantomanalysis
Python analysis tool for dust growth simulations in phantom

This python module provides pandas commands to analyze phantom ascii dumpfiles.

It can:
-> Read a dumpfile and give you a DataFrame containing all the data
-> Flag particles depending on a few of their parameters
-> Follow particles from dumpfiles to dumpfiles and fill a DataFrame accordingly
-> Create multiple radial, vertical, size and Stokes profiles of the data both for gas and dust
