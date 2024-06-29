#!/usr/bin/env python
import sys
import vtk
import os
#------------------------------------------------------------------------------
# Script Entry Point
#------------------------------------------------------------------------------
if __name__ == "__main__":

    print "vtkGraph: Building a graph using Unstructured Grid, dump it in a vtk file, vertex.vtu, to be visualized using ParaView"
    
    # Create a user specified number of points 
    pointSource = vtk.vtkPointSource()
    pointSource.Update()

    # Create an integer array to store vertex id data and link it with its degree value as a scalar.
    degree  = vtk.vtkIntArray()
    degree.SetNumberOfComponents(1) # set the dimentions (n) of the component i.e. only degree of a vertex is used as a component
    degree.SetName("degree")
    degree.SetNumberOfTuples(7)     # set the number of tuples (a component group) i.e 7 vertices
    degree.SetValue(0,2)            # vertex id 0 links to the degree of it i.e. 2 
    degree.SetValue(1,1)
    degree.SetValue(2,3)
    degree.SetValue(3,3)
    degree.SetValue(4,4)
    degree.SetValue(5,2)
    degree.SetValue(6,1)
 
    pointSource.GetOutput().GetPointData().AddArray(degree)
    
    # Assaign co-ordinates for vertices. vtkPoints represents 3D points
    Points = vtk.vtkPoints()
    
    Points.InsertNextPoint(0,1,0)
    Points.InsertNextPoint(0,0,0)
    Points.InsertNextPoint(1,1,0)
    Points.InsertNextPoint(1,0,0)
    Points.InsertNextPoint(2,1,0)
    Points.InsertNextPoint(2,0,0)
    Points.InsertNextPoint(3,0,0)

    # Establish the specified edges using CellArray. It represents cell connectivity
    line = vtk.vtkCellArray()
    line.Allocate(8)          # Allocate memory for 8 objects (Number of edges)
    line.InsertNextCell(2)    # Insert two cell objects
    line.InsertCellPoint(0)   # Add vertex with ID 0 in the cell
    line.InsertCellPoint(1)   # Add vertex with ID 1 in the same cell i.e connectivity between ID 0 and ID 1
    line.InsertNextCell(2)
    line.InsertCellPoint(0)
    line.InsertCellPoint(2)
    line.InsertNextCell(2)
    line.InsertCellPoint(2)
    line.InsertCellPoint(3)
    line.InsertNextCell(2)
    line.InsertCellPoint(2)
    line.InsertCellPoint(4)
    line.InsertNextCell(2)
    line.InsertCellPoint(3)
    line.InsertCellPoint(4)
    line.InsertNextCell(2)
    line.InsertCellPoint(3)
    line.InsertCellPoint(5)
    line.InsertNextCell(2)
    line.InsertCellPoint(4)
    line.InsertCellPoint(5)
    line.InsertNextCell(2)
    line.InsertCellPoint(4)
    line.InsertCellPoint(6)
           
    # Add the vertices and edges to unstructured Grid
    G = vtk.vtkUnstructuredGrid()
    G.GetPointData().SetScalars(degree)    # set the scalar data i.e. degree
    G.SetPoints(Points)                    # Insert point object (i.e coordinates of vertices)
    G.SetCells(vtk.VTK_LINE, line)         # vtkLine is a concrete implementation of vtkCell to represent a 1D line i.e edge
    
    # Dump the graph in VTK unstructured format (.vtu)
    gw = vtk.vtkXMLUnstructuredGridWriter() 
    gw.SetFileName("vertex.vtu")
    gw.SetInputData(G)
    gw.Write()
    print '---> ',
    
    print "Feed the vertex.vtu file in ParaView."
