#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class Street:

    def __init__(self,id,mask):
        self.id = id
        self.mask=mask
        self.intersections=[]
        self.interseCentroids=[]
	self.interseCentroidsTransf=[]
        self.direction=""
	self.wayDirection=""

class meta:
    def __init__(self,classes,names):
	self.classes = classes
	self.names = names
