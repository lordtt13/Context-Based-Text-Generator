# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 13:04:20 2019

@author: tanma
"""

import xlrd

wb = xlrd.open_workbook('Principal Stresses.xlsx')
sheet = wb.sheet_by_index(0) 

for i in range(sheet.ncols): 
    print(sheet.row_values(i))  