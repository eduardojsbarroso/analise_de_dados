#!/usr/bin/env python
import numpy as np
import scipy as sc
from scipy.special import erf
import random as rd
import matplotlib.pyplot as plt
import math

class LinearInterp:
    def __init__(self, x_data, y_data, interval):
        self.x_data = x_data
        self.y_data = y_data
        self.cumulative_data = self.cumulative_full_compute(interval[0], interval[1])
        self.interval = interval
   

    def do_interp(self, x):
        for a in range(0, len(self.y_data)):
            if x >= self.x_data[a] and x <= self.x_data[a+1]:
                dy = self.y_data[a+1] - self.y_data[a]
                dx = self.x_data[a+1] - self.x_data[a]
                coef_b = self.y_data [a] - (self.x_data[a] * dy) / dx  
                coef_a = dy / dx
                new_point = coef_b + x * coef_a
                return [new_point, (coef_a, coef_b),(self.x_data[a], self.x_data[a+1])]
    
    def do_cumulative(self, a, b):
        cumulative = 0
        for x in self.x_data:
            if x>=a and x<=b:
                result = self.do_interp(x)
                cumulative += result[2][1] ** 2 * result[1][0] /2 + result[2][1]*result[1][1] - result[2][0] ** 2 * result[1][0] /2 - result[2][0]*result[1][1]  
        return cumulative

    
    def cumulative_full_compute(self, interval1, interval2):
        temporary_list =[]
        for knot in range(0,len(self.x_data)):
            temporary_list.append(self.do_cumulative(interval1, self.x_data[knot]))
        return temporary_list


#This function generates the points for a given cumulative. I have to limit the possible cumulative because if I did't have that point to interpolate, I can't sample in this interval.
    def generate_random_point(self):
        c = rd.uniform(min(self.cumulative_data), max(self.cumulative_data))
        #c = rd.uniform(0,1)
        for knot in range(0,len(self.cumulative_data)):
            cumulative_i = self.cumulative_data[knot]
            if cumulative_i >= c:
                cumulative_j = self.cumulative_data[knot+1]
                dy = self.x_data[knot+1] - self.x_data[knot]
                dx = cumulative_j - cumulative_i
                coef_b = self.x_data[knot] - (cumulative_i * dy) / dx  
                coef_a = dy / dx
                new_point = coef_b + c * coef_a
                return [new_point, (coef_a, coef_b),(self.x_data[knot], self.x_data[knot+1]), c]
                

class PDF:
    def __init__(self):
        self.count = 1
        
        
    def gaussian_oned_pdf(self, x, mean, sigma):
        return np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma, 2.))) * (1/ (np.sqrt(2 * np.pi)*sigma))

    def gaussian_oned_cdf(self,b, mean, sigma):
        arg = (b - mean)/(np.sqrt(2) * sigma)
        return 1/2 * (1 + erf(arg))
            
    def coin_toss(self, points, probability):
        i =0
        j=0
        for a in range(0, points):
            p = rd.uniform(0,1)
            if p <= probability:
                i+=1
            else:
                j+=1
        return [i,j]
            
    def gaussian_twod(self, x, y, meanx, meany, sigmax, sigmay):
        gaus_x = np.exp(-np.power(x - meanx, 2.) / (2 * np.power(sigmax, 2.))) * (1/ (np.sqrt(2 * np.pi)*sigmax))
        gaus_y = np.exp(-np.power(y - meany, 2.) / (2 * np.power(sigmay, 2.))) * (1/ (np.sqrt(2 * np.pi)*sigmay))
        return gaus_x * gaus_y
        
        
        
class Util:
    def __init__(self):
        self.true = 0
    
    
    def mean_function(self,points):
        den = len(points)
        sum_p = [a/den for a in points]
        return np.sum(sum_p)

    def var_function(self, points, mean):
        den = len(points)
        var_p = [(a - mean)**2 / den for a in points]
        return np.sum(var_p)
        
    def skew(x, sigma, mean):
        list_sum = []
        for point in x:
            list_sum.append((point - mean)**3)
        return (np.sum(list_sum) / (len(x) * sigma**3))

    def curtoise(x, sigma, mean):
        list_sum = []
        for point in x:
            list_sum.append((point - mean)**4)
        return ((np.sum(list_sum) / (len(x) * sigma**4)) - 3)
        
        
    def covariance_func(self,x,y, x_mean, y_mean):
        sum_list = []
        for a in range(0, len(x)):
            sum_list.append((x[a]-x_mean)*(y[a]-y_mean) / (len(x) - 1))
        return np.sum(sum_list)

    def correlation_func(self,cov, sigmax, sigmay):
        return cov/(sigmax * sigmay)