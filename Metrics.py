#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:00:28 2023

@author: solar
"""
import numpy as np



def mean_bias_error(true, pred):
    mbe_loss = np.sum(pred - true)/true.size
    return mbe_loss


def mean_absolute_error(true, pred):
    mbe_loss = np.sum(abs(pred - true))/true.size
    return mbe_loss




def r_mean_bias_error(true, pred):
    mbe_loss = np.sum(pred - true)/true.size
    return mbe_loss/ true.mean() * 100


def r_mean_absolute_error(true, pred):
    mbe_loss = np.sum(abs(pred - true))/true.size
    return mbe_loss/ true.mean() * 100


def rmsd(true, pred):
    return np.sqrt(sum((pred - true) ** 2) / true.size) 


def rrmsd(true, pred):
    return np.sqrt(sum((pred - true) ** 2) / true.size) / true.mean() * 100