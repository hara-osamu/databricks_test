# Databricks notebook source
from setuptools import setup, find_packages

setup(
    name='addcol',
    version='1.0',
    license='proprietary',
    description='addcol',

    author='hara_osamu',
    author_email='hara_osamu@comture.com',
    url='None.com',

    packages=['addcol'],
    install_requires=['spark'],
)

