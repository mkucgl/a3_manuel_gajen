Programming Assignment 3
========================

Handed in by Manuel Kunz and Gajendira Sivajothi

Introduction
------------

The code that we produced to solve the programming assignment can be found in the file "assignment3.py". It can be run like this: `python3 assignment3.py`.

Exercise 1
----------

Technical Errors and Solutions
------------------------------

When trying to use the `IMD_Resolver` for the first time, we faced the error "'urllib' has no attribute 'parse'". Adding `import urllib.parse` solved the problem. We had to add other imports: `import json`, `import sys`, `import urllib.request`. One of us had to update VirtualBox to get rid of the error "ConnectionResetError(104, 'Connection reset by peer')".
