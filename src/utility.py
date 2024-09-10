#! /usr/bin/env python3
'''===================================================================================================================
 - Autor: M. Guesmi
 - Adress: Chair of process engineering TU Dresden, Helmholtzstraße 10, 01069 Dresden, Germany
 - Summary: Define utility functions
======================================================================================================================'''

import numpy as np
from scipy import optimize
# > import random
from functools import reduce

# > function to define an array with startpoint, endpoint and step
def linspace(start, stop, step=1.):
  """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
  """
  return np.linspace(start, stop, int((stop - start) / step + 1))



# > function to generate n numbers between a and b
def gen_avg(expected_avg=27, n=22, a=20, b=46):
    while True:
        l = [np.random.randint(a, b) for i in range(n)]
        avg = reduce(lambda x, y: x + y, l) / len(l)

        if avg == expected_avg:
            return l

# > function to return angle for slug flow with gas fraction = a
def angle(a):

    def f(x):
        return np.sin(x) - x + 2 * a * np.pi

    def df(x):
        return np.cos(x) - 1

    x = np.pi * 1.0
    iterationNumber = 1000.
    tol             = 1e-6
    x_old = x
    x = x - f(x) / df(x)
    d = abs(x - x_old)
    i = 0
    while ((d > tol)and ( i < iterationNumber)):
        x_old = x
        x     = x - f(x) / df(x)
        if (d <= tol):
            break
        if ( i > iterationNumber):
            print("error")
            break
        i = i + 1

    return x

# > bisectin method
def my_bisection(f, data, a, b, tol):
    # approximates a root, R, of f bounded
    # by a and b to within tolerance
    # | f(m) | < tol with m the midpoint
    # between a and b Recursive implementation

    # check if a and b bound a root
    if np.sign(f(a, *data)) == np.sign(f(b, *data)):
        raise Exception(
            "The scalars a and b do not bound a root")

    # get midpoint
    m = (a + b) / 2

    if np.abs(f(m, *data)) < tol:
        # stopping condition, report m as root
        return m
    elif np.sign(f(a, *data)) == np.sign(f(m, *data)):
        # case where m is an improvement on a.
        # Make recursive call with a = m
        return my_bisection(f, data, m, b, tol)
    elif np.sign(f(b, *data)) == np.sign(f(m, *data)):
        # case where m is an improvement on b.
        # Make recursive call with b = m
        return my_bisection(f, data, a, m, tol)

# > Linear Interpolation
def interp(T, T1, T2, v1, v2):
    coef = (v2 - v1) / (T2 - T1)
    v    = coef * (T - T1) + v1
    return v


def separate():
    print('\n' + '=' * 29 + '\n')


def pr(msg, maxlen=29):
    """Fit msg in the screen"""
    if type(msg) != str:
        msg = str(msg)

    if len(msg) <= maxlen:
        print(msg)
        return

    i = msg.find(" ")
    lasti = -1
    while i != -1:
        if i >= maxlen:
            if lasti == -1:
                break
            print(msg[:lasti])
            pr(msg[lasti + 1:], maxlen)
            break
        #
        lasti = i
        i = msg.find(" ", i + 1)
    # Couldn't split
    if lasti == -1:
        print(msg)
    elif i == -1:
        print(msg[:lasti])
        print(msg[lasti + 1:])


def get(varname):
    try:
        inp = float(input(varname + ' = '))
        return inp
    except KeyboardInterrupt:
        return
    except Exception as e:
        print(e)
        return get(varname)


def get_params(lst):
    params = {}
    for p in lst:
        params[p] = get(p)
    pr(params)
    return params

# scientific output format of results
sci = lambda x: "{:.4e}".format(x)


def out(desc, val):
    if val > 9999 or val < 0.000999999:
        val = sci(val)
    else:
        val = "{:.5}".format(val)
    print(desc + ' = ' + val)


def choose_opt(descr, optmap):
    """Only lowcase letter in optmap !"""
    intmap = {i + 1: opt for i, opt in enumerate(optmap)}
    # opt_txt = ', '.join([str(k)+': '+e for k,e in intmap.items()])
    pr('Choose ' + descr + ': ')
    pr(intmap)
    inp = input('> ')
    try:
        inp = int(inp)
        inp = intmap[inp]
    except:
        inp = inp.lower()
    separate()
    return optmap[inp]()

