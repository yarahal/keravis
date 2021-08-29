from math import sqrt

def find_closest_factors(n):
    '''
    Efficient algorithm for finding the closest two factors of a number n
    '''
    x = int(sqrt(n))
    y = n/x
    while y != int(y):
        x -= 1
        y = n/x
    return x, int(y)