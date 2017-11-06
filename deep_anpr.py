"""
    anpr function
    Copyright (C) Willi
"""
import numpy
import sys
import cv2

from detect import *
from detect import letter_probs_to_code


class DeepANPR(object):
    def __init__(self, param_file):
        f = numpy.load(param_file)
        self.param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    def select_code(self, codes):
        if len(codes) > 0:
            return codes[0]
        else:
            return []

    def detect_anpr(self, im):
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

        codes = []
        for pt1, pt2, present_prob, letter_probs in post_process(
                                                      detect(im_gray, self.param_vals)):
            pt1 = tuple(map(int, reversed(pt1)))
            pt2 = tuple(map(int, reversed(pt2)))

            code = letter_probs_to_code(letter_probs)
            codes.append( code)

        return self.select_code(codes)
