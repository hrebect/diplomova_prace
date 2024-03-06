import scipy.spatial as sc
import numpy, math
import turning_function

class Evaluate:
    def __init__(self):
        pass

    def stack(self, X, Y):
        shape = numpy.vstack((X, Y))
        return shape


    def distance(self, x1, y1, x2, y2):
        # distance of 2 pionts
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return dist

    def Haussdorf(self, X1, Y1, X2, Y2):
        shape1 = self.stack(X1, Y1).T
        shape2 = self.stack(X2, Y2).T

        hd = sc.distance.directed_hausdorff(shape1,shape2)

        return hd[0]

    def chamfer_distance(self, X1, Y1, X2, Y2):
        """
        Computes the chamfer distance between two sets of points A and B.
        https://medium.com/@sim30217/chamfer-distance-4207955e8612
        """
        shape1 = self.stack(X1, Y1).T
        shape2 = self.stack(X2, Y2).T

        tree = sc.KDTree(shape2)
        dist_A = tree.query(shape1)[0]
        tree = sc.KDTree(shape1)
        dist_B = tree.query(shape2)[0]
        return numpy.mean(dist_A) + numpy.mean(dist_B)

    def turn_func(self, X1, Y1, X2, Y2):
        # turning function
        shape1 = self.stack(X1, Y1).T
        shape2 = self.stack(X2, Y2).T

        shape1.tolist()
        shape2.tolist()

        return turning_function.distance(shape1, shape2, brute_force_updates=False)[0]

    def curviature(self, X1, Y1, X2, Y2):
        # compute curviature of 2 given lines as numpy array
        #get number of vertecis
        n = len(X1)

        # initialize result list
        K = []

        # loop all verteces (except 1. and last)
        for i in range(1,n-1):
            # get determinant
            det1 = numpy.linalg.det([[X1[i-1] - X1[i], X1[i+1] - X1[i]], [Y1[i-1] - Y1[i], Y1[i+1] - Y1[i]]])

            # get distances between verteces
            d1_1 = self.distance(X1[i-1], Y1[i-1], X1[i], Y1[i])
            d2_1 = self.distance(X1[i], Y1[i], X1[i+1], Y1[i+1])
            d3_1 = self.distance(X1[i-1], Y1[i-1], X1[i+1], Y1[i+1])

            # compute local curvaure
            k1 = -2 * det1 / (d1_1 * d2_1 * d3_1)

            # same for secon line
            det2 = numpy.linalg.det([[X2[i - 1] - X2[i], X2[i + 1] - X2[i]], [Y2[i - 1] - Y2[i], Y2[i + 1] - Y2[i]]])

            d1_2 = self.distance(X2[i - 1], Y2[i - 1], X2[i], Y2[i])
            d2_2 = self.distance(X2[i], Y2[i], X2[i + 1], Y2[i + 1])
            d3_2 = self.distance(X2[i - 1], Y2[i - 1], X2[i + 1], Y2[i + 1])

            k2 = -2 * det2 / (d1_2 * d2_2 * d3_2)

            # Compute curvature for whole linw
            try:
                k = abs((k1 - k2) / (k1 + k2))
            except:
                k = 0

            K.append(k)

        return numpy.mean(K)






    def get2LinesAngle(self, p1, p2, p3, p4):
        # Get angle between 2 vectors
        ux = p2[0] - p1[0]
        uy = p2[1] - p1[1]
        vx = p4[0] - p3[0]
        vy = p4[1] - p3[1]

        # Cross product
        uv = ux * vx + uy * vy

        # Norms
        nu = (ux ** 2 + uy ** 2) ** 0.5
        nv = (vx ** 2 + vy ** 2) ** 0.5

        # Angle
        try:
            return abs(math.acos(uv / (nu * nv)))
        except:
            return 0


    def llr(self, X1, Y1, X2, Y2):

        # initialize llr list
        LLR = []

        n = len(X1) # all are same saze in this case

        # Loop through feature nodes
        for i in range(1, n-1):

            # Angle between segments of first line
            omega1 = self.get2LinesAngle((X1[i], Y1[i]), (X1[i-1], Y1[i-1]) , (X1[i], Y1[i]), (X1[i+1], Y1[i+1]))

            # Compute local LLR (first)
            omega1/2
            llr1 = math.sin(omega1)

            # secomd line
            omega2 = self.get2LinesAngle((X2[i], Y2[i]), (X2[i-1], Y2[i-1]) , (X2[i], Y2[i]), (X2[i+1], Y2[i+1]))
            omega2/2
            llr2 = math.sin(omega2)


            # compute global LLR
            try:
                k = abs((llr1-llr2)/(llr1+llr2))
            except:
                k = 0

            LLR.append(k)



        return numpy.mean(LLR) #numpy.mean(LLR1) - numpy.mean(LLR2)


