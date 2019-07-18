import openface
import cv2
import time

start = time.time()

dlibFacePredictor = 'shape_predictor_68_face_landmarks.dat'
networkModel = 'models/nn4.small2.v1.t7'
imgDim = 96
verbose = True

align = openface.AlignDlib(dlibFacePredictor)

net = openface.TorchNeuralNet(networkModel, imgDim)

def getRep(bgrImg, imgPath=''):
    if bgrImg is None:
        print ("Unable to load image: {}".format(imgPath))
        return None

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        print("Unable to find a face: {}".format(imgPath))
        return None

    if verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        print("Unable to align image: {}".format(imgPath))
        return None
    if verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    if verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
    return rep