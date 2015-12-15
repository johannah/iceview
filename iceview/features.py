
class ZernikeMoments:
    def __init__(self, radius):
        """
        :param radius: the maximum radius for the Zernike polynomials, in pixels
        """
        from mahotas.features import zernike_moments
        self.radius = radius

    def detect_and_extract(self, image):
        self.moments = zernike_moments(image, self.radius)

def detect_and_extract(detector, img):
    detector.detect_and_extract(img)
    keypoints = detector.keypoints
    descriptors = detector.descriptors
    return keypoints, descriptors


