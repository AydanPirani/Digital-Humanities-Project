from ImageProcessor import ImageProcessor

class ImageProcessorFactory:
    def __init__(self, params={}):
        print("created factory!")
        self.MAX_FACES = params.get("max_faces", 1)
        self.USE_STDEVS = params.get("use_stdevs", False)
        self.DISPLAY_POINTS = params.get("display_points", False)
        self.POINTS_THRESHOLD = params.get("points_threshold", 20)

    def create_processor(self, img_id, img_path):
        return ImageProcessor(img_id, img_path, self.MAX_FACES, self.POINTS_THRESHOLD, self.DISPLAY_POINTS, self.USE_STDEVS)