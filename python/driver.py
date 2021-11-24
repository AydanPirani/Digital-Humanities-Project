from ImageProcessorFactory import ImageProcessorFactory

factory = ImageProcessorFactory()
processor1 = factory.create_processor("ariana_grande", "../images/ariana_grande.jpg")
# processor2 = factory.create_processor("beyonce", "../images/beyonce.jpg")
processor1.process()
print(processor1.get_diff())
print(processor1.get_spec())