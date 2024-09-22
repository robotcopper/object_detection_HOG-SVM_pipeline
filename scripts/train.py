import dlib


class TrainSVM:

    def train(self):
        # options are detailed here: http://dlib.net/python/index.html#dlib_pybind11.simple_object_detector_training_options
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = True
        options.be_verbose = True
        options.epsilon = 0.001
        options.C = 10
        options.num_threads = 15
        # options.upsample_limit
        # options.nuclear_norm_regularization_strength
        # options.max_runtime_seconds
        # options.detection_window_size

        dlib.train_simple_object_detector('../resources/train_label.xml', '../resources/my_detection_model.svm', options)
        
        # Test the trained object detector ()
        print("Testing the trained detector...")
        test_results = dlib.test_simple_object_detector('../resources/test_label.xml', '../resources/my_detection_model.svm')
        
        # Print the test results
        print(f"Average precision: {test_results.average_precision}")
        print(f"Precision: {test_results.precision}")
        print(f"Recall: {test_results.recall}")
        
if __name__ == '__main__':
    TrainSVM().train()

