import tensorflow as tf
import os
import cv2
from pathlib import Path

def create_tfrecord(training_dir, output_path):

    writer = tf.io.TFRecordWriter(str(output_path))
    training_dir = Path(training_dir)
    
    # get all image files
    image_files = sorted([f for f in training_dir.glob('frame_*.jpg')])
    total = len(image_files)
    
    print(f"processing {total} files...")
    
    for i, image_path in enumerate(image_files, 1):
        try:
            # build corresponding landmark file path
            landmark_path = training_dir / image_path.name.replace('frame_', 'landmarks_').replace('.jpg', '.txt')
            
            if not landmark_path.exists():
                print(f"cannot find landmark file, skip: {landmark_path}")
                continue
                
            # read landmark data
            landmarks = []
            with open(landmark_path, 'r') as f:
                for line in f:
                    _, x, y, z = map(float, line.strip().split(','))
                    landmarks.extend([x, y, z])  # each landmark x, y, z coordinates
            
            # read and encode image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"cannot read image, skip: {image_path}")
                continue
                
            # convert to RGB (OpenCV default is BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_encoded = cv2.imencode('.jpg', img)[1].tobytes()
            
            # Get filename without extension to check if it's for left hand (_mr ending)
            filename = Path(image_path).stem
            
            # Create feature dictionary with the additional hand type information
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_encoded])),
                'landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=landmarks)),
                'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()]))
            }
            
            # create Example and write
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
            if i % 10 == 0:
                print(f"processed: {i}/{total}")
            
        except Exception as e:
            print(f"error processing file {image_path.name}: {e}")
            continue
    
    writer.close()
    print(f"\nprocessing completed, saved to: {output_path}")

def create_example(image_path, landmarks):
    # read image
    with tf.io.gfile.GFile(str(image_path), 'rb') as fid:
        encoded_image = fid.read()
    
    # get filename (without path and extension)
    filename = Path(image_path).stem
    
    # create feature dictionary
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
        'landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=landmarks)),
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()]))
    }
    
    # create Example
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def main():
    # get script directory
    script_dir = Path(__file__).parent
    training_dir = script_dir / 'training'
    output_path = script_dir / 'training_data.tfrecord'
    
    if not training_dir.exists():
        print(f"training data directory not found: {training_dir}")
        return
    
    # convert Path object to string
    create_tfrecord(str(training_dir), str(output_path))

if __name__ == "__main__":
    main()
