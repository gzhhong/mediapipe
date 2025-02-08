import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def check_data_consistency(landmarks):
    # check coordinate range
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    z_coords = landmarks[:, 2]
    
    print("\ncoordinate range check:")
    print(f"X range: [{np.min(x_coords):.3f}, {np.max(x_coords):.3f}]")
    print(f"Y range: [{np.min(y_coords):.3f}, {np.max(y_coords):.3f}]")
    print(f"Z range: [{np.min(z_coords):.3f}, {np.max(z_coords):.3f}]")
    
    # check finger joint relative position
    finger_ranges = [
        (0, "wrist"),
        (1, 4, "thumb"),
        (5, 8, "index finger"),
        (9, 12, "middle finger"),
        (13, 16, "ring finger"),
        (17, 20, "pinky")
    ]
    
    print("\nfinger joint check:")
    for finger_range in finger_ranges:
        if len(finger_range) == 2:
            idx, name = finger_range
            print(f"{name} (point {idx}): ({landmarks[idx, 0]:.3f}, {landmarks[idx, 1]:.3f}, {landmarks[idx, 2]:.3f})")
        else:
            start, end, name = finger_range
            print(f"\n{name} chain:")
            for i in range(start, end + 1):
                print(f"joint {i}: ({landmarks[i, 0]:.3f}, {landmarks[i, 1]:.3f}, {landmarks[i, 2]:.3f})")

def verify_tfrecord():
    # verify TFRecord dataset content
    tfrecord_path = str(Path(__file__).parent / 'training_data.tfrecord')
    
    # parse function
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'landmarks': tf.io.FixedLenFeature([63], tf.float32),
        'filename': tf.io.FixedLenFeature([], tf.string)
    }
    
    def parse_tfrecord(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
        landmarks = parsed_features['landmarks']
        return image, landmarks
    
    # read dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    
    # check dataset size
    total_samples = sum(1 for _ in dataset)
    print(f"dataset total samples: {total_samples}")
    
    # check first 5 samples
    for i, (image, landmarks) in enumerate(dataset.take(5)):
        print(f"\n{'='*20} sample {i+1} {'='*20}")
        print(f"image shape: {image.shape}")
        print(f"landmarks shape: {landmarks.shape}")
        
        # reshape landmarks to (21, 3)
        landmarks = tf.reshape(landmarks, [21, 3])
        
        # check data consistency
        check_data_consistency(landmarks.numpy())
        
        # visualize
        plt.figure(figsize=(10, 5))
        
        # show image and landmarks
        plt.subplot(121)
        plt.imshow(image.numpy().astype(np.uint8))
        
        # reshape landmarks to (21, 3) and plot
        landmarks = tf.reshape(landmarks, [21, 3])
        for j, (x, y, z) in enumerate(landmarks):
            plt.plot(x * image.shape[1], y * image.shape[0], 'r.')
            plt.text(x * image.shape[1], y * image.shape[0], str(j), fontsize=8)
        
        # show landmarks 3D view
        ax = plt.subplot(122, projection='3d')
        ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='b', marker='o')
        
        # add point labels
        for j, (x, y, z) in enumerate(landmarks):
            ax.text(x, y, z, str(j), fontsize=8)
        
        # set axis labels
        ax.set_xlabel('X (左右)')
        ax.set_ylabel('Y (上下)')
        ax.set_zlabel('Z (深度)')
        
        # add grid lines
        ax.grid(True)
        
        # set axis range, ensure normalized range
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([-1, 0])  # z axis range is -1 to 0
        
        # add title
        ax.set_title('3D Landmarks View')
        
        plt.tight_layout()
        plt.show()
        
        # print landmarks coordinates
        print("\nlandmarks coordinates:")
        for j, (x, y, z) in enumerate(landmarks):
            print(f"point {j}: x={x:.3f}, y={y:.3f}, z={z:.3f}")

if __name__ == "__main__":
    verify_tfrecord() 