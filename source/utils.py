"""
Description: Generic functions that are used in the project

"""

import csv
import os
import random
import string
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

# pylint: disable=E1101

# Constants
LOG_FILE = "Logs/log.txt"
DATASET_TRAINING = "dataset/csv_dataset/pixel/train"
DATASET_TESTING = "dataset/csv_dataset/pixel/test"
DATASET_VALIDATION = "dataset/csv_dataset/pixel/validation"
CSV_DATASET_TRAINING = f"{DATASET_TRAINING}/train.csv"
CSV_DATASET_TESTING = f"{DATASET_TESTING}/test.csv"
CSV_DATASET_VALIDATION = f"{DATASET_VALIDATION}/validation.csv"

LANDMARKS_DATASET_DIR = "dataset/csv_dataset/landmarks/landmarks_indi"
LANDMARKS_DIR = "dataset/csv_dataset/landmarks"

IMAGES_DIR = "dataset/images"

EXTERNAL_DATASET_DIR = "dataset/external_dataset"
ESC_KEY_ASCII = 27
EXCLUDED_LETTERS = "JZ"
IMAGE_FILE_FORMATS = (".png", ".jpg", ".jpeg")

letter_predictions = [s for s in string.ascii_uppercase if s not in EXCLUDED_LETTERS]

letter_to_number = {letter: index for index, letter in enumerate(letter_predictions)}

number_to_letter = {index: l for index, l in enumerate(letter_predictions)}


def log_error(error: str) -> None:
    """
    Log the error.
    :param error: The error
    """
    # If the log file does not exist, create it
    if not os.path.exists(LOG_FILE):
        log_file()
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{error}\n")


def exit_on_close() -> bool:
    """
    Check if the window is closed
    :return: True if the window is closed, False otherwise
    """
    try:
        return cv2.getWindowProperty("frame", 0) < 0
    except cv2.error as e:  # pylint: disable=E0712
        # write into log file
        log_error(str(e))
        return True


def exit_on_key() -> bool:
    """
    Check if the key is pressed
    :return: True if the key is pressed, False otherwise
    """
    try:
        return cv2.waitKey(1) & 0xFF in [ord("q"), ESC_KEY_ASCII]
    except cv2.error as e:  # pylint: disable=E0712
        # write into log file
        log_error(str(e))
        return True


def detect_key(key: str) -> bool:
    """
    Detect if the key is pressed
    :param key: str
    :return: True if the key is pressed, False otherwise
    """
    try:
        return cv2.waitKey(1) & 0xFF == ord(key)
    except cv2.error as e:  # pylint: disable=E0712
        # write into log file
        log_error(str(e))
        return True


def adjust_min_max(  # pylint: disable=R0913
    offset: int, original_frame: Any, x_max: int, x_min: int, y_max: int, y_min: int
) -> tuple[int, int, int, int]:
    """
    Adjust the minimum and maximum coordinates
    :param offset: The offset
    :param original_frame: The original frame
    :param x_max: The maximum x coordinate
    :param x_min: The minimum x coordinate
    :param y_max: The maximum y coordinate
    :param y_min: The minimum y coordinate
    :return: The adjusted minimum and maximum coordinates
    """
    # Adjust for the offset
    x_min, y_min = x_min - offset, y_min - offset
    x_max, y_max = x_max + offset, y_max + offset
    # Make sure the coordinates fit within the frame
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(original_frame.shape[0], y_max)
    x_max = min(original_frame.shape[1], x_max)
    return x_max, x_min, y_max, y_min


def coords_calc(  # pylint: disable=R0913
    id_input: int,
    frame: Any,
    lm: Any,
    lm_list: list[list[float]],
    x_max: int,
    x_min: int,
    y_max: int,
    y_min: int,
) -> tuple[int, int, int, int, int, int]:
    """
    Calculate the coordinates
    :param id_input: The id of the input
    :param frame: The frame
    :param lm: The landmark
    :param lm_list: The list of landmarks
    :param x_max: The maximum x coordinate
    :param x_min: The minimum x coordinate
    :param y_max: The maximum y coordinate
    :param y_min: The minimum y coordinate
    :return: The calculated coordinates
    """
    h, w, _ = frame.shape
    cx, cy = int(lm.x * w), int(lm.y * h)
    lm_list.append([id_input, cx, cy])
    x_min, y_min = min(x_min, cx), min(y_min, cy)
    x_max, y_max = max(x_max, cx), max(y_max, cy)
    return x_max, x_min, y_max, y_min, cx, cy


def clear_img_input() -> None:
    """
    Clear the img_input folder of all the images
    """

    files = os.listdir("img_input")
    for f in files:
        if f.endswith(".png"):
            os.remove(os.path.join("img_input", f))


def log_file() -> None:
    """
    Create a log file
    """
    with open("Logs/log.txt", "w", encoding="utf-8") as f:
        f.write("Log file\n")


def create_directory(directory: str) -> None:
    """
    Create a directory
    :param directory: The directory
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    # add placeholder file in directory
    with open(f"{directory}/dummy.txt", "w", encoding="utf-8") as f:
        f.write("dummy file")


def create_alphabet_directory(directory: str) -> None:
    """
    Create a directory for each alphabet
    :param directory: The directory
    """

    for letter in string.ascii_lowercase:
        create_directory(os.path.join(directory, f"{letter}_dir"))


def create_number_directory(directory: str) -> None:
    """
    Create a directory for each number
    :param directory: The directory
    """

    for number in range(10):
        create_directory(os.path.join(directory, f"{number}_dir"))


def img_set_dir_to_csv(img_dir: str) -> None:
    """
    Directory of subdirectories, where the content of subdirectories is the images
    that need to be converted to a csv file. The name of the subdirectory is the
    label of the images in the subdirectory.

    :param img_dir: directory of subdirectories.
    """
    # iterate through each subdirectory
    for subdir, _, files in os.walk(img_dir):
        # Split the files into training, validation, and testing, using sample
        # size of 0.6, 0.3, and 0.1 respectively
        if isinstance(files, list) and len(files) > 0:
            training_files = random.sample(files, int(0.6 * len(files)))
            remaining_files = [x for x in files if x not in training_files]
            validation_files = random.sample(remaining_files, int(0.3 * len(files)))
            testing_files = [x for x in remaining_files if x not in validation_files]

            label = os.path.basename(subdir)[0]

            print(f"Processing {label.upper()}")

            if label.upper() in EXCLUDED_LETTERS:
                print(f"Excluded letter: {label}")
                continue

            label = letter_to_number[label.upper()] if label[0].isalpha() else label[0]

            print(f"Training: {len(training_files)}")
            print(f"Validation: {len(validation_files)}")
            print(f"Testing: {len(testing_files)}")

            # create csv file
            write_to_csv("train.csv", str(label), subdir, training_files, "train")

            write_to_csv(
                "validation.csv", str(label), subdir, validation_files, "validation"
            )

            write_to_csv("test.csv", str(label), subdir, testing_files, "test")

    train_df = pd.read_csv(CSV_DATASET_TRAINING)
    val_df = pd.read_csv(CSV_DATASET_VALIDATION)
    test_df = pd.read_csv(CSV_DATASET_TESTING)

    print("Training data:", len(train_df))
    print("Validation data:", len(val_df))
    print("Testing data:", len(test_df))

    # Shuffle the data 5 times
    for _ in range(5):
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        val_df = val_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)

    train_df.to_csv(CSV_DATASET_TRAINING, index=False)
    val_df.to_csv(CSV_DATASET_VALIDATION, index=False)
    test_df.to_csv(CSV_DATASET_TESTING, index=False)

    print("Shuffled data")
    labels = train_df["label"].unique()
    labels.sort()
    print("Labels:", labels)
    print("Labels:", [number_to_letter[label] for label in labels])


def write_to_csv(
    csv_file: str, label: str, subdir: str, files: list[str], set_type: str
) -> None:
    """
    Write the files to the csv file
    :param csv_file: The csv file
    :param label: The label
    :param subdir: The subdirectory
    :param files: The files
    :param set_type: The set type (train, validation, test)
    """
    with open(
        f"dataset/csv_dataset/pixel/{set_type}/{csv_file}",
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        # create first row of csv file | label, pixel0, pixel1, pixel2, ...
        if os.stat(f"dataset/csv_dataset/pixel/{set_type}/{csv_file}").st_size == 0:
            writer.writerow(["label"] + [f"pixel{i}" for i in range(784)])

        for i in files:
            if i.endswith(IMAGE_FILE_FORMATS):
                img = cv2.imread(os.path.join(subdir, i), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (28, 28))
                img = img.flatten()
                img = img.tolist()
                img.insert(0, label)  # type: ignore
                writer.writerow(img)


def check_rename_files(subdir: str, label: str, file: str) -> None:
    """
    Check if the file exists in the dataset, if it does, rename the file with length
    of 15 random strings. Check again if the file exists in the dataset, if it does,
    rename the file again.

    :param subdir: subdirectory of the file
    :param label: label of the image
    :param file: file name
    """

    while True:
        new_name = (
            "".join(random.choices(string.ascii_lowercase + string.digits, k=15))
            + ".png"
        )
        if not os.path.exists(f"{EXTERNAL_DATASET_DIR}/{label}_dir/{new_name}"):
            os.rename(
                os.path.join(subdir, file),
                f"{EXTERNAL_DATASET_DIR}/{label}_dir/{new_name}",
            )
            break


def labeler(file: str) -> str:
    """
    Label the file based on the first letter of the file name.

    :param file: The file name
    :return: The label
    """
    return file[0].lower() if file[0].isalpha() else file[0]


def user_files_to_dir(user_dir: str) -> None:
    """
    Move the files from the given directory to the appropriate directory in the
    dataset based on the label of the image.

    If the file is not an image, it is ignored. If the file already exists in the
    dataset, rename the file with length of 15 random strings. Check again if the
    file exists in the dataset, if it does, rename the file again.

    :param user_dir: directory of user files
    """

    for subdir, _, files in os.walk(user_dir):
        for file in files:
            if file.endswith(IMAGE_FILE_FORMATS):
                # Check first letter of the file to determine the label
                label = labeler(file)
                if not os.path.exists(f"{EXTERNAL_DATASET_DIR}/{label}_dir/{file}"):
                    os.rename(
                        os.path.join(subdir, file),
                        f"dataset/{label}_dir/{file}",
                    )
                else:
                    check_rename_files(subdir, label, file)


def user_files_to_dir_counted(user_dir: str, reset: bool = False) -> None:
    """
    Moves the files from the given directory to the appropriate directory in the
    dataset based on the label of the image. If the file is not an image,
    it is ignored. If the file already exists in the dataset, rename the file with
    the next available number. (based on the current number of files in the directory)

    :param user_dir: directory of user files
    :param reset: boolean value that determines if the count of the files in the
    directory is reset
    """
    for subdir, _, files in os.walk(user_dir):
        count = 0 if reset else len(os.listdir(subdir))
        for file in files:
            if file.endswith(IMAGE_FILE_FORMATS):
                # Check first letter of the file to determine the label
                label = subdir.split("\\")[-1][0].lower()
                os.rename(
                    os.path.join(subdir, file),
                    os.path.join(subdir, f"{label}{count}.png"),
                )
                count += 1


def get_file_count(user_dir: str) -> int:
    """
    Get the total number of files in the given directory.
    :param user_dir: The directory
    :return: The total number of files
    """
    return sum(
        file.endswith(IMAGE_FILE_FORMATS)
        for _, _, files in os.walk(user_dir)
        for file in files
    )


def move_files_to_dir_counted(user_dir: str, move_dir: str) -> None:
    """
    Moves the files from the given directory to the appropriate directory in the
    dataset based on the label of the image. If the file is not an image,
    it is ignored. If the file already exists in the dataset, rename the file with
    the next available number. (based on the current number of files in the directory)

    :param user_dir: directory of user files
    :param move_dir: directory to move the files to
    """
    for subdir, _, files in os.walk(user_dir):
        print(f"Processing {subdir}")
        for file in files:
            if file.endswith(IMAGE_FILE_FORMATS):
                # Check first letter of the file to determine the label
                label = labeler(subdir.split("\\")[-1])
                print(f"Moving {file} to {move_dir}/{label}_dir")
                # Get the count of files in the move_dir
                count = get_file_count(f"{move_dir}/{label}_dir")

                os.rename(
                    os.path.join(subdir, file),
                    os.path.join(f"{move_dir}/{label}_dir", f"{label}{count}.png"),
                )
                count += 1


def rename_to_uppercase(directory: str) -> None:
    """
    Rename the files in the given directory to uppercase.
    :param directory: The directory
    """
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(IMAGE_FILE_FORMATS):
                os.rename(
                    os.path.join(subdir, file),
                    os.path.join(subdir, file[0].upper() + file[1:]),
                )


def add_header_to_csv(
    training: str = CSV_DATASET_TRAINING,
    validation: str = CSV_DATASET_VALIDATION,
    testing: str = CSV_DATASET_TESTING,
) -> None:
    """
    Add header to the beginning of the csv file
    """
    for csv_file in (training, validation, testing):
        with open(csv_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            data = tuple(reader)

        # Check to avoid duplicate headers
        if data[0] == ["label"] + [f"pixel{i}" for i in range(784)]:
            continue

        with open(csv_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["label"] + [f"pixel{i}" for i in range(784)])
            for row in data:
                writer.writerow(row)


def add_mnist_to_csv():
    """
    Add the mnist dataset to the csv file
    """
    mnist_train = "dataset/csv_dataset/train/sign_mnist_train.csv"
    mnist_test = "dataset/csv_dataset/test/sign_mnist_test.csv"

    data = pd.read_csv(mnist_train).values.tolist()
    data += pd.read_csv(mnist_test).values.tolist()

    # Split the data into training, validation, and testing, using sample
    # size of 0.6, 0.3, and 0.1 respectively
    training_data = random.sample(data, int(0.6 * len(data)))
    remaining_data = [x for x in data if x not in training_data]
    validation_data = random.sample(remaining_data, int(0.3 * len(data)))
    test_data = [x for x in remaining_data if x not in validation_data]

    print("Adding mnist to csv")
    print(f"Training data: {len(training_data)}")
    print(f"Validation data: {len(validation_data)}")
    print(f"Test data: {len(test_data)}")
    print(f"Total data: {len(training_data) + len(validation_data) + len(test_data)}")
    for csv_file, data_set in zip(
        (CSV_DATASET_TRAINING, CSV_DATASET_VALIDATION, CSV_DATASET_TESTING),
        (training_data, validation_data, test_data),
    ):
        with open(csv_file, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for row in data_set:
                writer.writerow(row)


def reduce_dataset(new_size: int):
    # choose new_size numbers from 0 to 23
    # numbers = random.sample(range(24), new_size)
    numbers = range(new_size)
    # mnist_train = "dataset/csv_dataset/pixel/train/sign_mnist_train.csv"
    # mnist_test = "dataset/csv_dataset/pixel/test/sign_mnist_test.csv"

    data = pd.read_csv(CSV_DATASET_TRAINING)
    # Add the validation data to the training data
    data = pd.concat([data, pd.read_csv(CSV_DATASET_VALIDATION)], ignore_index=True)
    # Add the test data to the training data
    data = pd.concat([data, pd.read_csv(CSV_DATASET_TESTING)], ignore_index=True)
    
    # Convert labels to integers
    data["label"] = data["label"].astype(int)

    reduced_df = data[data["label"].isin(numbers)]
    
    print(f"Reduced data: {len(reduced_df)}")
    train_samples = int(0.6 * len(reduced_df))
    val_samples = int(0.3 * len(reduced_df))
    test_samples = len(reduced_df) - train_samples - val_samples
    
    training_df = reduced_df.sample(n=train_samples, random_state=1)
    remaining_df = reduced_df.drop(training_df.index)
    validation_df = remaining_df.sample(n=val_samples, random_state=1)
    test_data_df = remaining_df.drop(validation_df.index)

    print("Writing data to csv")
    print(f"Training data: {len(training_df)}")
    # print(f"Remaining data: {len(remaining_df)}")
    print(f"Validation data: {len(validation_df)}")
    print(f"Test data: {len(test_data_df)}")
    print(f"Total data: {len(training_df) + len(validation_df) + len(test_data_df)}")

    for csv_file, data_set in zip(
        (
            f"{DATASET_TRAINING}/train_{new_size}.csv",
            f"{DATASET_VALIDATION}/validation_{new_size}.csv",
            f"{DATASET_TESTING}/test_{new_size}.csv",
        ),
        (training_df, validation_df, test_data_df),
    ):
        data_set.to_csv(csv_file, index=False, header=True)


def reduce_landmarks_dataset(new_size: int):
    # choose new_size numbers from 0 to 23
    # numbers = random.sample(range(24), new_size)
    numbers = range(new_size)

    train_df = pd.read_csv(f"{LANDMARKS_DIR}/train/landmarks_train.csv")
    test_df = pd.read_csv(f"{LANDMARKS_DIR}/test/landmarks_test.csv")
    validation_df = pd.read_csv(f"{LANDMARKS_DIR}/validation/landmarks_validation.csv")
    
    landmark_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)
    
    reduced_data = landmark_df[landmark_df["label"].isin(numbers)].values.tolist()
    print("Reducing dataset")
    print(f"Reduced data: {len(reduced_data)}")

    training_data = random.sample(reduced_data, int(0.6 * len(reduced_data)))
    remaining_data = [x for x in reduced_data if x not in training_data]
    validation_data = random.sample(remaining_data, int(0.3 * len(reduced_data)))
    test_data = [x for x in remaining_data if x not in validation_data]

    # Shuffle the data 5 times
    for _ in range(5):
        random.shuffle(training_data)
        random.shuffle(validation_data)
        random.shuffle(test_data)

    print("Adding landmarks to csv")
    print(f"Training data: {len(training_data)}")
    print(f"Validation data: {len(validation_data)}")
    print(f"Test data: {len(test_data)}")

    for csv_file, data_set in zip(
        (
            f"{LANDMARKS_DIR}/train/landmarks_train_{new_size}.csv",
            f"{LANDMARKS_DIR}/validation/landmarks_validation_{new_size}.csv",
            f"{LANDMARKS_DIR}/test/landmarks_test_{new_size}.csv",
        ),
        (training_data, validation_data, test_data),
    ):
        with open(csv_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for row in data_set:
                writer.writerow(row)

    add_landmarks_header_to_csv(
        f"{LANDMARKS_DIR}/train/landmarks_train_{new_size}.csv",
        f"{LANDMARKS_DIR}/validation/landmarks_validation_{new_size}.csv",
        f"{LANDMARKS_DIR}/test/landmarks_test_{new_size}.csv",
    )


def normalize_dataset(save_sample: bool = False, reduced: bool = False):
    print(f"Processing{' reduced ' if reduced else ' '}dataset")
    
    nrows = 4 if not reduced else 3
    ncols = 6 if not reduced else 2
    
    # Concatenate the training and testing datasets
    if reduced:
        train_df = pd.read_csv(f"{DATASET_TRAINING}/train_6.csv")
        test_df = pd.read_csv(f"{DATASET_TESTING}/test_6.csv")
        validation_df = pd.read_csv(f"{DATASET_VALIDATION}/validation_6.csv")
    else:
        train_df = pd.read_csv(CSV_DATASET_TRAINING)
        test_df = pd.read_csv(CSV_DATASET_TESTING)
        validation_df = pd.read_csv(CSV_DATASET_VALIDATION)

    df = pd.concat([train_df, validation_df, test_df])

    data = df.values

    # separate the label column
    labels = data[:, 0]
    total = len(np.unique(labels))
    print(f"Min label: {min(labels)}")
    print(f"Max label: {max(labels)}")

    data = data[:, 1:]
    print(labels)
    # Get a list of indices from the labels, one for each letter
    indices = [np.where(labels == i)[0][0] for i in range(total)]

    print(f"Indices chosen: {indices}")

    # Calculate the mean image
    mean_image = np.mean(data, axis=0)
    print("Calculate mean image")

    # Choose a random image to display
    # grid_size = 5
    if save_sample:
        mean_image_reshaped = mean_image.reshape(28, 28)
        plt.imshow(mean_image_reshaped, cmap="gray")
        plt.title("Mean Image")
        plt.axis("off")
        plt.savefig(f"mean_image{'_6' if reduced else ''}.png")

        plt.figure(figsize=(10, 10))

        # Show an image of each letter before normalization
        for idx, i in enumerate(indices):
            image = data[i]
            image_reshaped = image.reshape(28, 28)

            # Create a subplot for each image
            plt.subplot(nrows, ncols, idx + 1)
            plt.imshow(image_reshaped, cmap="gray")
            plt.title(f"Letter {number_to_letter[labels[i]]}")
            plt.axis("off")  # Hide axis
        plt.suptitle("Original images")
        # Save the entire grid to a single file
        plt.savefig(f"letters_grid{'_6' if reduced else ''}.png")

    # Normalize the images by subtracting the mean image
    centered_data = data - mean_image
    print("Centered dataset")
    if save_sample:
        plt.figure(figsize=(10, 10))

        # Show an image of each letter before normalization
        for idx, i in enumerate(indices):
            image = centered_data[i]
            image_reshaped = image.reshape(28, 28)

            # Create a subplot for each image
            plt.subplot(nrows, ncols, idx + 1)
            plt.imshow(image_reshaped, cmap="gray")
            plt.title(f"Letter {number_to_letter[labels[i]]}")
            plt.axis("off")  # Hide axis
        plt.suptitle("Centered images")

        # Save the entire grid to a single file
        plt.savefig(f"letters_grid_centered{'_6' if reduced else ''}.png")

    # Rescale the dataset to have values between 0 and 1
    normalized_data = centered_data / 255.0
    print("Rescaled dataset")

    if save_sample:
        plt.figure(figsize=(10, 10))

        # Show an image of each letter before normalization
        for idx, i in enumerate(indices):
            image = normalized_data[i]
            image_reshaped = image.reshape(28, 28)

            # Create a subplot for each image
            plt.subplot(nrows, ncols, idx + 1)
            plt.imshow(image_reshaped, cmap="gray")
            plt.title(f"Letter {number_to_letter[labels[i]]}")
            plt.axis("off")  # Hide axis
        plt.suptitle("Normalized images")
        # Save the entire grid to a single file
        plt.savefig(f"letters_grid_normalized{'_6' if reduced else ''}.png")

    # Calculate the standard deviation of the dataset
    std_dev = np.std(normalized_data)
    print(f"Calculate standard deviation of {std_dev}")

    # Normalize the dataset by dividing by the standard deviation
    standardized_data = normalized_data / std_dev
    print("Standardized dataset")

    if save_sample:
        plt.figure(figsize=(10, 10))

        # Show an image of each letter before normalization
        for idx, i in enumerate(indices):
            image = standardized_data[i]
            image_reshaped = image.reshape(28, 28)

            # Create a subplot for each image
            plt.subplot(nrows, ncols, idx + 1)
            plt.imshow(image_reshaped, cmap="gray")
            plt.title(f"Letter {number_to_letter[labels[i]]}")
            plt.axis("off")  # Hide axis
        plt.suptitle("Standardized images")
        # Save the entire grid to a single file
        plt.savefig(f"letters_grid_standardized{'_6' if reduced else ''}.png")

    # Add the label column back
    final_data = np.column_stack((labels, standardized_data))

    print("Added label column")

    # Split the data into training and testing datasets
    train_data = final_data[: train_df.shape[0]]
    validation_data = final_data[
        train_df.shape[0] : train_df.shape[0] + validation_df.shape[0]
    ]
    test_data = final_data[
        train_df.shape[0] + validation_df.shape[0] :
    ]

    # Convert the numpy array back to a DataFrame
    final_train_df = pd.DataFrame(train_data)
    final_test_df = pd.DataFrame(test_data)
    final_val_df = pd.DataFrame(validation_data)
    # Convert 'label' column back to integers
    final_train_df[0] = final_train_df[0].astype(int)
    final_test_df[0] = final_test_df[0].astype(int)
    final_val_df[0] = final_val_df[0].astype(int)

    print("Converted to DataFrame")

    # Create headers
    headers = ["label"] + ["pixel" + str(i) for i in range(final_data.shape[1] - 1)]
    final_train_df.columns = headers
    final_test_df.columns = headers
    final_val_df.columns = headers
    print("Added headers")

    # Write the data to a new csv file
    new_file = CSV_DATASET_TRAINING.replace(".csv", f"{'_6' if reduced else ''}_normalized.csv")
    final_train_df.to_csv(new_file, index=False)

    print("Saved normalized dataset to", new_file)

    new_file = CSV_DATASET_TESTING.replace(".csv", f"{'_6' if reduced else ''}_normalized.csv")
    final_test_df.to_csv(new_file, index=False)

    print("Saved normalized dataset to", new_file)

    new_file = CSV_DATASET_VALIDATION.replace(".csv", f"{'_6' if reduced else ''}_normalized.csv")
    final_val_df.to_csv(new_file, index=False)

    print("Saved normalized dataset to", new_file)
    
    # Print the size of the dataset
    print("Training data:", len(final_train_df))
    print("Validation data:", len(final_val_df))
    print("Testing data:", len(final_test_df))


def add_landmarks_header_to_csv(
    training: str = CSV_DATASET_TRAINING,
    validation: str = CSV_DATASET_VALIDATION,
    testing: str = CSV_DATASET_TESTING,
) -> None:
    """
    Add header to the beginning of the csv file
    """
    for csv_file in (training, validation, testing):
        with open(csv_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            data = tuple(reader)

        # Check to avoid duplicate headers
        if data[0] == ["label"] + [f"pixel{i}" for i in range(21)]:
            continue

        with open(csv_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["label"] + [f"pixel{i}" for i in range(21)])
            for row in data:
                writer.writerow(row)


def split_landmarks():
    training_set = []
    validation_set = []
    testing_set = []

    # Read all the landmark csv files in the directory
    for subdir, _, files in os.walk(LANDMARKS_DATASET_DIR):
        for file in files:
            if file.endswith(".csv"):
                with open(
                    os.path.join(subdir, file), "r", newline="", encoding="utf-8"
                ) as f:
                    reader = csv.reader(f)
                    data = list(reader)

                    # Remove the header
                    data = data[1:]

                    # Add the label to the data
                    label = file.split("\\")[-1].split("_")[0]
                    # translate the label to a number
                    if label.upper() in EXCLUDED_LETTERS:
                        print(f"Excluded letter: {label}")
                        continue
                    print(f"Processing {label.upper()}")
                    label = letter_to_number[label.upper()]

                    data = [[label] + row for row in data]

                    # Split the data into training, validation, and testing
                    training_data = random.sample(data, int(0.6 * len(data)))
                    remaining_data = [x for x in data if x not in training_data]
                    validation_data = random.sample(
                        remaining_data, int(0.3 * len(data))
                    )
                    testing_data = [
                        x for x in remaining_data if x not in validation_data
                    ]

                    training_set.extend(training_data)
                    validation_set.extend(validation_data)
                    testing_set.extend(testing_data)

    print("Training set of length", len(training_set))
    print("Validation set of length", len(validation_set))
    print("Testing set of length", len(testing_set))

    # Shuffle the data
    random.shuffle(training_set)
    random.shuffle(validation_set)
    random.shuffle(testing_set)

    print(f"Sample training data: {training_set[0]}")

    print("Writing to training csv")
    # Write the data to a new csv file in each respective directory
    with open(
        f"{LANDMARKS_DIR}/train/landmarks_train.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        for row in training_set:
            writer.writerow(row)

    print("Writing to validation csv")
    with open(
        f"{LANDMARKS_DIR}/validation/landmarks_validation.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        for row in validation_set:
            writer.writerow(row)

    print("Writing to testing csv")
    with open(
        f"{LANDMARKS_DIR}/test/landmarks_test.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        for row in testing_set:
            writer.writerow(row)

    # Add headers to the csv files
    add_landmarks_header_to_csv(
        training=f"{LANDMARKS_DIR}/train/landmarks_train.csv",
        validation=f"{LANDMARKS_DIR}/validation/landmarks_validation.csv",
        testing=f"{LANDMARKS_DIR}/test/landmarks_test.csv",
    )


def chart_distribution_mnist():
    # Load the sign mnist dataset
    mnist_train = "dataset/csv_dataset/pixel/train/sign_mnist_train.csv"
    mnist_test = "dataset/csv_dataset/pixel/test/sign_mnist_test.csv"

    train_df = pd.read_csv(mnist_train)
    test_df = pd.read_csv(mnist_test)

    # Substitute the numbers with the corresponding letters
    train_df["label"] = train_df["label"].apply(lambda x: number_to_letter[x])
    test_df["label"] = test_df["label"].apply(lambda x: number_to_letter[x])
    # Graph the distribution of the labels
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # Sort the labels and substitute the numbers with the corresponding letters
    train_df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Training Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.subplot(1, 2, 2)
    test_df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Testing Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def chart_distribution():
    # Load the datasets
    train_df = pd.read_csv(CSV_DATASET_TRAINING)
    val_df = pd.read_csv(CSV_DATASET_VALIDATION)
    test_df = pd.read_csv(CSV_DATASET_TESTING)

    # Substitute the numbers with the corresponding letters
    train_df["label"] = train_df["label"].apply(lambda x: letter_predictions[x])
    val_df["label"] = val_df["label"].apply(lambda x: letter_predictions[x])
    test_df["label"] = test_df["label"].apply(lambda x: letter_predictions[x])

    # Graph the distribution of the labels
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    # Sort the labels
    train_df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Training Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.subplot(1, 3, 2)
    val_df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Validation Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.subplot(1, 3, 3)
    test_df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Testing Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("dataset_distribution.png")

    train_df = pd.read_csv(f"{LANDMARKS_DIR}/train/landmarks_train.csv")
    validation_df = pd.read_csv(f"{LANDMARKS_DIR}/validation/landmarks_validation.csv")
    testing_df = pd.read_csv(f"{LANDMARKS_DIR}/test/landmarks_test.csv")

    # Substitute the numbers with the corresponding letters
    train_df["label"] = train_df["label"].apply(lambda x: letter_predictions[x])
    validation_df["label"] = validation_df["label"].apply(
        lambda x: letter_predictions[x]
    )
    testing_df["label"] = testing_df["label"].apply(lambda x: letter_predictions[x])

    # Graph the distribution of the labels
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    # Sort the labels
    train_df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Training Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.subplot(1, 3, 2)
    validation_df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Validation Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.subplot(1, 3, 3)
    testing_df["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Testing Data")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("landmarks_distribution.png")


def augment_dataset(dataset: str = IMAGES_DIR):
    for subdir, _, files in os.walk(dataset):
        print(f"Processing {subdir}")
        count = len(files) - 1
        letter = subdir.split("\\")[-1][0]
        print(f"Current count of files: {len(files)}")
        # Loop through the files
        for file in files.copy():
            if file.endswith(".png"):
                # Load the image
                img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_COLOR)
                new_name = os.path.join(subdir, f"{letter.upper()}{count}.png")
                # Rotate the image with a random angle
                angles = [random.randint(-5, 5) for _ in range(5)]
                for angle in angles:
                    new_name = os.path.join(subdir, f"{letter.upper()}{count}.png")
                    img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_COLOR)
                    # Zoom in the image by 20%
                    old_size = img.shape[:2]
                    img = cv2.resize(img, (0, 0), fx=1.2, fy=1.2)
                    # Get the center of the image
                    center = (img.shape[1] // 2, img.shape[0] // 2)
                    # Rotate the image
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
                    img = cv2.warpAffine(
                        img, rotation_matrix, (img.shape[1], img.shape[0])
                    )
                    # Crop the image to the original size
                    img = cv2.getRectSubPix(
                        img, (old_size[1], old_size[0]), (img.shape[1] // 2, img.shape[0] // 2)
                    )
                    # Save the image
                    cv2.imwrite(new_name, img)
                    count += 1
                # Do a random contrast and brightness change
                for _ in range(5):
                    new_name = os.path.join(subdir, f"{letter.upper()}{count}.png")
                    img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_COLOR)
                    contrast = random.uniform(0.5, 1.5)
                    brightness = random.randint(-10, 10)
                    img = cv2.addWeighted(
                        img, contrast, np.zeros(img.shape, img.dtype), 0, brightness
                    )
                    cv2.imwrite(new_name, img)
                    count += 1
        print(f"New count of files: {count + 1}")


def reset_dataset():
    for subdir, _, files in os.walk("dataset/images"):
        for file in files:
            if file.endswith(".png"):
                # Check the index of the file
                index = file.split("\\")[-1].split(".")[0][1:]
                try:
                    index = int(index)
                except ValueError:
                    continue
                if index > 70:
                    os.remove(os.path.join(subdir, file))
                    print(f"Removed {file}")

if __name__ == "__main__":
    # augment_dataset()
    # img_set_dir_to_csv("dataset/images")
    reduce_dataset(6)
    # normalize_dataset(True)
