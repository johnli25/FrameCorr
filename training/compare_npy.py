import numpy as np

def compare_npy_arrays(file1, file2):
    """Compares two NumPy arrays loaded from .npy files.

    Args:
        file1 (str): Path to the first .npy file.
        file2 (str): Path to the second .npy file.

    Returns:
        None. Prints comparison results.
    """

    try:
        arr1 = np.load(file1,allow_pickle=True)
        arr2 = np.load(file2,allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return

    # 1. Check if shapes match
    if arr1.shape != arr2.shape:
        print("Arrays have different shapes.")
        return

    # 2. Check if all elements are equal (exact match)
    if np.array_equal(arr1, arr2):
        print("Arrays are identical.")
        return

    # 3. Detailed comparison for non-identical arrays
    print("Arrays are different. Element-wise differences:")
    diff_mask = arr1 != arr2
    where_different = np.where(diff_mask)

    for index in zip(*where_different):
        print(f"At index {index}:  {arr1[index]} != {arr2[index]}")

# Example usage
compare_npy_arrays("rcv_data.npy", "sender_data.npy") 