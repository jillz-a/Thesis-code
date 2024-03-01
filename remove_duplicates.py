import os
import re
import sys

# Get the absolute path of the project directory
project_path = os.path.dirname(os.path.abspath(os.path.join((__file__))))
# Add the project directory to sys.path if it's not already present
if project_path not in sys.path:
    sys.path.append(project_path)


def remove_duplicate_files(folder_path):
    deleted = 0
    seen = set()  # To keep track of middle parts that have been seen
    for file in os.listdir(folder_path):
        # Check if the file matches the desired pattern
        match = re.match(r'cf_(\d+)_\d+\.csv', file)
        if match:
            middle_part = match.group(1)  # Extract the middle part of the filename
            if middle_part in seen:
                # If we've seen this middle part before, delete the file
                os.remove(os.path.join(folder_path, file))
                print(f"Deleted {file}")
                deleted += 1
            else:
                # Otherwise, remember this middle part
                seen.add(middle_part)

    return deleted

# Example usage
folder_path = f'{project_path}/DiCE_uncertainty/BNN_cf_results/inputs/FD001/denoised'
deleted = remove_duplicate_files(folder_path)
print(f'Deleted {deleted} files')
