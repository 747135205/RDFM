import os
import shutil

# List of directories to be deleted
directories_to_delete = [
 '*'
]

for directory in directories_to_delete:
    try:
        # Check if the directory exists
        if os.path.exists(directory):
            # Delete the directory and all its contents
            shutil.rmtree(directory)
            print(f"Successfully deleted: {directory}")
        else:
            print(f"Directory does not exist: {directory}")
    except Exception as e:
        print(f"Error deleting {directory}: {e}")
