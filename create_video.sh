#!/bin/bash


# place this file into folder with frames to generate videos
PARENT_DIR=$(pwd)
for dir in */ ; do
    cd "$dir"
    DIR_NAME=${dir%/}

    # Check if there are any frame files in the directory
    if ls frame_*.png 1> /dev/null 2>&1; then
        # Check if a video with the directory's name already exists in PARENT_DIR
        if [ ! -f "${PARENT_DIR}/${DIR_NAME}.mp4" ]; then
            ffmpeg -framerate 23 -i frame_%d.png -c:v libx264 -pix_fmt yuv420p "${PARENT_DIR}/${DIR_NAME}.mp4"
        else
            echo "Video ${DIR_NAME}.mp4 already exists in parent directory. Skipping..."
        fi
    else
        echo "No frame files found in $DIR_NAME. Skipping..."
    fi

    cd "$PARENT_DIR"
done
