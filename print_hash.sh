# for all files in clustering directory

for file in `find ./clustering/images/blob/*`; do
    # if [ -f "$file" ]; then
    # echo "File: $file"
    md5sum "$file"
    # fi
done

for file in `find ./clustering/images/blob2_small/*`; do
    # if [ -f "$file" ]; then
    # echo "File: $file"
    md5sum "$file"
    # fi
done

# # for all files in placement directory
for file in `find ./placement/images/*`; do
    if [ -f "$file" ]; then
        # echo "File: $file"
        md5sum "$file"
    fi
done